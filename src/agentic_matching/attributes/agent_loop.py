"""Bounded (N=3 by default) agentic loop for matching-attribute construction -- the
attribute-stage counterpart to blocking/agent_loop.py. The LLM proposes an attribute
set (seeded from seed_rules.SEED_ATTRIBUTES when available, e.g. yogurt; from scratch
otherwise, e.g. beans), it's checked for pairwise correlation on the block's own
population, and the LLM revises based on the correlation flags — stopping early once no
attribute pair is flagged (or the set stops changing). Each round is logged to
data/artifacts/ for SME review, and the final attribute set is persisted as a versioned
JSON artifact under attributes/generated/<block>/ — this is the "versioned, testable
artifact" the extraction logic is defined by (attributes are keyword-rule based,
computed uniformly by attributes/rules.py's apply_attribute, so a declarative versioned
JSON plays the role generated Python code would for a richer rule language).
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from agentic_matching.attributes.metrics import check_correlations
from agentic_matching.attributes.revision import revise_attributes
from agentic_matching.attributes.rules import attribute_set_signature, compute_attribute_values, filter_valid_attributes
from agentic_matching.attributes.seed_rules import get_seed_attribute_notes, get_seed_attributes
from agentic_matching.blocking.metrics import CANONICAL_BLOCK_TERMS
from agentic_matching.config import BLOCKS_DIR, agent_loop_settings, new_run_id, round_temperature, run_artifacts_dir
from agentic_matching.llm.client import ChatClient, get_llm_client
from agentic_matching.llm.prompts import build_attribute_prompt

log = logging.getLogger(__name__)

GENERATED_DIR = Path(__file__).resolve().parent / "generated"

_TOKEN_RE = re.compile(r"[a-z]+")

# Generic connector/context words -- never useful as a matching attribute regardless of
# how often they split the block. Deliberately small and boring (unlike blocking's
# genericity problem, an over-broad attribute candidate isn't a memory risk, just noise
# that correlation_check/degeneracy_check downstream can catch) -- see
# _candidate_boolean_terms's docstring for why within-block frequency, not catalog-wide
# frequency, is the relevant signal here.
_ATTRIBUTE_MINING_STOPWORDS = {
    "the", "and", "or", "of", "with", "a", "an", "in", "to", "for", "as", "is", "on",
    "not", "no", "ns", "nfs", "type", "from", "added", "fat", "other", "than", "made",
    "based", "ready",
    # OFF's search_text folds in its categories_tags taxonomy (e.g.
    # "en:plant-based-foods-and-beverages en:legumes-and-their-products"), which
    # contributes real signal for some concepts (e.g. "meat" mainly shows up via
    # category tags like "en:meat-based-products" rather than literally in product
    # names) but also boilerplate scaffolding words common to nearly every OFF category.
    "plant", "foods", "beverages", "products", "their", "food", "legumes",
}


@dataclass
class AttributeRound:
    round: int
    attributes: list[dict[str, Any]]
    correlation_flags: list[dict[str, Any]]
    rationale: str


def select_final_attributes(rounds: list[AttributeRound]) -> list[dict[str, Any]]:
    """Pick which round's attribute set to persist as attributes/generated/<block>/
    latest.json -- NOT just the last round. Hitting max_rounds before the
    correlation-flag count returns to 0 means the LAST round isn't necessarily the
    best one produced (same principle as blocking/agent_loop.py's _select_final_rule
    and linking/agent_loop.py's select_best_round -- this loop's only quality signal is
    correlation_flags, since it never trains a real model against real match outcomes;
    see this module's docstring for why that's a deliberately cheaper, earlier check).

    Scored as (non-empty, fewest correlation flags, round index) in that priority
    order: an empty attribute set is disqualified outright (trivially "0 correlation
    flags" but useless), then prefer fewer flags, then the most-revised (latest) round
    among ties.
    """

    def _score(r: AttributeRound) -> tuple[bool, int, int]:
        return (bool(r.attributes), -len(r.correlation_flags), r.round)

    return max(rounds, key=_score).attributes


def _load_block(block_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fndds_path = BLOCKS_DIR / f"{block_name}_fndds.parquet"
    catalog_path = BLOCKS_DIR / f"{block_name}_catalog.parquet"
    if not fndds_path.exists() or not catalog_path.exists():
        raise FileNotFoundError(
            f"Block subsets not found for '{block_name}'. Run the blocking agent "
            "(scripts/05_run_blocking_agent.py) first."
        )
    return pd.read_parquet(fndds_path), pd.read_parquet(catalog_path)


def _sample_pairs(fndds_df: pd.DataFrame, catalog_df: pd.DataFrame, n: int = 30) -> list[dict[str, Any]]:
    rng = random.Random(42)
    fndds_sample = fndds_df["description"].dropna().tolist()
    catalog_sample = catalog_df["product_name"].dropna().tolist()
    rng.shuffle(fndds_sample)
    rng.shuffle(catalog_sample)
    k = min(n, len(fndds_sample), len(catalog_sample)) or 0
    return [{"fndds_description": f, "catalog_product_name": o} for f, o in zip(fndds_sample[:k], catalog_sample[:k])]


def _field_stats(fndds_df: pd.DataFrame, catalog_df: pd.DataFrame, k: int = 15) -> dict[str, Any]:
    """Most common real values of each side's categorical fields *within this block's
    population* (not the whole catalog -- see profiling.py for that, used by the
    blocking stage instead). Grounds categorical attribute proposals (e.g. fat_level's
    categories, beans' bean_type categories) in values that actually occur here, rather
    than the LLM inventing plausible-sounding category names/keywords from world
    knowledge alone."""
    stats: dict[str, Any] = {}
    if "wweia_food_category_description" in fndds_df.columns:
        counts = fndds_df["wweia_food_category_description"].dropna().value_counts().head(k)
        stats["fndds_wweia_food_category"] = [{"value": v, "count": int(c)} for v, c in counts.items()]
    if "categories_tags" in catalog_df.columns:
        tags = pd.Series([t for tags in catalog_df["categories_tags"].dropna() for t in tags])
        counts = tags.value_counts().head(k)
        stats["catalog_categories_tags"] = [{"value": v, "count": int(c)} for v, c in counts.items()]
    if "brands" in catalog_df.columns:
        brands = catalog_df["brands"].dropna()
        counts = brands[brands.str.strip() != ""].value_counts().head(k)
        stats["catalog_brands"] = [{"value": v, "count": int(c)} for v, c in counts.items()]
    return stats


def _doc_freq(texts: list[Any], min_len: int = 4) -> tuple[Counter, int]:
    counts: Counter[str] = Counter()
    n = 0
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        n += 1
        for tok in {tok for tok in _TOKEN_RE.findall(t.lower()) if len(tok) >= min_len}:
            counts[tok] += 1
    return counts, n


def _candidate_boolean_terms(
    fndds_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    block_name: str,
    min_frac: float = 0.03,
    max_frac: float = 0.9,
    k: int = 15,
) -> list[dict[str, Any]]:
    """Tokens that split *this block's own population* into a meaningful
    minority/majority on at least one side (not near-0% or near-100%, which wouldn't
    discriminate anything) -- candidate boolean matching attributes (e.g. has_meat,
    with_rice for beans), mined directly from the block's own free text rather than
    invented from scratch. This is the attribute-side counterpart to blocking's keyword
    mining, but the direction of the frequency check is the opposite: blocking rejects
    tokens too common *catalog-wide* (a memory/precision risk at OFF's ~4.66M-row
    scale); this instead wants tokens that are moderately common *within the block*
    (a near-0%-or-100% split can't discriminate matches from non-matches at all).

    Mines FNDDS's `description` and OFF's `search_text` (product name + categories_tags
    + brands -- see catalog_text.py) -- some real signal (e.g. "meat") shows up mainly via
    OFF's categories_tags taxonomy rather than literally in the product name, so it's
    worth including, at the cost of some category-taxonomy boilerplate that
    `_ATTRIBUTE_MINING_STOPWORDS` filters out. Ranks candidates by the *minimum* of
    their two sides' fractions (i.e. prefers terms with corroborating signal on BOTH
    sides) rather than by raw combined frequency -- FNDDS's "beans" block catches some
    unrelated mixed-dish categories (pasta, potato, sandwich dishes that happen to also
    mention beans), whose incidental vocabulary ("potato", "sandwich", ...) would
    otherwise dominate a pure-frequency ranking despite being one-sided noise.

    Does NOT attempt to group synonyms/translations across languages (e.g. "pork" vs
    "porc") -- recognizing those as the same concept is exactly the kind of reasoning a
    real LLM should do better than frequency counting; a mined candidate's keyword list
    is just the literal token itself, applied identically on both sides. This is also
    why single mined tokens (e.g. "meat" alone) cover less than a hand-curated
    multi-synonym keyword list (e.g. "pork"/"beef"/"bacon"/"sausage"/"ham"/"meat")
    would -- a real LLM's proposal should be more complete than this mock's.
    """
    canonical_terms = CANONICAL_BLOCK_TERMS.get(block_name, [block_name])
    exclude = set(_ATTRIBUTE_MINING_STOPWORDS)
    for t in canonical_terms:
        t = t.lower()
        exclude |= {t, t + "s"}

    fndds_freq, n_fndds = _doc_freq(fndds_df["description"].tolist())
    catalog_freq, n_catalog = _doc_freq(catalog_df["search_text"].tolist())

    def in_band(frac: float) -> bool:
        return min_frac <= frac <= max_frac

    candidates = []
    for tok in set(fndds_freq) | set(catalog_freq):
        if tok in exclude:
            continue
        fndds_fraction = (fndds_freq.get(tok, 0) / n_fndds) if n_fndds else 0.0
        catalog_fraction = (catalog_freq.get(tok, 0) / n_catalog) if n_catalog else 0.0
        if in_band(fndds_fraction) or in_band(catalog_fraction):
            candidates.append({"term": tok, "fndds_fraction": round(fndds_fraction, 4), "catalog_fraction": round(catalog_fraction, 4)})
    candidates.sort(key=lambda c: (-min(c["fndds_fraction"], c["catalog_fraction"]), -(c["fndds_fraction"] + c["catalog_fraction"])))
    return candidates[:k]


def _pooled_values(attrs: list[dict[str, Any]], fndds_df: pd.DataFrame, catalog_df: pd.DataFrame) -> pd.DataFrame:
    fndds_vals = compute_attribute_values(attrs, fndds_df["fndds_search_text"].tolist(), side="fndds")
    catalog_vals = compute_attribute_values(attrs, catalog_df["search_text"].tolist(), side="catalog")
    fndds_frame = pd.DataFrame(fndds_vals)
    catalog_frame = pd.DataFrame(catalog_vals)
    return pd.concat([fndds_frame, catalog_frame], ignore_index=True)


def run_attribute_agent(
    block_name: str, client: ChatClient | None = None, run_dir: Path | None = None
) -> list[AttributeRound]:
    """`run_dir` -- where this run's artifacts (attributes_round<N>.json) are written;
    data/artifacts/<block_name>/<run_id>/. Defaults to a fresh one (own new_run_id())
    for a standalone call (e.g. scripts/06_run_matching_agent.py); outer_loop.py passes
    the SAME run_dir it's using for blocking/linking too, so one outer-loop
    invocation's artifacts all land in one run folder together. Unrelated to
    attributes/generated/<block>/latest.json (see this module's docstring) -- that's
    the persisted, currently-in-effect attribute definition linking reads, not a
    per-run diagnostic record, and isn't run-scoped."""
    run_dir = run_dir or run_artifacts_dir(block_name, new_run_id())
    client = client or get_llm_client()
    fndds_df, catalog_df = _load_block(block_name)
    sample_pairs = _sample_pairs(fndds_df, catalog_df)
    field_stats = _field_stats(fndds_df, catalog_df)
    candidate_terms = _candidate_boolean_terms(fndds_df, catalog_df, block_name)
    # For info_requests.py's "need more info" fulfillment (see attributes/revision.py)
    # -- plain text already loaded for this round, no extra query capability needed.
    fndds_texts = fndds_df["description"].tolist()
    catalog_texts = catalog_df["search_text"].tolist()

    rounds: list[AttributeRound] = []
    existing = get_seed_attributes(block_name)
    guidance = get_seed_attribute_notes(block_name)
    correlation_flags: list[dict[str, Any]] = []

    for round_idx in range(agent_loop_settings.max_rounds):
        temperature = round_temperature(round_idx, has_prior_state=existing is not None)
        log.info(
            "Generating matching attributes for '%s' round %d: asking the LLM to %s matching attributes (temperature=%.2f)",
            block_name,
            round_idx,
            "propose" if existing is None else "revise",
            temperature,
        )
        try:
            if existing is None:
                # Nothing to keep/drop/gap-fill yet -- the single "propose from
                # scratch" call (unchanged from before the decomposed-revision work).
                sys_p, user_p = build_attribute_prompt(
                    block_name,
                    sample_pairs,
                    field_stats=field_stats,
                    candidate_terms=candidate_terms,
                    guidance=guidance,
                )
                response = client.complete_json(sys_p, user_p, temperature=temperature)
                attrs = filter_valid_attributes(response["attributes"])
                rationale = response.get("rationale", "")
            else:
                # Decomposed revision (see attributes/revision.py) -- this loop has no
                # trained-model evaluation/error_examples (it never trains a real
                # linker), only correlation_flags.
                attrs, rationale, _gap_concept = revise_attributes(
                    client,
                    block_name,
                    existing,
                    correlation_flags=correlation_flags or None,
                    evaluation=None,
                    error_examples=None,
                    sample_pairs=sample_pairs,
                    field_stats=field_stats,
                    candidate_terms=candidate_terms,
                    guidance=guidance,
                    fndds_texts=fndds_texts,
                    catalog_texts=catalog_texts,
                    temperature=temperature,
                )
        except Exception:
            # See blocking/agent_loop.py's identical handling for why: a real LLM
            # backend can fail a round outright, and rounds already completed (and
            # already materialized on disk) shouldn't be thrown away over it.
            if rounds:
                log.exception(
                    "Round %d failed for block '%s'; stopping here and using the best "
                    "round completed so far.",
                    round_idx,
                    block_name,
                )
                break
            if existing is not None:
                log.exception(
                    "Round 0 failed for block '%s' with no completed rounds to fall "
                    "back to; using the seed attribute set unchanged instead.",
                    block_name,
                )
                _persist_generated(block_name, existing, version=0)
                return rounds
            raise

        values_df = _pooled_values(attrs, fndds_df, catalog_df)
        correlation_flags = check_correlations(values_df)

        rounds.append(
            AttributeRound(round=round_idx, attributes=attrs, correlation_flags=correlation_flags, rationale=rationale)
        )
        _write_artifact(run_dir, rounds[-1])

        # Content-sensitive, not name-only (see attribute_set_signature's docstring
        # for the real case a name-only check missed: a "redefine" that keeps an
        # attribute's name but changes its keywords looks like "no change" under a
        # name-only comparison, incorrectly stopping the loop on a real revision).
        prev_signature = attribute_set_signature(existing) if existing else None
        if not correlation_flags and prev_signature == attribute_set_signature(attrs):
            log.info("Attribute loop for '%s' stabilized after round %d", block_name, round_idx)
            break
        existing = attrs

    _persist_generated(block_name, select_final_attributes(rounds), version=len(rounds))
    client.log_usage_summary(label=f"{block_name} attributes, cumulative")
    return rounds


def _write_artifact(run_dir: Path, r: AttributeRound) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"attributes_round{r.round}.json"
    path.write_text(json.dumps(asdict(r), indent=2))
    log.info(
        "Wrote %s (%d attributes, %d correlation flags)",
        path,
        len(r.attributes),
        len(r.correlation_flags),
    )


def _persist_generated(block_name: str, attrs: list[dict[str, Any]], version: int) -> None:
    out_dir = GENERATED_DIR / block_name
    out_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = out_dir / f"v{version}.json"
    latest_path = out_dir / "latest.json"
    payload = json.dumps(attrs, indent=2)
    versioned_path.write_text(payload)
    latest_path.write_text(payload)
    log.info("Persisted final attribute set for '%s' -> %s", block_name, versioned_path)


def load_latest_attributes(block_name: str) -> list[dict[str, Any]]:
    path = GENERATED_DIR / block_name / "latest.json"
    if not path.exists():
        raise FileNotFoundError(f"No generated attributes for '{block_name}'; run the attribute agent first.")
    return json.loads(path.read_text())

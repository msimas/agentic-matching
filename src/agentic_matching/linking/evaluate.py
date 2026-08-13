"""Score a trained FNDDS<->OFF linkage model.

FNDDS<->OFF has no direct ground truth (FNDDS has no UPC), so two different reports are
produced:

1. Proxy precision/recall/F1 against the Branded<->OFF calibration holdout: the
   holdout's known-true (Branded fdc_id, catalog_code) pairs, restricted to this block, are
   scored using the block's *already-trained* comparison weights (same attribute
   definitions applied to Branded's text fields as the FNDDS-side stand-in -- see
   calibration.py's documented design), mixed with sampled non-match decoys so
   precision/recall are computable, then thresholded and compared to the known labels.
2. A plausibility report for the actual FNDDS<->OFF predictions (score distribution,
   top/bottom examples) -- offered because (1) is a proxy, not a direct measurement of
   this specific dataset pair.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from splink import DuckDBAPI, Linker

from agentic_matching.attributes.rules import compute_attribute_values
from agentic_matching.blocking.metrics import combined_exclude_keywords, exclude_predicate_sql, term_predicate_sql
from agentic_matching.blocking.seed_rules import get_seed_rule
from agentic_matching.config import CALIBRATION_DIR
from agentic_matching.linking import splink_model
from agentic_matching.linking.splink_model import stringify

log = logging.getLogger(__name__)


def _load_block_holdout(block_name: str) -> pd.DataFrame:
    branded_term_pred = term_predicate_sql("lower(coalesce(branded_food_category, ''))", block_name)
    catalog_term_pred = term_predicate_sql("lower(array_to_string(catalog_categories_tags, ' '))", block_name)
    # Same false-positive-category problem as metrics.pair_completeness (see its
    # docstring comment for the verified "onion ring" -> "French Fries, Potatoes &
    # Onion Rings" case) -- applied here too so the linking holdout doesn't score
    # against a ground truth that disagrees with the block's own known exclusions.
    # This function has no per-round `rule` to draw excludes from (unlike
    # pair_completeness), so it falls back to the seed rule -- the persisted,
    # SME-authored source of this domain knowledge (see blocking/seed_rules.py) -- if
    # one exists for this block; blocks with no seed rule get no extra exclusion here,
    # same as before this fix.
    #
    # TODO: a block with no seed_rules.json entry (or a seed whose excludes just
    # haven't caught up with what the block's actual rule has since learned) silently
    # gets LESS exclusion than its real, currently-materialized blocking rule would
    # apply -- verified real case: `beans` had no seed entry at all until one was
    # added retroactively (see seed_rules.json's own exclude_keywords_rationale for
    # beans' off-side), and even that seed initially missed 'jelly'/'vanilla bean'/
    # 'protein powder'/'crisps' (162 jelly-bean-candy holdout rows slipped through
    # unexcluded, verified by querying data/calibration/holdout.parquet directly)
    # because those excludes only existed in a real round's LEARNED rule (that run's
    # blocking_round1.json, under data/artifacts/beans/<run_id>/), not in the static
    # seed file. A more
    # systemic fix would draw excludes from the block's actual current/best rule
    # instead of (or in addition to) the seed -- deliberately not done yet: that
    # rule is itself LLM-proposed, so using it here to define the ground truth the
    # same LLM's attribute revisions get scored against risks a milder version of the
    # exact circularity `term_predicate_sql`'s docstring already avoids on the
    # inclusion side (an overly aggressive learned exclude list could shrink the
    # holdout toward the easier cases the current attributes already handle well,
    # quietly inflating holdout f1 with the model's own choices). Needs a persisted
    # canonical "final rule" artifact for blocking (blocking/agent_loop.py doesn't
    # write one today, unlike attributes/generated/<block>/latest.json) before this
    # is even possible, plus a real answer to the circularity question above -- see
    # README.md's "Known constraints" section for the fuller writeup.
    seed_rule = get_seed_rule(block_name) or {}
    excludes = combined_exclude_keywords(seed_rule)
    branded_exclude_pred = exclude_predicate_sql("lower(coalesce(branded_food_category, ''))", excludes)
    catalog_exclude_pred = exclude_predicate_sql("lower(array_to_string(catalog_categories_tags, ' '))", excludes)
    holdout_path = str(CALIBRATION_DIR / "holdout.parquet").replace("'", "''")
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT fdc_id, catalog_code, fndds_style_description, branded_food_category,
               catalog_product_name, catalog_categories_tags
        FROM read_parquet('{holdout_path}')
        WHERE ({branded_term_pred} OR {catalog_term_pred})
          AND NOT ({branded_exclude_pred} OR {catalog_exclude_pred})
        """
    ).df()
    con.close()
    return df


def _catalog_block_pool(block_name: str, limit: int = 20000) -> pd.DataFrame:
    """A pool of OFF records from this block to draw negative-pair decoys from."""
    from agentic_matching.config import BLOCKS_DIR

    return pd.read_parquet(BLOCKS_DIR / f"{block_name}_catalog.parquet").head(limit)


def build_eval_frames(
    block_name: str,
    attrs: list[dict[str, Any]],
    n_decoys_per_positive: int = 5,
    seed: int = 7,
    max_holdout_positives: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Returns (fndds_side_df, catalog_side_df, true_labels) where true_labels maps
    fndds unique_id -> the true off unique_id (only for positives).

    `score_against_holdout` scores this pair of frames with an exhaustive ("1=1")
    blocking rule -- fine only if both sides stay small. Branded Foods republishes many
    rows per GTIN, so a category's raw holdout can be tens of thousands of rows (e.g.
    ~28K for yogurt); left unbounded, the exhaustive cross join with the decoy pool
    produces on the order of a billion candidate pairs and exhausts memory. Capping the
    (deterministically sampled) positive count keeps the exhaustive comparison small
    while remaining a representative proxy sample.
    """
    holdout = _load_block_holdout(block_name)
    if len(holdout) > max_holdout_positives:
        holdout = holdout.sample(n=max_holdout_positives, random_state=seed).reset_index(drop=True)
    catalog_pool = _catalog_block_pool(block_name)

    fndds_vals = compute_attribute_values(attrs, holdout["fndds_style_description"].tolist(), side="fndds")
    fndds_df = pd.DataFrame(
        {
            "unique_id": holdout["fdc_id"].astype(str),
            "description": holdout["fndds_style_description"],
            "search_text": holdout["fndds_style_description"].str.lower(),
            **{name: [stringify(v) for v in vals] for name, vals in fndds_vals.items()},
        }
    )

    true_catalog_codes = set(holdout["catalog_code"].astype(str))
    decoy_pool = catalog_pool[~catalog_pool["code"].astype(str).isin(true_catalog_codes)]
    n_decoys = min(len(decoy_pool), n_decoys_per_positive * max(len(holdout), 1))
    decoys = decoy_pool.sample(n=n_decoys, random_state=seed) if n_decoys > 0 else decoy_pool.iloc[:0]

    positives_catalog = pd.DataFrame(
        {
            "code": holdout["catalog_code"].astype(str),
            "product_name": holdout["catalog_product_name"],
            "search_text": holdout["catalog_product_name"].fillna("").str.lower(),
        }
    )
    catalog_side = pd.concat(
        [positives_catalog, decoys[["code", "product_name", "search_text"]]], ignore_index=True
    ).drop_duplicates(subset="code")

    catalog_vals = compute_attribute_values(attrs, catalog_side["search_text"].tolist(), side="catalog")
    catalog_df = pd.DataFrame(
        {
            "unique_id": catalog_side["code"].astype(str),
            "description": catalog_side["product_name"],
            "search_text": catalog_side["search_text"],
            **{name: [stringify(v) for v in vals] for name, vals in catalog_vals.items()},
        }
    )

    true_labels = dict(zip(holdout["fdc_id"].astype(str), holdout["catalog_code"].astype(str)))
    return fndds_df, catalog_df, true_labels


def _build_holdout_predictions(
    block_name: str, attrs: list[dict[str, Any]], trained_settings: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame | None]:
    """Shared setup behind score_against_holdout and holdout_error_examples: build the
    labeled holdout eval frames and run one exhaustive prediction pass against them
    (the trained model's comparison weights, scored on the small, capped holdout
    sample -- see build_eval_frames' docstring for why "1=1" blocking is safe only
    here). Split out so both consumers pay for this once each, not twice."""
    fndds_df, catalog_df, true_labels = build_eval_frames(block_name, attrs)
    if fndds_df.empty or catalog_df.empty:
        return fndds_df, catalog_df, true_labels, None

    eval_settings = dict(trained_settings)
    eval_settings["blocking_rules_to_generate_predictions"] = ["1=1"]  # eval sets are small; exhaustive is fine

    linker = Linker([fndds_df, catalog_df], eval_settings, db_api=DuckDBAPI(), input_table_aliases=["fndds", "catalog"])
    preds = linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()
    # See splink_model.normalize_prediction_sides' docstring -- splink assigns _l/_r by
    # alphabetical table-name order, not input order, so this call is required to keep
    # every consumer below's "unique_id_r is always the catalog side" assumption true.
    preds = splink_model.normalize_prediction_sides(preds)
    return fndds_df, catalog_df, true_labels, preds


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower() if value not in (None, "") else ""


def _catalog_to_true_fndds(true_labels: dict[str, str]) -> dict[str, set[str]]:
    """Reverses true_labels (fdc_id -> its one true catalog_code) into catalog_code -> the set
    of fdc_id(s) genuinely known to be correct for it. A set, not a single id, because
    the holdout data itself sometimes labels more than one fdc_id as correct for the
    same catalog_code -- verified real case on the beans holdout: catalog_code
    0072036980786 is the labeled true partner for 5 different fdc_ids. Any one of them
    is an acceptable "correct" answer once we're picking one fndds match per OFF item
    (see score_against_holdout) -- not just whichever happened to be sampled first."""
    catalog_to_fndds: dict[str, set[str]] = {}
    for fndds_id, catalog_id in true_labels.items():
        catalog_to_fndds.setdefault(catalog_id, set()).add(fndds_id)
    return catalog_to_fndds


def _resolvable_ceiling(true_partners: dict[str, set[str]], fndds_desc_by_id: dict[str, str]) -> dict[str, Any]:
    """The best exact-id precision/recall/f1 ANY model could achieve on this holdout
    sample using text-derived signal alone, given via _resolvable_ceiling's own
    n_resolvable/n_ambiguous split -- exists because exact-id f1 isn't bounded by 1.0
    in practice (see score_against_holdout's docstring for why 0.0-1.0 is the
    mathematical range) but the holdout's own duplicate-description structure caps it
    far lower, and that cap is worth knowing so a low f1 doesn't get chased as if 1.0
    were reachable.

    An OFF item is "resolvable" if its true fdc_id(s)' description text doesn't also
    belong to some OTHER fdc_id that ISN'T one of its true partners (an "impostor") --
    a model can then in principle always prefer the true one. It's "ambiguous" if an
    impostor with byte-identical description text exists: no text signal (and every
    attribute this pipeline builds is text-derived) can tell the true fdc_id apart from
    the impostor, so even a perfect matcher is reduced to a coin flip there.

    Verified real magnitude on the beans holdout's actual eval sample (500 positives,
    seed=7): only 204 of 369 distinct true OFF items (55%) are resolvable -- the other
    165 (45%) have an impostor. A model scoring f1=0.068 there isn't 6.8% of the way to
    a perfect matcher; it's about 12% of the way to the ~0.55 ceiling this sample
    actually allows.

    Pure dict-in/dict-out (like _category_level_score) so it's testable without a real
    trained linker."""
    text_to_fndds_ids: dict[str, set[str]] = {}
    for fndds_id, desc in fndds_desc_by_id.items():
        text_to_fndds_ids.setdefault(_normalize_text(desc), set()).add(fndds_id)

    n_true = len(true_partners)
    n_resolvable = 0
    for true_ids in true_partners.values():
        texts = {_normalize_text(fndds_desc_by_id[t]) for t in true_ids if t in fndds_desc_by_id}
        impostors = set()
        for t in texts:
            impostors |= text_to_fndds_ids.get(t, set()) - true_ids
        if not impostors:
            n_resolvable += 1

    ceiling = n_resolvable / n_true if n_true else None
    return {
        "n_resolvable": n_resolvable,
        "n_ambiguous": n_true - n_resolvable,
        "max_achievable_precision": ceiling,
        "max_achievable_recall": ceiling,
        "max_achievable_f1": ceiling,
    }


def _category_level_score(
    predicted: dict[str, str], true_partners: dict[str, set[str]], fndds_desc_by_id: dict[str, str]
) -> dict[str, Any]:
    """Same precision/recall/f1 shape as score_against_holdout's exact-id scoring
    (below), but counting a prediction "correct" whenever the predicted FNDDS record's
    description text matches one of its true partner(s)' description text
    (case/whitespace-insensitive), not only when the predicted id is the literal gold
    fdc_id.

    `predicted` maps catalog_id -> the single fndds_id chosen as its best match (the same
    selection production actually ships -- see score_against_holdout). `true_partners`
    maps catalog_id -> the set of fdc_id(s) genuinely known correct for it (see
    _catalog_to_true_fndds).

    Exists because the exact-id gold pairs are far stricter than what this pipeline
    can, or needs to, resolve. Verified on the beans holdout: 94% of true pairs
    (12,045/12,767) share their FNDDS-side description text with at least one OTHER
    true pair -- e.g. "BLACK BEANS" is the gold description for 281 different
    (fdc_id, catalog_code) pairs. There is no text signal on either side that could
    distinguish which specific fdc_id is "the" one right answer among hundreds of
    identically-worded records, and this pipeline's category-level attribute set
    (bean_type, contains_meat, ...) was never designed to resolve that level of
    identity. Landing on a DIFFERENT but textually-identical fdc_id is a correct,
    nutrition-equivalent match for this pipeline's actual job (attaching a nutrition
    profile to a commercial product) even though exact-id scoring counts it as a dead
    loss.

    Pure dict-in/dict-out (like _holdout_error_examples) so it's testable without a
    real trained linker."""
    n_true = len(true_partners)
    n_pred = len(predicted)
    n_correct = 0
    for catalog_id, predicted_fndds_id in predicted.items():
        true_ids = true_partners.get(catalog_id)
        if not true_ids:
            continue
        if predicted_fndds_id in true_ids:
            n_correct += 1
            continue
        predicted_desc = fndds_desc_by_id.get(predicted_fndds_id)
        if predicted_desc is None:
            continue  # can't compare text we don't have -- neither correct nor incorrect, just excluded
        true_descs = {_normalize_text(fndds_desc_by_id[t]) for t in true_ids if t in fndds_desc_by_id}
        if _normalize_text(predicted_desc) in true_descs:
            n_correct += 1

    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_true if n_true else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n_category_correct": n_correct,
        "category_precision": precision,
        "category_recall": recall,
        "category_f1": f1,
    }


def score_against_holdout(
    block_name: str, attrs: list[dict[str, Any]], trained_settings: dict[str, Any], threshold: float = 0.5
) -> dict[str, Any]:
    """Scores using the SAME selection direction production actually ships with:
    best_match_per_catalog picks, per OFF/commercial product, its single best FNDDS
    candidate (drop_duplicates on unique_id_r, the OFF side) -- so this dedups on
    unique_id_r too. Previously this dedup'd on unique_id_l (the FNDDS side) instead,
    answering "if we pick FNDDS records' best OFF partner, do we find the right one" --
    a DIFFERENT question than the one production's actual selection logic asks, since
    a true pair can win reading FNDDS->best-OFF while losing its slot reading
    OFF->best-FNDDS (or vice versa), especially with the duplicate-description
    competition documented on _category_level_score. The old per-FNDDS-item numbers
    are gone entirely, not kept alongside these -- they measured a selection algorithm
    this pipeline doesn't actually run.

    The OFF side includes sampled decoys with no true fdc_id at all (see
    build_eval_frames) -- they compete for best-match slots on their OWN catalog_id (which
    is realistic: a real product's candidate pool always includes near-miss FNDDS
    records), but since they have no gold answer they're excluded from the
    precision/recall accounting entirely (see the catalog_id in catalog_to_true_fndds filter
    below), not counted as either correct or incorrect.

    Also returns category-level precision/recall/f1 (see _category_level_score)
    alongside the exact-id numbers; category_f1 is here for a human/LLM to read
    alongside f1 and understand how much of the "failure" is exact-id's inherent
    strictness on a duplicate-description-heavy holdout, not a real matching problem.
    And returns max_achievable_{precision,recall,f1} (see _resolvable_ceiling) -- the
    best exact-id score this SAMPLE'S own duplicate-description structure allows any
    model to reach, so f1 is read against a reachable ceiling, not against 1.0."""
    fndds_df, catalog_df, true_labels, preds = _build_holdout_predictions(block_name, attrs, trained_settings)
    if fndds_df.empty or catalog_df.empty:
        return {
            "n_holdout_positives": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "category_precision": None,
            "category_recall": None,
            "category_f1": None,
            "n_resolvable": None,
            "n_ambiguous": None,
            "max_achievable_precision": None,
            "max_achievable_recall": None,
            "max_achievable_f1": None,
        }

    catalog_to_true_fndds = _catalog_to_true_fndds(true_labels)

    # Best (highest-probability) FNDDS match per OFF-side unique_id -- mirrors
    # best_match_per_catalog's real production selection (see docstring above).
    best = preds.sort_values("match_probability", ascending=False).drop_duplicates(subset="unique_id_r")
    best = best[best["match_probability"] >= threshold]

    predicted = dict(zip(best["unique_id_r"], best["unique_id_l"]))  # catalog_id -> predicted fndds_id
    predicted_true_only = {catalog_id: fndds_id for catalog_id, fndds_id in predicted.items() if catalog_id in catalog_to_true_fndds}

    n_true = len(catalog_to_true_fndds)  # distinct OFF items with a genuine known-correct partner
    n_pred = len(predicted_true_only)
    n_correct = sum(1 for catalog_id, fndds_id in predicted_true_only.items() if fndds_id in catalog_to_true_fndds[catalog_id])

    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_true if n_true else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    fndds_desc_by_id = dict(zip(fndds_df["unique_id"], fndds_df["description"]))

    return {
        "n_holdout_positives": n_true,
        "n_predicted": n_pred,
        "n_correct": n_correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **_category_level_score(predicted_true_only, catalog_to_true_fndds, fndds_desc_by_id),
        **_resolvable_ceiling(catalog_to_true_fndds, fndds_desc_by_id),
    }


def _holdout_error_examples(
    preds: pd.DataFrame, true_labels: dict[str, str], threshold: float = 0.5, n: int = 5
) -> dict[str, list[dict[str, Any]]]:
    """Pure logic behind holdout_error_examples, split out so it's testable against a
    small synthetic `preds` frame without needing a real trained linker (same rationale
    as _attribute_discriminative_power's split from attribute_discriminative_power)."""
    cols = [c for c in ("unique_id_l", "unique_id_r", "description_l", "description_r", "match_probability") if c in preds.columns]
    if preds.empty or not true_labels:
        return {"false_positives": [], "false_negatives": []}

    # False positives: the model's best-scoring FNDDS match per OFF record (mirrors
    # production's real selection -- see score_against_holdout's docstring for why this
    # direction, not per-FNDDS-record, is the one that matters), confident (>=
    # threshold) but not one of that OFF record's true partner(s) -- the n
    # highest-probability such mistakes, i.e. what's most urgently over-matching right
    # now.
    catalog_to_true_fndds = _catalog_to_true_fndds(true_labels)
    best = preds.sort_values("match_probability", ascending=False).drop_duplicates(subset="unique_id_r")
    confident = best[best["match_probability"] >= threshold]
    if confident.empty:
        # DataFrame.apply(axis=1) on an empty frame returns an empty DataFrame, not a
        # boolean Series, which breaks confident[is_wrong] below -- skip straight to
        # "no false positives" instead.
        false_positives = confident
    else:
        is_wrong = confident.apply(
            lambda r: r["unique_id_l"] not in catalog_to_true_fndds.get(r["unique_id_r"], set()), axis=1
        )
        false_positives = confident[is_wrong].sort_values("match_probability", ascending=False).head(n)

    # False negatives: for every TRUE calibration pair, what score did the model
    # actually give *that specific pair* (not whatever else won for that FNDDS
    # record) -- merged, not looped, so this stays fast even with hundreds of holdout
    # positives. A true pair that never became a candidate at all (no row in `preds`)
    # is a blocking-shaped gap, not an attribute one, so it's left out here on purpose
    # -- outer_loop.diagnose_blocking_problem is where that concern belongs.
    true_pairs = pd.DataFrame(list(true_labels.items()), columns=["unique_id_l", "true_unique_id_r"])
    merged = true_pairs.merge(
        preds, left_on=["unique_id_l", "true_unique_id_r"], right_on=["unique_id_l", "unique_id_r"], how="inner"
    )
    false_negatives = merged[merged["match_probability"] < threshold].sort_values("match_probability").head(n)

    false_positive_records = false_positives[cols].to_dict(orient="records")
    if false_positive_records:
        # Tag each false positive with whether it's actually resolvable, or a known
        # metric artifact no new attribute could ever fix -- verified real case:
        # "BLACK BEANS" predicted-matched to "BLACK BEANS" at 0.99 confidence, flagged
        # wrong only because it wasn't the arbitrary fdc_id the holdout happened to
        # label as truth for that catalog_code. Without this tag, identify_gap (see
        # llm/prompts.py's _GAP_SYSTEM) has no way to tell that apart from a genuine
        # miss like "CADIA REFRIED VEGETARIAN BEANS" vs "Vegetarian Refried Pinto
        # Beans" (different text, a real gap an attribute could plausibly close), and
        # can waste a round proposing a new attribute to fix something no keyword rule
        # ever could. Same collision logic as _resolvable_ceiling, applied per-example
        # instead of aggregated.
        fndds_desc_by_id = dict(zip(preds["unique_id_l"], preds["description_l"]))
        for record in false_positive_records:
            true_ids = catalog_to_true_fndds.get(record["unique_id_r"], set())
            true_descs = {_normalize_text(fndds_desc_by_id[t]) for t in true_ids if t in fndds_desc_by_id}
            record["same_text_as_true_partner"] = bool(true_descs) and _normalize_text(record["description_l"]) in true_descs

    return {
        "false_positives": false_positive_records,
        "false_negatives": false_negatives[cols].to_dict(orient="records") if not false_negatives.empty else [],
    }


def holdout_error_examples(
    block_name: str, attrs: list[dict[str, Any]], trained_settings: dict[str, Any], threshold: float = 0.5, n: int = 5
) -> dict[str, list[dict[str, Any]]]:
    """Concrete false-positive/false-negative example PAIRS from the calibration
    holdout -- the piece score_against_holdout's aggregate precision/recall/f1 (and
    attribute_discriminative_power's per-attribute agreement rates) can't provide:
    which SPECIFIC records the current attribute set gets wrong, and what they actually
    look like, so the attribute-revision LLM can reason about what new signal would fix
    a real mistake instead of only seeing that mistakes exist in aggregate.

    "false_positives": each OFF record's best FNDDS match, confidently (>= `threshold`)
    predicted, that is NOT one of that OFF record's true calibration partner(s) --
    same per-OFF-record selection production's best_match_per_catalog actually uses (see
    score_against_holdout's docstring). Each carries a "same_text_as_true_partner" flag
    -- True means the predicted (wrong) record's description text is byte-identical to
    the true partner's, i.e. this is a duplicate-description metric artifact no
    attribute could ever fix (see _holdout_error_examples), not a real gap.
    "false_negatives": true calibration pairs the model scored below `threshold` for
    that specific pair. Both capped at `n`, ranked by how confidently wrong (false
    positives) or how far short of the threshold (false negatives) they are -- the
    most actionable examples first.
    """
    _, _, true_labels, preds = _build_holdout_predictions(block_name, attrs, trained_settings)
    if preds is None:
        return {"false_positives": [], "false_negatives": []}
    return _holdout_error_examples(preds, true_labels, threshold=threshold, n=n)


def _attribute_discriminative_power(
    fndds_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    true_labels: dict[str, str],
    attrs: list[dict[str, Any]],
    n_decoys_per_positive: int = 5,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Pure value-comparison logic behind `attribute_discriminative_power`, split out
    so it's testable against small synthetic frames without needing the real
    calibration parquet (same rationale as profiling._rank_terms's split from
    high_frequency_terms)."""
    if fndds_df.empty or catalog_df.empty or not true_labels:
        return []

    fndds_by_id = fndds_df.set_index("unique_id")
    catalog_by_id = catalog_df.set_index("unique_id")
    catalog_ids = list(catalog_by_id.index)
    rng = random.Random(seed)

    results = []
    for attr in attrs:
        name = attr["name"]
        if name not in fndds_by_id.columns or name not in catalog_by_id.columns:
            continue
        true_agree = []
        decoy_agree = []
        for fndds_id, true_catalog_id in true_labels.items():
            if fndds_id not in fndds_by_id.index or true_catalog_id not in catalog_by_id.index:
                continue
            l_val = fndds_by_id.loc[fndds_id, name]
            true_agree.append(l_val == catalog_by_id.loc[true_catalog_id, name])
            decoy_ids = [oid for oid in rng.sample(catalog_ids, min(n_decoys_per_positive, len(catalog_ids))) if oid != true_catalog_id]
            decoy_agree.extend(l_val == catalog_by_id.loc[oid, name] for oid in decoy_ids)
        if not true_agree or not decoy_agree:
            continue
        results.append(
            {
                "attribute": name,
                "agreement_rate_true_pairs": round(sum(true_agree) / len(true_agree), 3),
                "agreement_rate_decoy_pairs": round(sum(decoy_agree) / len(decoy_agree), 3),
                "n_true_pairs": len(true_agree),
                "n_decoy_pairs": len(decoy_agree),
            }
        )
    return results


def attribute_discriminative_power(
    block_name: str, attrs: list[dict[str, Any]], n_decoys_per_positive: int = 5, seed: int = 13
) -> list[dict[str, Any]]:
    """For each attribute, compare its agreement rate on known-true (Branded<->OFF
    calibration) pairs against its agreement rate on random non-match (decoy) pairs from
    the same holdout sample -- a query-derived signal (run once per revision round, fed
    back into build_attribute_prompt's `evaluation`) showing the attribute-generation
    LLM which of its OWN proposed attributes actually discriminate matches from
    non-matches, complementing correlation_check's attribute-vs-attribute check (which
    only catches redundancy between attributes, not whether any of them are useful at
    all) and the aggregate holdout f1 (which conflates every attribute's contribution
    into one number). An attribute agreeing at roughly the same rate on true pairs and
    decoys (e.g. "is_frozen: 0.90 true vs 0.88 decoy") isn't pulling any weight -- most
    records agree on it regardless of whether they're a real match; a big gap (e.g.
    "vegetable_type: 0.95 true vs 0.12 decoy") means it is.

    Pure value comparison against attribute columns splink_model already computes for
    build_eval_frames' output -- no model training/prediction involved, so this is cheap
    to run every round regardless of block size."""
    fndds_df, catalog_df, true_labels = build_eval_frames(block_name, attrs)
    return _attribute_discriminative_power(
        fndds_df, catalog_df, true_labels, attrs, n_decoys_per_positive=n_decoys_per_positive, seed=seed
    )


def best_match_per_catalog(predictions: pd.DataFrame, min_probability: float = 0.0) -> pd.DataFrame:
    """Collapse `predictions` (every FNDDS<->OFF candidate pair splink scored) down to
    the single best (highest match_probability) FNDDS record per OFF record.

    This is the actual deliverable this pipeline exists to produce: the real goal is
    attaching nutritional information (FNDDS) to commercial products (OFF today; a
    proprietary retail catalog like Circana in a real deployment -- nothing here is
    OFF-specific, it only assumes a `unique_id_r` column identifying the "commercial
    product" side, same as the rest of this module), and a commercial product should
    end up with AT MOST ONE nutrition profile attached, not several competing FNDDS
    candidates. `matches_<block>_round<N>.csv` (export_predictions_csv, below) is the
    SME review artifact showing every candidate pair for diagnosis; this is the
    downstream-consumable result.

    Deliberately does NOT also enforce uniqueness on the FNDDS side (`unique_id_l`) --
    one FNDDS nutrition profile legitimately applies to many distinct commercial
    products (e.g. many different "Black Beans" brands should all be able to attach to
    the same "Black beans, canned" FNDDS record), so that direction is not a duplicate
    to collapse the way multiple FNDDS candidates for one OFF record are.

    `min_probability` defaults to 0.0 (matching export_predictions_csv's own
    reasoning: confidence is conveyed by the match_probability column, not a cutoff
    baked into this function) -- pass a real threshold if you want the output itself
    gated on a decision boundary instead of left to the caller/reviewer.
    """
    if predictions.empty:
        return predictions
    df = predictions[predictions["match_probability"] >= min_probability]
    if df.empty:
        return df
    return df.sort_values("match_probability", ascending=False).drop_duplicates(subset="unique_id_r", keep="first")


def export_predictions_csv(
    predictions: pd.DataFrame, attrs: list[dict[str, Any]], out_path: Path, top_n: int | None = 5000
) -> int:
    """Write the top `top_n` predicted FNDDS<->OFF pairs (by `match_probability`,
    descending) to a CSV for direct SME inspection -- confidence is conveyed by the
    `match_probability` column itself, not a hard cutoff, so the file always shows the
    *best available* candidates even when none clear a nominal "confident match"
    threshold (a block/attribute combination too weak to produce any high-confidence
    match is exactly the kind of thing this review artifact needs to surface, not hide
    behind an empty file -- verified case: yogurt's `predict(threshold=0.5)` returned
    zero rows even though 304K real candidate pairs existed, topping out at
    match_probability 0.21). `top_n=None` exports every row `predictions` contains;
    pass an already-thresholded `predictions` if you want the file itself gated on a
    decision threshold instead. match_probability first so it opens sorted by
    confidence in any spreadsheet tool, one column pair per matching attribute so an
    SME can see exactly why a pair did or didn't agree. Returns the number of rows
    written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = ["match_probability", "match_weight", "unique_id_l", "description_l", "unique_id_r", "description_r"]
    attr_cols = [c for a in attrs for c in (f"{a['name']}_l", f"{a['name']}_r")]
    cols = [c for c in base_cols + attr_cols if c in predictions.columns]
    if predictions.empty:
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        return 0
    sorted_preds = predictions.sort_values("match_probability", ascending=False)
    if top_n is not None:
        sorted_preds = sorted_preds.head(top_n)
    out = sorted_preds[cols].rename(
        columns={
            "unique_id_l": "fndds_id",
            "description_l": "fndds_description",
            "unique_id_r": "catalog_code",
            "description_r": "catalog_product_name",
        }
    )
    out.to_csv(out_path, index=False)
    return len(out)


def plausibility_report(predictions: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    if predictions.empty:
        return {"n_pairs": 0}
    desc = predictions["match_probability"].describe().to_dict()
    top = predictions.sort_values("match_probability", ascending=False).head(top_n)
    bottom = predictions.sort_values("match_probability", ascending=True).head(top_n)
    cols = ["unique_id_l", "unique_id_r", "description_l", "description_r", "match_probability"]
    cols = [c for c in cols if c in predictions.columns]
    return {
        "n_pairs": len(predictions),
        "match_probability_summary": desc,
        "top_examples": top[cols].to_dict(orient="records"),
        "bottom_examples": bottom[cols].to_dict(orient="records"),
    }

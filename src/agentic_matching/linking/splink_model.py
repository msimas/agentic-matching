"""Build and train a per-block splink Linker (link_only, DuckDB backend) from the
block's FNDDS/OFF record subsets (data/blocks/) and its matching-attribute set
(attributes/generated/<block>/latest.json).

Both sides are projected to a *common* schema (unique_id, description, search_text,
one column per attribute) so splink comparisons/blocking rules can reference the same
column names on both inputs.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

from agentic_matching.attributes.rules import compute_attribute_values
from agentic_matching.config import BLOCKS_DIR

log = logging.getLogger(__name__)


def load_block_frames(block_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fndds_path = BLOCKS_DIR / f"{block_name}_fndds.parquet"
    off_path = BLOCKS_DIR / f"{block_name}_off.parquet"
    return pd.read_parquet(fndds_path), pd.read_parquet(off_path)


def stringify(v: Any) -> Any:
    if isinstance(v, bool):
        return str(v)
    return v


def _drop_unobservable_attrs(
    attrs: list[dict[str, Any]], fndds_vals: dict[str, list[Any]], off_vals: dict[str, list[Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[Any]], dict[str, list[Any]]]:
    """Drop any attribute whose computed value is None for *every* record on one side.

    Only a categorical attribute can produce this (rules.apply_attribute always
    resolves a boolean attribute to True/False, never None) -- it happens when the LLM
    (or a hand-authored seed) gives every category real off_keywords but leaves every
    category's fndds_keywords empty (or vice versa), e.g. verified against this
    project's "breaded_vegetables" block: has_breaded_vegetable_tag had off_keywords
    like "en:breaded-products" but fndds_keywords: [] on both categories, so every FNDDS
    record's value was None -- unconditionally, since apply_attribute never assigns a
    category name from an empty keyword list. A column that's 100% null on one side can
    never form an observed "exact match" or "not equal" comparison level (every pair
    falls into the always-fixed, uninformative "value is null" level instead), so
    splink's EM training has zero pairs to estimate that comparison's m/u from at all --
    it just logs "... not fully trained" and predict() silently falls back to defaults,
    which is a confusing failure mode to hit at predict time. Filtering the attribute
    out here, before it ever reaches splink, is cheap insurance against a bad
    LLM-proposed (or seed) attribute breaking training on any block, not just this one.
    """
    kept: list[dict[str, Any]] = []
    kept_fndds_vals: dict[str, list[Any]] = {}
    kept_off_vals: dict[str, list[Any]] = {}
    for attr in attrs:
        name = attr["name"]
        fndds_all_null = all(v is None for v in fndds_vals[name])
        off_all_null = all(v is None for v in off_vals[name])
        if fndds_all_null or off_all_null:
            log.warning(
                "Dropping matching attribute '%s': entirely null on the %s side (no "
                "category's %s_keywords ever matched a record there), so it can never "
                "produce an observed comparison level and would otherwise break "
                "splink's EM training for it. Fix the attribute's %s_keywords, or drop "
                "it, in the generated attribute set if this recurs.",
                name,
                "fndds" if fndds_all_null else "off",
                "fndds" if fndds_all_null else "off",
                "fndds" if fndds_all_null else "off",
            )
            continue
        kept.append(attr)
        kept_fndds_vals[name] = fndds_vals[name]
        kept_off_vals[name] = off_vals[name]
    return kept, kept_fndds_vals, kept_off_vals


def prepare_frames(
    block_name: str, attrs: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    fndds_raw, off_raw = load_block_frames(block_name)

    fndds_vals = compute_attribute_values(attrs, fndds_raw["fndds_search_text"].tolist(), side="fndds")
    off_vals = compute_attribute_values(attrs, off_raw["search_text"].tolist(), side="off")
    attrs, fndds_vals, off_vals = _drop_unobservable_attrs(attrs, fndds_vals, off_vals)

    fndds_df = pd.DataFrame(
        {
            "unique_id": fndds_raw["fdc_id"].astype(str),
            "description": fndds_raw["description"],
            "search_text": fndds_raw["fndds_search_text"],
            **{name: [stringify(v) for v in vals] for name, vals in fndds_vals.items()},
        }
    )
    off_df = pd.DataFrame(
        {
            "unique_id": off_raw["code"].astype(str),
            "description": off_raw["product_name"],
            "search_text": off_raw["search_text"],
            **{name: [stringify(v) for v in vals] for name, vals in off_vals.items()},
        }
    )
    return fndds_df, off_df, attrs


def build_comparisons(attrs: list[dict[str, Any]]) -> list[Any]:
    """One ExactMatch comparison per matching attribute -- deliberately NOT also a
    fuzzy `description` text-similarity comparison (a JaroWinklerAtThresholds on raw
    FNDDS description vs. OFF product name used to be appended here). Removed after
    verifying it showed the SAME degeneracy (both `collapsed` and `label_switching`)
    in effectively every trained round across every block tested (beans, yogurt,
    breaded_vegetables) -- structural, not a fluke: FNDDS uses standardized USDA
    nomenclature ("BLACK BEANS") while OFF uses branded consumer product names
    ("Trader Joe's Organic Black Beans, 15oz"), so raw full-string fuzzy similarity has
    little real correlation with whether two records actually match. Its practical
    contribution to match_weight was already being suppressed toward zero by
    nutrition_priors.dampen_description_weight (now removed along with it) -- this
    just stops fitting a comparison whose own trained parameters never showed real
    signal, and stops outer_loop.diagnose_blocking_problem from repeatedly
    misdiagnosing that non-signal as a re-blockable problem (verified: re-blocking
    never fixed it, in any of 3 blocks, across repeated outer-loop rounds). The
    `description` TEXT COLUMN itself is unaffected -- see build_linker's
    additional_columns_to_retain -- only its use as a splink comparison is gone, so
    error-example/plausibility-report display text is unchanged.

    Raises ValueError if `attrs` is empty. Before `description` was removed, this
    could never happen -- it was an always-present comparison, so a Linker never had
    zero of them even if every actual attribute got dropped. Verified real crash
    (LLM_DEVICE=ollama, qwen3:4b-instruct-2507-q4_K_M -- a much smaller/weaker model
    than this project's other verified backends, considerably more prone to producing
    attributes that end up entirely unobservable on one side and get filtered by
    prepare_frames' _drop_unobservable_attrs): an empty `comparisons` list makes
    splink generate SQL with an empty bayes-factor product term (`... * , 1e-300 ...`,
    a dangling comma where a multiplication operand should be) and fail deep inside
    its own SQL execution with a cryptic `SplinkException` / DuckDB parser error,
    instead of a clear, immediate, actionable failure at the actual point of the
    problem. Callers (see run_linking_agent's per-round try/except) should catch this
    specifically and stop that round/run gracefully rather than let it propagate as an
    opaque crash from inside splink internals."""
    if not attrs:
        raise ValueError(
            "build_comparisons: no matching attributes to build a comparison from -- "
            "splink cannot train a model with zero comparisons. This block has no "
            "usable attributes (all were dropped as unobservable, or none were ever "
            "defined) -- fix the attribute set upstream rather than retry training as-is."
        )
    return [cl.ExactMatch(attr["name"]) for attr in attrs]


def build_blocking_rules(attrs: list[dict[str, Any]]) -> list[Any]:
    """splink's own (internal) blocking, on top of the category block already applied
    upstream -- needed to keep pairwise comparison volume tractable. Block on the first
    categorical attribute if there is one (narrows a lot within-block), OR on a short
    text prefix as a catch-all so records lacking that attribute can still be compared."""
    categorical = [a["name"] for a in attrs if a["kind"] == "categorical"]
    rules = []
    if categorical:
        rules.append(block_on(categorical[0]))
    rules.append("substr(l.search_text, 1, 4) = substr(r.search_text, 1, 4)")
    return rules


def build_linker(
    block_name: str, attrs: list[dict[str, Any]], retain_intermediate_calculation_columns: bool = False
) -> tuple[Linker, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Returns (linker, fndds_df, off_df, attrs) -- `attrs` is echoed back because
    prepare_frames may drop attributes that turned out to be unobservable on one side
    (see `_drop_unobservable_attrs`); callers must train() with THIS returned list, not
    the one they passed in, or splink_model.train()'s EM blocking will reference a
    comparison column that was never added to the settings/dataframes."""
    fndds_df, off_df, attrs = prepare_frames(block_name, attrs)
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=build_comparisons(attrs),
        blocking_rules_to_generate_predictions=build_blocking_rules(attrs),
        # `description` is no longer a comparison (see build_comparisons' docstring),
        # but splink only retains a column in predictions if it's either used in a
        # comparison or explicitly listed here -- without this, description_l/
        # description_r would silently disappear from predictions, breaking every
        # display/diagnostic consumer that reads them (plausibility_report's top/bottom
        # examples, holdout_error_examples, evaluate.py's same_text_as_true_partner
        # tagging, ...). Retained purely for display now, not compared.
        additional_columns_to_retain=["description"],
        # Intermediate per-comparison-level calculation columns are only useful for
        # manually eyeballing a handful of pairs (e.g. linking/charts.py's waterfall
        # chart, which requires this to be True); retaining them for every predicted
        # pair roughly multiplies the width of the (already wide, block-sized) result
        # table and was a major contributor to this stage exhausting memory on this
        # box's block sizes (~1K x ~100K+) before the EM-blocking fix in train() below.
        # degeneracy_check.py and the main agent loop don't need row-level columns, so
        # this defaults to False for the production train/evaluate path.
        retain_intermediate_calculation_columns=retain_intermediate_calculation_columns,
    )
    linker = Linker(
        [fndds_df, off_df], settings, db_api=DuckDBAPI(), input_table_aliases=["fndds", "off"]
    )
    return linker, fndds_df, off_df, attrs


def linker_from_settings(fndds_df: pd.DataFrame, off_df: pd.DataFrame, settings: dict[str, Any]) -> Linker:
    """Build a Linker directly from an already-fully-specified settings dict (e.g. a
    trained model's exported settings, possibly adjusted -- see linking/
    nutrition_priors.py::apply_nutrition_priors) rather than from an attrs list +
    build_comparisons. No EM training is implied or run here; the caller is
    responsible for the settings already reflecting whatever trained/adjusted state
    it should. Same construction `evaluate.py::_build_holdout_predictions` already
    uses for the identical reason (scoring against a settings dict, not retraining)."""
    return Linker([fndds_df, off_df], settings, db_api=DuckDBAPI(), input_table_aliases=["fndds", "off"])


def train(linker: Linker, attrs: list[dict[str, Any]]) -> None:
    """Two-pass EM: each pass blocks on a different ATTRIBUTE column so every
    comparison gets an m-probability estimate from at least one pass (a comparison's m
    can't be estimated in a pass that blocks on that same column). Used to be
    attrs[0]/`description` (description was always the second pass's blocking column,
    via a short-text-prefix proxy rule) -- now attrs[0]/attrs[-1] when there are 2+
    attributes, since `description` is no longer a comparison at all (see
    build_comparisons' docstring) and so has nothing left to block on for EM's sake.

    With exactly ONE attribute, blocking that attribute's own EM pass on itself would
    be worse than merely leaving its m unestimated (the pre-existing edge case this
    used to accept): verified real crash (breaded_vegetables, a tiny ~11 FNDDS x 59 OFF
    block) -- the combined attribute-equality + search-text-prefix condition matched
    ZERO comparison pairs, which splink treats as a hard EMTrainingException, not a
    soft "untrained" outcome, taking down the whole round. The plain search-text-prefix
    rule (`prefix_rule`, no attribute constraint -- the same fallback already used for
    probability_two_random_records_match when there's no categorical attribute to
    block on) is used for this single pass instead: a much broader net, unlikely to
    produce zero pairs even on a tiny block, and it lets the sole attribute's own m
    actually get estimated (not blocking on it this time).

    Every EM/u-estimation blocking rule that blocks on an attribute is combined (AND)
    with the search-text prefix condition, so a low-cardinality/skewed attribute can't
    blow up block size on its own -- e.g. a boolean attribute that's False for the vast
    majority of records (common: is_greek, is_baby, ...) would otherwise pair up every
    False-valued FNDDS record with every False-valued OFF record, which on this
    project's block sizes (~1K FNDDS x ~100K OFF) produces tens of millions of candidate
    pairs from a single blocking rule and can exhaust memory.
    """
    prefix_rule = "substr(l.search_text, 1, 4) = substr(r.search_text, 1, 4)"
    categorical = [a["name"] for a in attrs if a["kind"] == "categorical"]
    br_estimate = block_on(categorical[0]) if categorical else prefix_rule
    linker.training.estimate_probability_two_random_records_match([br_estimate], recall=0.7)
    linker.training.estimate_u_using_random_sampling(max_pairs=2e6)

    comparison_cols = [a["name"] for a in attrs]
    if len(comparison_cols) > 1:
        em_blocking_rules = [
            block_on(comparison_cols[0], "substr(l.search_text, 1, 4)"),
            block_on(comparison_cols[-1], "substr(l.search_text, 1, 4)"),
        ]
    elif len(comparison_cols) == 1:
        em_blocking_rules = [prefix_rule]
    else:
        em_blocking_rules = []
    for rule in em_blocking_rules:
        linker.training.estimate_parameters_using_expectation_maximisation(rule)


def predict(linker: Linker, threshold: float = 0.5) -> pd.DataFrame:
    return linker.inference.predict(threshold_match_probability=threshold).as_pandas_dataframe()

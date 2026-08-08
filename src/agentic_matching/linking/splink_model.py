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

from agentic_matching.attributes.library import compute_attribute_values
from agentic_matching.config import BLOCKS_DIR

log = logging.getLogger(__name__)

TEXT_COMPARISON_THRESHOLDS = [0.9, 0.7]


def load_block_frames(block_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fndds_path = BLOCKS_DIR / f"{block_name}_fndds.parquet"
    off_path = BLOCKS_DIR / f"{block_name}_off.parquet"
    return pd.read_parquet(fndds_path), pd.read_parquet(off_path)


def stringify(v: Any) -> Any:
    if isinstance(v, bool):
        return str(v)
    return v


def prepare_frames(
    block_name: str, attrs: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fndds_raw, off_raw = load_block_frames(block_name)

    fndds_vals = compute_attribute_values(attrs, fndds_raw["fndds_search_text"].tolist(), side="fndds")
    off_vals = compute_attribute_values(attrs, off_raw["search_text"].tolist(), side="off")

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
    return fndds_df, off_df


def build_comparisons(attrs: list[dict[str, Any]]) -> list[Any]:
    comparisons = [cl.ExactMatch(attr["name"]) for attr in attrs]
    comparisons.append(cl.JaroWinklerAtThresholds("description", TEXT_COMPARISON_THRESHOLDS))
    return comparisons


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
    block_name: str, attrs: list[dict[str, Any]]
) -> tuple[Linker, pd.DataFrame, pd.DataFrame]:
    fndds_df, off_df = prepare_frames(block_name, attrs)
    settings = SettingsCreator(
        link_type="link_only",
        comparisons=build_comparisons(attrs),
        blocking_rules_to_generate_predictions=build_blocking_rules(attrs),
        # Intermediate per-comparison-level calculation columns are only useful for
        # manually eyeballing a handful of pairs; retaining them for every predicted
        # pair roughly multiplies the width of the (already wide, block-sized) result
        # table and was a major contributor to this stage exhausting memory on this
        # box's block sizes (~1K x ~100K+). degeneracy_check.py only reads the exported
        # m/u settings JSON, not row-level columns, so nothing downstream needs these.
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker(
        [fndds_df, off_df], settings, db_api=DuckDBAPI(), input_table_aliases=["fndds", "off"]
    )
    return linker, fndds_df, off_df


def train(linker: Linker, attrs: list[dict[str, Any]]) -> None:
    """Two-pass EM: each pass blocks on a different column so every comparison gets an
    m-probability estimate from at least one pass (a comparison's m can't be estimated
    in a pass that blocks on that same column).

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

    comparison_cols = [a["name"] for a in attrs] + ["description"]
    em_blocking_cols = [comparison_cols[0], comparison_cols[-1]] if len(comparison_cols) > 1 else comparison_cols
    for col in dict.fromkeys(em_blocking_cols):  # de-dup, preserve order
        if col == "description":
            rule = "substr(l.search_text, 1, 6) = substr(r.search_text, 1, 6)"
        else:
            rule = block_on(col, "substr(l.search_text, 1, 4)")
        linker.training.estimate_parameters_using_expectation_maximisation(rule)


def predict(linker: Linker, threshold: float = 0.5) -> pd.DataFrame:
    return linker.inference.predict(threshold_match_probability=threshold).as_pandas_dataframe()

"""Score a trained FNDDS<->OFF linkage model.

FNDDS<->OFF has no direct ground truth (FNDDS has no UPC), so two different reports are
produced:

1. Proxy precision/recall/F1 against the Branded<->OFF calibration holdout: the
   holdout's known-true (Branded fdc_id, off_code) pairs, restricted to this block, are
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
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from splink import DuckDBAPI, Linker

from agentic_matching.attributes.rules import compute_attribute_values
from agentic_matching.blocking.metrics import combined_exclude_keywords, exclude_predicate_sql, term_predicate_sql
from agentic_matching.blocking.seed_rules import get_seed_rule
from agentic_matching.config import CALIBRATION_DIR
from agentic_matching.linking.splink_model import stringify

log = logging.getLogger(__name__)


def _load_block_holdout(block_name: str) -> pd.DataFrame:
    branded_term_pred = term_predicate_sql("lower(coalesce(branded_food_category, ''))", block_name)
    off_term_pred = term_predicate_sql("lower(array_to_string(off_categories_tags, ' '))", block_name)
    # Same false-positive-category problem as metrics.pair_completeness (see its
    # docstring comment for the verified "onion ring" -> "French Fries, Potatoes &
    # Onion Rings" case) -- applied here too so the linking holdout doesn't score
    # against a ground truth that disagrees with the block's own known exclusions.
    # This function has no per-round `rule` to draw excludes from (unlike
    # pair_completeness), so it falls back to the seed rule -- the persisted,
    # SME-authored source of this domain knowledge (see blocking/seed_rules.py) -- if
    # one exists for this block; blocks with no seed rule get no extra exclusion here,
    # same as before this fix.
    seed_rule = get_seed_rule(block_name) or {}
    excludes = combined_exclude_keywords(seed_rule)
    branded_exclude_pred = exclude_predicate_sql("lower(coalesce(branded_food_category, ''))", excludes)
    off_exclude_pred = exclude_predicate_sql("lower(array_to_string(off_categories_tags, ' '))", excludes)
    holdout_path = str(CALIBRATION_DIR / "holdout.parquet").replace("'", "''")
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT fdc_id, off_code, fndds_style_description, branded_food_category,
               off_product_name, off_categories_tags
        FROM read_parquet('{holdout_path}')
        WHERE ({branded_term_pred} OR {off_term_pred})
          AND NOT ({branded_exclude_pred} OR {off_exclude_pred})
        """
    ).df()
    con.close()
    return df


def _off_block_pool(block_name: str, limit: int = 20000) -> pd.DataFrame:
    """A pool of OFF records from this block to draw negative-pair decoys from."""
    from agentic_matching.config import BLOCKS_DIR

    return pd.read_parquet(BLOCKS_DIR / f"{block_name}_off.parquet").head(limit)


def build_eval_frames(
    block_name: str,
    attrs: list[dict[str, Any]],
    n_decoys_per_positive: int = 5,
    seed: int = 7,
    max_holdout_positives: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Returns (fndds_side_df, off_side_df, true_labels) where true_labels maps
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
    off_pool = _off_block_pool(block_name)

    fndds_vals = compute_attribute_values(attrs, holdout["fndds_style_description"].tolist(), side="fndds")
    fndds_df = pd.DataFrame(
        {
            "unique_id": holdout["fdc_id"].astype(str),
            "description": holdout["fndds_style_description"],
            "search_text": holdout["fndds_style_description"].str.lower(),
            **{name: [stringify(v) for v in vals] for name, vals in fndds_vals.items()},
        }
    )

    true_off_codes = set(holdout["off_code"].astype(str))
    decoy_pool = off_pool[~off_pool["code"].astype(str).isin(true_off_codes)]
    n_decoys = min(len(decoy_pool), n_decoys_per_positive * max(len(holdout), 1))
    decoys = decoy_pool.sample(n=n_decoys, random_state=seed) if n_decoys > 0 else decoy_pool.iloc[:0]

    positives_off = pd.DataFrame(
        {
            "code": holdout["off_code"].astype(str),
            "product_name": holdout["off_product_name"],
            "search_text": holdout["off_product_name"].fillna("").str.lower(),
        }
    )
    off_side = pd.concat(
        [positives_off, decoys[["code", "product_name", "search_text"]]], ignore_index=True
    ).drop_duplicates(subset="code")

    off_vals = compute_attribute_values(attrs, off_side["search_text"].tolist(), side="off")
    off_df = pd.DataFrame(
        {
            "unique_id": off_side["code"].astype(str),
            "description": off_side["product_name"],
            "search_text": off_side["search_text"],
            **{name: [stringify(v) for v in vals] for name, vals in off_vals.items()},
        }
    )

    true_labels = dict(zip(holdout["fdc_id"].astype(str), holdout["off_code"].astype(str)))
    return fndds_df, off_df, true_labels


def score_against_holdout(
    block_name: str, attrs: list[dict[str, Any]], trained_settings: dict[str, Any], threshold: float = 0.5
) -> dict[str, Any]:
    fndds_df, off_df, true_labels = build_eval_frames(block_name, attrs)
    if fndds_df.empty or off_df.empty:
        return {"n_holdout_positives": 0, "precision": None, "recall": None, "f1": None}

    eval_settings = dict(trained_settings)
    eval_settings["blocking_rules_to_generate_predictions"] = ["1=1"]  # eval sets are small; exhaustive is fine

    linker = Linker([fndds_df, off_df], eval_settings, db_api=DuckDBAPI(), input_table_aliases=["fndds", "off"])
    preds = linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()

    # Best (highest-probability) OFF match per FNDDS-side unique_id.
    best = preds.sort_values("match_probability", ascending=False).drop_duplicates(subset="unique_id_l")
    best = best[best["match_probability"] >= threshold]

    predicted = dict(zip(best["unique_id_l"], best["unique_id_r"]))
    n_true = len(true_labels)
    n_pred = len(predicted)
    n_correct = sum(1 for k, v in predicted.items() if true_labels.get(k) == v)

    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_true if n_true else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "n_holdout_positives": n_true,
        "n_predicted": n_pred,
        "n_correct": n_correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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
            "unique_id_r": "off_code",
            "description_r": "off_product_name",
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

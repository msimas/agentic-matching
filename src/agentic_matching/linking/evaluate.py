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


def _build_holdout_predictions(
    block_name: str, attrs: list[dict[str, Any]], trained_settings: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame | None]:
    """Shared setup behind score_against_holdout and holdout_error_examples: build the
    labeled holdout eval frames and run one exhaustive prediction pass against them
    (the trained model's comparison weights, scored on the small, capped holdout
    sample -- see build_eval_frames' docstring for why "1=1" blocking is safe only
    here). Split out so both consumers pay for this once each, not twice."""
    fndds_df, off_df, true_labels = build_eval_frames(block_name, attrs)
    if fndds_df.empty or off_df.empty:
        return fndds_df, off_df, true_labels, None

    eval_settings = dict(trained_settings)
    eval_settings["blocking_rules_to_generate_predictions"] = ["1=1"]  # eval sets are small; exhaustive is fine

    linker = Linker([fndds_df, off_df], eval_settings, db_api=DuckDBAPI(), input_table_aliases=["fndds", "off"])
    preds = linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()
    return fndds_df, off_df, true_labels, preds


def score_against_holdout(
    block_name: str, attrs: list[dict[str, Any]], trained_settings: dict[str, Any], threshold: float = 0.5
) -> dict[str, Any]:
    fndds_df, off_df, true_labels, preds = _build_holdout_predictions(block_name, attrs, trained_settings)
    if fndds_df.empty or off_df.empty:
        return {"n_holdout_positives": 0, "precision": None, "recall": None, "f1": None}

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


def _holdout_error_examples(
    preds: pd.DataFrame, true_labels: dict[str, str], threshold: float = 0.5, n: int = 5
) -> dict[str, list[dict[str, Any]]]:
    """Pure logic behind holdout_error_examples, split out so it's testable against a
    small synthetic `preds` frame without needing a real trained linker (same rationale
    as _attribute_discriminative_power's split from attribute_discriminative_power)."""
    cols = [c for c in ("unique_id_l", "unique_id_r", "description_l", "description_r", "match_probability") if c in preds.columns]
    if preds.empty or not true_labels:
        return {"false_positives": [], "false_negatives": []}

    # False positives: the model's best-scoring OFF match per FNDDS record, confident
    # (>= threshold) but NOT the true calibration pair -- the n highest-probability
    # such mistakes, i.e. what's most urgently over-matching right now.
    best = preds.sort_values("match_probability", ascending=False).drop_duplicates(subset="unique_id_l")
    confident = best[best["match_probability"] >= threshold]
    if confident.empty:
        # DataFrame.apply(axis=1) on an empty frame returns an empty DataFrame, not a
        # boolean Series, which breaks confident[is_wrong] below -- skip straight to
        # "no false positives" instead.
        false_positives = confident
    else:
        is_wrong = confident.apply(lambda r: true_labels.get(r["unique_id_l"]) != r["unique_id_r"], axis=1)
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

    return {
        "false_positives": false_positives[cols].to_dict(orient="records"),
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

    "false_positives": pairs confidently (>= `threshold`) predicted as a match that
    are NOT the true calibration pair. "false_negatives": true calibration pairs the
    model scored below `threshold` for that specific pair. Both capped at `n`,
    ranked by how confidently wrong (false positives) or how far short of the
    threshold (false negatives) they are -- the most actionable examples first.
    """
    _, _, true_labels, preds = _build_holdout_predictions(block_name, attrs, trained_settings)
    if preds is None:
        return {"false_positives": [], "false_negatives": []}
    return _holdout_error_examples(preds, true_labels, threshold=threshold, n=n)


def _attribute_discriminative_power(
    fndds_df: pd.DataFrame,
    off_df: pd.DataFrame,
    true_labels: dict[str, str],
    attrs: list[dict[str, Any]],
    n_decoys_per_positive: int = 5,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Pure value-comparison logic behind `attribute_discriminative_power`, split out
    so it's testable against small synthetic frames without needing the real
    calibration parquet (same rationale as profiling._rank_terms's split from
    high_frequency_terms)."""
    if fndds_df.empty or off_df.empty or not true_labels:
        return []

    fndds_by_id = fndds_df.set_index("unique_id")
    off_by_id = off_df.set_index("unique_id")
    off_ids = list(off_by_id.index)
    rng = random.Random(seed)

    results = []
    for attr in attrs:
        name = attr["name"]
        if name not in fndds_by_id.columns or name not in off_by_id.columns:
            continue
        true_agree = []
        decoy_agree = []
        for fndds_id, true_off_id in true_labels.items():
            if fndds_id not in fndds_by_id.index or true_off_id not in off_by_id.index:
                continue
            l_val = fndds_by_id.loc[fndds_id, name]
            true_agree.append(l_val == off_by_id.loc[true_off_id, name])
            decoy_ids = [oid for oid in rng.sample(off_ids, min(n_decoys_per_positive, len(off_ids))) if oid != true_off_id]
            decoy_agree.extend(l_val == off_by_id.loc[oid, name] for oid in decoy_ids)
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
    fndds_df, off_df, true_labels = build_eval_frames(block_name, attrs)
    return _attribute_discriminative_power(
        fndds_df, off_df, true_labels, attrs, n_decoys_per_positive=n_decoys_per_positive, seed=seed
    )


def best_match_per_off(predictions: pd.DataFrame, min_probability: float = 0.0) -> pd.DataFrame:
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

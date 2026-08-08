"""Blocking-rule evaluation: pair completeness, reduction ratio, and block-size
diagnostics.

FNDDS has no UPC, so there is no direct FNDDS<->OFF gold-match population to measure
pair completeness against. Per the project's calibration design (see calibration.py),
we use the Branded<->OFF UPC-matched gold pairs as a proxy: Branded Foods' free-text
fields (description, branded_food_category) stand in for what FNDDS's text fields would
look like, since both are USDA product-level text descriptions of the same general kind.

"True" block membership for a gold pair (needed to compute pair completeness) is
determined by a canonical keyword independent of whatever rule the LLM is currently
proposing (to avoid circularity): a pair belongs to the block if the block's canonical
term appears in the Branded side's category text OR the OFF side's category/product
text.
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb

from agentic_matching.blocking.rules import fndds_predicate_sql, off_predicate_sql
from agentic_matching.config import FDC_DUCKDB_PATH, OFF_SEARCH_TEXT_PARQUET

log = logging.getLogger(__name__)

# Canonical ground-truth term per block, used ONLY to determine calibration-proxy block
# membership -- never shown to the LLM as part of the candidate rule it's evaluated on.
CANONICAL_BLOCK_TERMS = {
    "yogurt": "yogurt",
    "beans": "bean",
}


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(FDC_DUCKDB_PATH))
    off_path = str(OFF_SEARCH_TEXT_PARQUET).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW off_search AS SELECT * FROM read_parquet('{off_path}')")
    con.execute(
        """
        CREATE OR REPLACE VIEW fndds_search AS
        SELECT
            fdc_id,
            description,
            wweia_food_category_description,
            additional_description,
            lower(
                coalesce(description, '') || ' ' ||
                coalesce(wweia_food_category_description, '') || ' ' ||
                coalesce(additional_description, '')
            ) AS fndds_search_text
        FROM v_fndds
        """
    )
    return con


def block_sizes(con: duckdb.DuckDBPyConnection, rule: dict[str, Any]) -> dict[str, int]:
    n_fndds_total = con.execute("SELECT count(*) FROM fndds_search").fetchone()[0]
    n_off_total = con.execute("SELECT count(*) FROM off_search").fetchone()[0]
    fndds_pred = fndds_predicate_sql(rule)
    off_pred = off_predicate_sql(rule)
    n_fndds_block = con.execute(f"SELECT count(*) FROM fndds_search WHERE {fndds_pred}").fetchone()[0]
    n_off_block = con.execute(f"SELECT count(*) FROM off_search WHERE {off_pred}").fetchone()[0]
    return {
        "n_fndds_total": n_fndds_total,
        "n_off_total": n_off_total,
        "n_fndds_block": n_fndds_block,
        "n_off_block": n_off_block,
    }


def reduction_ratio(sizes: dict[str, int]) -> float:
    total_pairs = sizes["n_fndds_total"] * sizes["n_off_total"]
    block_pairs = sizes["n_fndds_block"] * sizes["n_off_block"]
    if total_pairs == 0:
        return 0.0
    return 1.0 - (block_pairs / total_pairs)


def pair_completeness(
    con: duckdb.DuckDBPyConnection, block_name: str, rule: dict[str, Any]
) -> dict[str, Any]:
    term = CANONICAL_BLOCK_TERMS[block_name].replace("'", "''")
    fndds_pred = fndds_predicate_sql(rule, text_col="lower(fndds_style_description)")
    off_pred = off_predicate_sql(rule, text_col="lower(coalesce(off_product_name, ''))")
    row = con.execute(
        f"""
        WITH proxy AS (
            SELECT
                fdc_id, off_code,
                fndds_style_description, off_product_name, branded_food_category,
                off_categories_tags
            FROM gold_pairs
        ),
        true_block AS (
            SELECT * FROM proxy
            WHERE lower(coalesce(branded_food_category, '')) LIKE '%{term}%'
               OR lower(array_to_string(off_categories_tags, ' ')) LIKE '%{term}%'
        )
        SELECT
            count(*) AS n_true_block,
            sum(CASE WHEN {fndds_pred} AND {off_pred} THEN 1 ELSE 0 END) AS n_recovered
        FROM true_block
        """
    ).fetchone()
    n_true_block, n_recovered = row
    n_recovered = n_recovered or 0
    pc = (n_recovered / n_true_block) if n_true_block else 0.0
    return {"n_true_block": n_true_block, "n_recovered": n_recovered, "pair_completeness": pc}


def evaluate_rule(block_name: str, rule: dict[str, Any]) -> dict[str, Any]:
    con = _connect()
    try:
        sizes = block_sizes(con, rule)
        rr = reduction_ratio(sizes)
        pc = pair_completeness(con, block_name, rule)
        metrics = {**sizes, "reduction_ratio": rr, **pc}
        log.info(
            "block=%s fndds=%d/%d off=%d/%d reduction_ratio=%.4f pair_completeness=%.3f (%d/%d)",
            block_name,
            sizes["n_fndds_block"],
            sizes["n_fndds_total"],
            sizes["n_off_block"],
            sizes["n_off_total"],
            rr,
            pc["pair_completeness"],
            pc["n_recovered"],
            pc["n_true_block"],
        )
        return metrics
    finally:
        con.close()

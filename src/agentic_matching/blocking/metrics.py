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

# Canonical ground-truth term(s) per block, used ONLY to determine calibration-proxy
# block membership (and to seed round-0 samples/category options) -- never shown to the
# LLM as part of the candidate rule it's evaluated on. A list, not a single string,
# because not every block anchors to one word: "breaded_vegetables" has no single term
# that works (FNDDS's own "breaded" entries are almost all chicken/turkey/fish, not
# vegetables -- the concept lives in FNDDS's "Fried vegetables" WWEIA category instead,
# and "fried" alone is far too broad a keyword), so its anchor is several specific
# dish-name terms OR'd together instead.
CANONICAL_BLOCK_TERMS: dict[str, list[str]] = {
    "yogurt": ["yogurt"],
    "beans": ["bean"],
    "breaded_vegetables": [
        "fried vegetable", "breaded vegetable", "onion ring", "fried okra",
        "fried mushroom", "fried cauliflower", "fried eggplant", "fried squash",
        "fried green tomato", "fried green bean", "fried broccoli",
        "vegetable tempura", "pakora", "fried pickle",
        # "sweet potato fries"/"sweet potato tot"/"yuca fries" deliberately dropped:
        # the block's definition was refined to "breaded" vegetables specifically, not
        # just "fried" ones (see blocking/seed_rules.py's exclude_keywords, which now
        # exclude "fries"/"fry"/"french fries" on both sides for the same reason) --
        # keeping these plain-fried, unbreaded items in the calibration ground truth
        # would make pair_completeness/holdout f1 structurally unrecoverable (they'd
        # always disagree with the rule they're supposed to be calibrating against),
        # not a genuine measure of match quality. Verified: with them still listed here,
        # pair_completeness and linking holdout f1 both read exactly 0.0 despite the
        # actual matches (matches_breaded_vegetables_round0.csv) looking correct
        # (Pakora<->Pakora, Vegetable Tempura<->Vegetable Tempura, ~0.87 probability).
    ],
}


def combined_exclude_keywords(rule: dict[str, Any]) -> list[str]:
    """The union of both sides' exclude_keywords from a blocking rule -- used to keep
    the calibration ground truth (pair_completeness/holdout, below) honest about the
    same false-positive patterns the rule itself was told to exclude."""
    excludes = {k.lower() for k in rule.get("fndds", {}).get("exclude_keywords", []) if k}
    excludes |= {k.lower() for k in rule.get("off", {}).get("exclude_keywords", []) if k}
    return sorted(excludes)


def exclude_predicate_sql(text_col: str, excludes: list[str]) -> str:
    """OR'd LIKE predicate over `excludes` against `text_col` -- "FALSE" (matches
    nothing) if there are none, so callers can unconditionally wrap with `NOT (...)`."""
    if not excludes:
        return "FALSE"
    return "(" + " OR ".join(f"{text_col} LIKE '%{e.replace(chr(39), chr(39)*2)}%'" for e in excludes) + ")"


def term_predicate_sql(text_col: str, block_name: str) -> str:
    """OR'd LIKE predicate over CANONICAL_BLOCK_TERMS[block_name] against `text_col`.
    Shared by pair_completeness (below) and blocking/agent_loop.py's round-0 sampling
    and category-options lookup -- anywhere the canonical (withheld-from-the-LLM)
    anchor term(s) are used to identify plausibly-in-block records."""
    terms = [t.lower().replace("'", "''") for t in CANONICAL_BLOCK_TERMS[block_name]]
    return "(" + " OR ".join(f"{text_col} LIKE '%{t}%'" for t in terms) + ")"


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
    branded_term_pred = term_predicate_sql("lower(coalesce(branded_food_category, ''))", block_name)
    off_term_pred = term_predicate_sql("lower(array_to_string(off_categories_tags, ' '))", block_name)
    # A canonical term matched against a *category* string, not a product name, can be
    # much coarser than intended: verified case (this project's "breaded_vegetables"
    # block) -- Branded Foods lumps onion rings into one mega-category, "French Fries,
    # Potatoes & Onion Rings", so the canonical term "onion ring" alone pulled ~3,947
    # gold pairs into true_block via *that whole category*, the overwhelming majority of
    # which are plain fries/potato products, not onion rings. The rule's own
    # exclude_keywords (e.g. "fries", "fry") already encode exactly this kind of
    # false-positive pattern for the *block* predicate below -- applying the same
    # excludes to the true_block definition itself keeps the calibration ground truth
    # consistent with the domain knowledge the rule was given, instead of silently
    # scoring the rule against a ground truth it was explicitly told to disagree with.
    excludes = combined_exclude_keywords(rule)
    branded_exclude_pred = exclude_predicate_sql("lower(coalesce(branded_food_category, ''))", excludes)
    off_exclude_pred = exclude_predicate_sql("lower(array_to_string(off_categories_tags, ' '))", excludes)
    # Branded Foods (the FNDDS-side text stand-in here -- see module docstring) has no
    # WWEIA-equivalent categorization, so the fndds-side category predicate can't be
    # evaluated against this proxy; category_col=None falls back to keywords only for
    # this side. The OFF side's real categories_tags IS available in gold_pairs.
    fndds_pred = fndds_predicate_sql(rule, text_col="lower(fndds_style_description)", category_col=None)
    off_pred = off_predicate_sql(rule, text_col="lower(coalesce(off_product_name, ''))", category_col="off_categories_tags")
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
            WHERE ({branded_term_pred} OR {off_term_pred})
              AND NOT ({branded_exclude_pred} OR {off_exclude_pred})
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

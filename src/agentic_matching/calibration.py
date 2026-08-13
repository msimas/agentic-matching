"""Build the calibration (gold-match) dataset: Branded Foods <-> Open Food Facts pairs
sharing the same UPC/GTIN.

FNDDS has no UPC field, so it cannot supply gold matches directly. Branded Foods
(`gtin_upc`) and Open Food Facts (`code`) are both UPC/GTIN-bearing product catalogs, so
matching on normalized code gives a high-confidence "known match" population. Per the
project's confirmed design, this calibration set is the objective used to validate
blocking/matching methodology (pair completeness, reduction ratio, precision/recall);
that methodology is then applied to the actual target linkage, FNDDS<->OFF, which has no
direct ground truth of its own.

UPC/GTIN normalization: barcodes appear at multiple standard lengths (UPC-A 12,
EAN-13/GTIN-13 13, GTIN-14 14, UPC-E/EAN-8 8, ...) and shorter forms are usually just
zero-padded/truncated variants of the same number (EAN-13 = "0" + UPC-A, GTIN-14 pads
further). Stripping non-digits and leading zeros collapses these to a comparable numeric
core; this is a reasonable POC-grade normalization, not a full GS1 checksum/UPC-E
expansion implementation.
"""

from __future__ import annotations

import logging
import re

import duckdb

from agentic_matching.config import CALIBRATION_DIR, FDC_DUCKDB_PATH, OFF_PARQUET, configure_logging

log = logging.getLogger(__name__)

_NON_DIGIT_RE = re.compile(r"[^0-9]")

# Same logic as normalize_code(), expressed in SQL for use in the big duckdb join
# (row-wise Python normalization would be far too slow over ~2M x ~4.6M candidate rows).
_NORMALIZE_SQL = """
    nullif(ltrim(regexp_replace({col}, '[^0-9]', '', 'g'), '0'), '')
"""

# Word-level Jaccard similarity between the two sides' text -- a cheap, dependency-free
# sanity check that a UPC/GTIN match is ALSO a plausible text match, not just a shared
# barcode (which could reflect barcode reuse, a mislabeled listing, or a plain data-entry
# error on either side -- nothing about the UPC join alone rules that out). DuckDB-native
# (regexp_extract_all, same tokenization pattern as profiling.py) rather than adding a
# fuzzy-matching dependency for what's fundamentally bag-of-words overlap at this scale
# (~1.8M pairs) -- fast enough to run as one vectorized query, not per-row Python.
#
# DELIBERATELY NOT used to auto-filter gold_pairs/gold_train/gold_holdout -- verified
# against a random sample of the real zero-similarity tail (40,741 of 1,777,551 pairs,
# 2.3%, at build time on this project's actual data) that a blind low-similarity cutoff
# would be WRONG far too often to be safe:
#   - cross-language listings for the same real product ('DARE, BRETON, WHITE BEAN
#     CRACKERS WITH SALT & PEPPER' <-> 'Craquelin haricots blancs avec sel et poivre',
#     zero English/French token overlap, same product)
#   - a real brand name that shares no words with the generic description ('ZESTY RANCH
#     CRUNCHY BROAD BEANS' <-> 'bada bean bada boom')
#   - Branded's own description sometimes just names the PACKAGING, not the product
#     ('Aluminum Cans' <-> 'Seltzer Cranberry Lime') -- a real match, Branded's data is
#     just uninformative here, not wrong
#   - tokenization edge cases (concatenated brand names: 'BAHAMA MAMA ENERGY DRINK' <->
#     'GFUEL BahamaMama' reads as zero overlap only because "BahamaMama" is one token)
# Genuinely wrong-looking pairs exist in the same tail too (e.g. a long real branded
# description paired with OFF product_name "yea"), but a score this noisy can't reliably
# tell the two apart on its own -- surfaced as a visible column + a dedicated low-
# similarity review export (see export_low_similarity_examples) for a human to judge,
# the same SME-review pattern this project already uses elsewhere, rather than a silent
# automated cut that would measurably discard real matches.
def _description_similarity_sql(text_a: str, text_b: str) -> str:
    """Word-level Jaccard similarity between two text expressions -- a single vectorized
    expression (no correlated subquery/row-by-row evaluation), so it stays fast at this
    table's ~1.8M-row scale."""
    t1 = f"list_distinct(regexp_extract_all(lower(coalesce({text_a}, '')), '[a-z]+'))"
    t2 = f"list_distinct(regexp_extract_all(lower(coalesce({text_b}, '')), '[a-z]+'))"
    return (
        f"CASE WHEN len(list_distinct(list_concat({t1}, {t2}))) = 0 THEN 0.0 "
        f"ELSE len(list_intersect({t1}, {t2}))::DOUBLE / len(list_distinct(list_concat({t1}, {t2}))) END"
    )


def normalize_code(raw: str | None) -> str | None:
    """Normalize a UPC/EAN/GTIN string to a comparable digit-only, zero-stripped core.
    Returns None for missing/empty/all-zero input."""
    if raw is None:
        return None
    digits = _NON_DIGIT_RE.sub("", raw)
    stripped = digits.lstrip("0")
    return stripped or None


def _attach_catalog(con: duckdb.DuckDBPyConnection) -> None:
    catalog_path = str(OFF_PARQUET).replace("'", "''")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW catalog_raw AS
        SELECT
            code,
            product_name,
            categories_tags,
            brands,
            ingredients_text,
            quantity
        FROM read_parquet('{catalog_path}')
        """
    )


def build_gold_pairs(con: duckdb.DuckDBPyConnection) -> None:
    """Populate `gold_pairs`: one row per Branded<->OFF match on normalized UPC/GTIN."""
    norm_gtin = _NORMALIZE_SQL.format(col="bf.gtin_upc")
    norm_code = _NORMALIZE_SQL.format(col="off.code")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE gold_pairs AS
        WITH branded_norm AS (
            SELECT bf.*, {norm_gtin} AS norm_upc
            FROM v_branded bf
        ),
        catalog_norm AS (
            SELECT
                off.code,
                -- product_name is STRUCT(lang, text)[]; take the first entry's text.
                (list_transform(off.product_name, x -> x.text))[1] AS catalog_product_name,
                off.categories_tags,
                off.brands,
                (list_transform(off.ingredients_text, x -> x.text))[1] AS catalog_ingredients_text,
                off.quantity,
                {norm_code} AS norm_upc
            FROM catalog_raw off
        )
        SELECT
            b.fdc_id,
            b.description AS fndds_style_description,  -- Branded's food.csv description
            b.brand_owner,
            b.brand_name,
            b.branded_food_category,
            b.ingredients AS branded_ingredients,
            b.gtin_upc,
            o.code AS catalog_code,
            o.catalog_product_name,
            o.categories_tags AS catalog_categories_tags,
            o.brands AS catalog_brands,
            o.catalog_ingredients_text,
            o.quantity AS catalog_quantity,
            b.norm_upc,
            {_description_similarity_sql("b.description", "o.catalog_product_name")} AS description_similarity
        FROM branded_norm b
        JOIN catalog_norm o ON o.norm_upc = b.norm_upc
        WHERE b.norm_upc IS NOT NULL
        """
    )
    n = con.execute("SELECT count(*) FROM gold_pairs").fetchone()[0]
    n_distinct_upc = con.execute("SELECT count(DISTINCT norm_upc) FROM gold_pairs").fetchone()[0]
    log.info(
        "gold_pairs: %d Branded<->OFF row-level matches on normalized UPC/GTIN "
        "(%d distinct GTINs; Branded Foods re-publishes many rows per GTIN)",
        n,
        n_distinct_upc,
    )
    sim = con.execute(
        """
        SELECT
            round(avg(description_similarity), 4),
            round(median(description_similarity), 4),
            round(quantile_cont(description_similarity, 0.1), 4),
            sum(CASE WHEN description_similarity = 0 THEN 1 ELSE 0 END),
            sum(CASE WHEN catalog_product_name IS NULL THEN 1 ELSE 0 END)
        FROM gold_pairs
        """
    ).fetchone()
    log.info(
        "gold_pairs description_similarity (word-Jaccard, UPC-matched pairs' text -- "
        "NOT auto-filtered on, see _DESCRIPTION_SIMILARITY_SQL's docstring for why): "
        "mean=%.4f median=%.4f p10=%.4f, %d/%d pairs (%.1f%%) at zero overlap, "
        "%d with no OFF product_name text at all",
        *sim[:4],
        n,
        (sim[3] / n * 100) if n else 0.0,
        sim[4],
    )


def stratified_sample(
    con: duckdb.DuckDBPyConnection, n_per_stratum: int = 25, seed: float = 0.42
) -> None:
    """Stratify by branded_food_category (coarse product category) so the calibration
    sample spans the range of categories rather than only the easiest/most common ones."""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE calibration_sample AS
        SELECT * FROM (
            SELECT *,
                row_number() OVER (
                    PARTITION BY branded_food_category
                    ORDER BY hash(fdc_id || catalog_code || {seed}::VARCHAR)
                ) AS rn
            FROM gold_pairs
        )
        WHERE rn <= {n_per_stratum}
        """
    )
    n = con.execute("SELECT count(*) FROM calibration_sample").fetchone()[0]
    n_strata = con.execute(
        "SELECT count(DISTINCT branded_food_category) FROM calibration_sample"
    ).fetchone()[0]
    log.info(
        "calibration_sample: %d pairs across %d branded_food_category strata "
        "(up to %d per stratum)",
        n,
        n_strata,
        n_per_stratum,
    )


def train_holdout_split(con: duckdb.DuckDBPyConnection, holdout_frac: float = 0.3) -> None:
    """Split the full gold-pair population (not just the SME sample) into train/holdout
    so downstream blocking/matching evaluation has a large enough holdout to be
    statistically meaningful.

    Branded Foods re-publishes the same physical product many times under the same
    GTIN (this dataset has ~2.0M rows but only ~465K distinct GTINs, up to 38x
    duplication for a single barcode) — splitting by individual (fdc_id, catalog_code) pair
    would let near-duplicate rows for the same GTIN leak across train and holdout. We
    split by norm_upc (the GTIN group) instead, so every row for a given barcode lands
    on the same side.
    """
    con.execute(
        f"""
        CREATE OR REPLACE TABLE gold_holdout AS
        SELECT * FROM gold_pairs
        WHERE (hash(norm_upc) % 1000) / 1000.0 < {holdout_frac}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE gold_train AS
        SELECT g.* FROM gold_pairs g
        LEFT JOIN gold_holdout h ON h.fdc_id = g.fdc_id AND h.catalog_code = g.catalog_code
        WHERE h.fdc_id IS NULL
        """
    )
    n_train = con.execute("SELECT count(*) FROM gold_train").fetchone()[0]
    n_holdout = con.execute("SELECT count(*) FROM gold_holdout").fetchone()[0]
    log.info("gold_train: %d, gold_holdout: %d", n_train, n_holdout)


def export_low_similarity_examples(con: duckdb.DuckDBPyConnection, n: int = 200) -> None:
    """Writes the `n` lowest-`description_similarity` real gold pairs to
    `low_similarity_examples.csv`, for a human to spot-check -- NOT an automated filter
    (see _DESCRIPTION_SIMILARITY_SQL's docstring for why a blind cutoff would discard
    real matches too often to be safe). This is the same SME-review pattern
    `sme_spot_check_sample.csv` already uses elsewhere, just prioritized toward the
    pairs a similarity score flags as worth a second look, rather than a random spread.
    Excludes pairs with no OFF product_name text at all (missing text, not a text
    MISMATCH -- a different, already-visible problem via description_similarity=0.0's
    own count in build_gold_pairs' log line, not something a human review of the text
    can usefully judge)."""
    path = str(CALIBRATION_DIR / "low_similarity_examples.csv").replace("'", "''")
    con.execute(
        f"""
        COPY (
            SELECT fdc_id, catalog_code, branded_food_category, brand_name,
                   fndds_style_description AS branded_description, catalog_product_name,
                   description_similarity, gtin_upc
            FROM gold_pairs
            WHERE catalog_product_name IS NOT NULL
            ORDER BY description_similarity ASC, fdc_id
            LIMIT {n}
        ) TO '{path}' (FORMAT CSV, HEADER)
        """
    )
    log.info("Wrote %d lowest-description_similarity pairs to %s for SME review", n, path)


def export_artifacts(con: duckdb.DuckDBPyConnection) -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    for table, fname in [
        ("gold_pairs", "gold_pairs.parquet"),
        ("gold_train", "train.parquet"),
        ("gold_holdout", "holdout.parquet"),
    ]:
        path = str(CALIBRATION_DIR / fname).replace("'", "''")
        con.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET)")
    sme_csv = str(CALIBRATION_DIR / "sme_spot_check_sample.csv").replace("'", "''")
    con.execute(
        f"""
        COPY (
            SELECT fdc_id, catalog_code, branded_food_category, brand_name,
                   fndds_style_description AS branded_description, catalog_product_name,
                   description_similarity, gtin_upc, catalog_code AS catalog_upc_raw
            FROM calibration_sample
            ORDER BY branded_food_category, fdc_id
        ) TO '{sme_csv}' (FORMAT CSV, HEADER)
        """
    )
    export_low_similarity_examples(con)
    log.info("Exported calibration artifacts to %s", CALIBRATION_DIR)


def build(holdout_frac: float = 0.3, n_per_stratum: int = 25) -> None:
    con = duckdb.connect(str(FDC_DUCKDB_PATH))
    try:
        _attach_catalog(con)
        build_gold_pairs(con)
        stratified_sample(con, n_per_stratum=n_per_stratum)
        train_holdout_split(con, holdout_frac=holdout_frac)
        export_artifacts(con)
    finally:
        con.close()


if __name__ == "__main__":
    configure_logging()
    build()

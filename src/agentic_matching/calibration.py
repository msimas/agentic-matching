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
            b.norm_upc
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
                   gtin_upc, catalog_code AS catalog_upc_raw
            FROM calibration_sample
            ORDER BY branded_food_category, fdc_id
        ) TO '{sme_csv}' (FORMAT CSV, HEADER)
        """
    )
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

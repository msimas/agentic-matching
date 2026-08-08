"""Build data/fdc.duckdb: link everything linkable on the USDA side using fdc_id.

Each of the four dataset zips (Foundation, SR Legacy, FNDDS, Branded) extracts its own
copy of `food.csv` (only rows for that data_type) plus shared reference tables
(`nutrient.csv`, `food_attribute_type.csv`, `measure_unit.csv`, ...) that are identical
across zips. This step:
  1. Loads every per-dataset Parquet table as a raw DuckDB table (dataset slug prefixed).
  2. Unions + de-duplicates the reference tables that are repeated across zips.
  3. Builds one unified view per dataset type (v_foundation, v_sr_legacy, v_fndds,
     v_branded), each joined from food + its dataset-specific table + food category /
     WWEIA category + an "Additional Description" pivot from food_attribute, all keyed
     on fdc_id — this is the direct USDA-side linkage the vision asks for.
"""

from __future__ import annotations

import logging

import duckdb

from agentic_matching.config import FDC_DUCKDB_PATH, PARQUET_FDC_DIR

log = logging.getLogger(__name__)

DATASET_SLUGS = ["foundation", "sr_legacy", "fndds", "branded"]


def _load_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    for slug in DATASET_SLUGS:
        slug_dir = PARQUET_FDC_DIR / slug
        for pq in sorted(slug_dir.glob("*.parquet")):
            table_name = f"{slug}__{pq.stem}"
            path_sql = str(pq).replace("'", "''")
            con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{path_sql}')"
            )
            log.info("Loaded raw table %s", table_name)


def _build_reference_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Union + de-dup tables that are repeated verbatim (or near-verbatim) across the
    four dataset zips: nutrient, food_attribute_type, measure_unit."""
    con.execute(
        """
        CREATE OR REPLACE TABLE nutrient AS
        SELECT DISTINCT * FROM (
            SELECT * FROM foundation__nutrient
            UNION ALL SELECT * FROM sr_legacy__nutrient
            UNION ALL SELECT * FROM fndds__nutrient
            UNION ALL SELECT * FROM branded__nutrient
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE food_attribute_type AS
        SELECT DISTINCT * FROM (
            SELECT * FROM foundation__food_attribute_type
            UNION ALL SELECT * FROM sr_legacy__food_attribute_type
            UNION ALL SELECT * FROM fndds__food_attribute_type
            UNION ALL SELECT * FROM branded__food_attribute_type
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE measure_unit AS
        SELECT DISTINCT * FROM (
            SELECT * FROM foundation__measure_unit
            UNION ALL SELECT * FROM sr_legacy__measure_unit
            UNION ALL SELECT * FROM fndds__measure_unit
            UNION ALL SELECT * FROM branded__measure_unit
        )
        """
    )
    # food.csv: each zip only contains rows for its own data_type(s) (Foundation's also
    # includes related acquisition/market sample sub-records) -> union covers all fdc_ids.
    # branded__food carries a few extra columns (market_country, trade_channel,
    # microbe_data) not present in the other three zips -> select the common columns.
    common_food_cols = "fdc_id, data_type, description, food_category_id, publication_date"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE food AS
        SELECT DISTINCT {common_food_cols} FROM (
            SELECT {common_food_cols} FROM foundation__food
            UNION ALL SELECT {common_food_cols} FROM sr_legacy__food
            UNION ALL SELECT {common_food_cols} FROM fndds__food
            UNION ALL SELECT {common_food_cols} FROM branded__food
        )
        """
    )
    # food_category.csv only ships with Foundation/SR Legacy (Branded/FNDDS use their
    # own category schemes: branded_food_category text field, WWEIA category).
    con.execute(
        """
        CREATE OR REPLACE TABLE food_category AS
        SELECT DISTINCT * FROM (
            SELECT * FROM foundation__food_category
            UNION ALL SELECT * FROM sr_legacy__food_category
        )
        """
    )
    log.info("Built de-duplicated reference tables: nutrient, food_attribute_type, "
              "measure_unit, food, food_category")


def _additional_description_pivot_sql(slug: str) -> str:
    """FNDDS (and other datasets) store a free-text 'Additional Description' as a row
    in food_attribute rather than a column; pivot it to one row per fdc_id."""
    return f"""
        SELECT
            fa.fdc_id,
            string_agg(fa.value, '; ') AS additional_description
        FROM {slug}__food_attribute fa
        JOIN food_attribute_type fat ON TRY_CAST(fa.food_attribute_type_id AS BIGINT) = fat.id
        WHERE fat.name = 'Additional Description'
        GROUP BY fa.fdc_id
    """


def _build_unified_views(con: duckdb.DuckDBPyConnection) -> None:
    # --- FNDDS ---
    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_fndds AS
        SELECT
            f.fdc_id,
            f.description,
            f.publication_date,
            sf.food_code,
            sf.wweia_category_number,
            wc.wweia_food_category_description,
            sf.start_date,
            sf.end_date,
            ad.additional_description
        FROM food f
        JOIN fndds__survey_fndds_food sf ON sf.fdc_id = f.fdc_id
        LEFT JOIN fndds__wweia_food_category wc
            ON wc.wweia_food_category = sf.wweia_category_number
        LEFT JOIN ({_additional_description_pivot_sql("fndds")}) ad ON ad.fdc_id = f.fdc_id
        WHERE f.data_type = 'survey_fndds_food'
        """
    )

    # --- SR Legacy ---
    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_sr_legacy AS
        SELECT
            f.fdc_id,
            f.description,
            f.food_category_id,
            fc.description AS food_category_description,
            f.publication_date,
            sl.ndb_number,
            ad.additional_description
        FROM food f
        JOIN sr_legacy__sr_legacy_food sl ON sl.fdc_id = f.fdc_id
        LEFT JOIN food_category fc ON TRY_CAST(f.food_category_id AS BIGINT) = TRY_CAST(fc.id AS BIGINT)
        LEFT JOIN ({_additional_description_pivot_sql("sr_legacy")}) ad ON ad.fdc_id = f.fdc_id
        WHERE f.data_type = 'sr_legacy_food'
        """
    )

    # --- Foundation Foods ---
    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_foundation AS
        SELECT
            f.fdc_id,
            f.description,
            f.food_category_id,
            fc.description AS food_category_description,
            f.publication_date,
            ff.ndb_number,
            ff.footnote,
            ad.additional_description
        FROM food f
        JOIN foundation__foundation_food ff ON ff.fdc_id = f.fdc_id
        LEFT JOIN food_category fc ON TRY_CAST(f.food_category_id AS BIGINT) = TRY_CAST(fc.id AS BIGINT)
        LEFT JOIN ({_additional_description_pivot_sql("foundation")}) ad ON ad.fdc_id = f.fdc_id
        WHERE f.data_type = 'foundation_food'
        """
    )

    # --- Branded Foods ---
    con.execute(
        f"""
        CREATE OR REPLACE VIEW v_branded AS
        SELECT
            f.fdc_id,
            f.description,
            f.publication_date,
            bf.gtin_upc,
            bf.brand_owner,
            bf.brand_name,
            bf.subbrand_name,
            bf.branded_food_category,
            bf.ingredients,
            bf.serving_size,
            bf.serving_size_unit,
            bf.household_serving_fulltext,
            bf.short_description,
            bf.market_country,
            bf.data_source,
            bf.discontinued_date,
            ad.additional_description
        FROM food f
        JOIN branded__branded_food bf ON bf.fdc_id = f.fdc_id
        LEFT JOIN ({_additional_description_pivot_sql("branded")}) ad ON ad.fdc_id = f.fdc_id
        WHERE f.data_type = 'branded_food'
        """
    )
    log.info("Built unified views: v_fndds, v_sr_legacy, v_foundation, v_branded")


def build(force: bool = False) -> None:
    if FDC_DUCKDB_PATH.exists() and force:
        FDC_DUCKDB_PATH.unlink()
    con = duckdb.connect(str(FDC_DUCKDB_PATH))
    try:
        _load_raw_tables(con)
        _build_reference_tables(con)
        _build_unified_views(con)
    finally:
        con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(force=True)
    con = duckdb.connect(str(FDC_DUCKDB_PATH), read_only=True)
    for view in ("v_fndds", "v_sr_legacy", "v_foundation", "v_branded"):
        n = con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
        print(f"{view}: {n} rows")
    con.close()

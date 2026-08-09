"""Convert every extracted FDC CSV (per dataset) to Parquet via DuckDB.

Each of the four dataset zips extracts its own copy of shared reference tables
(food.csv, nutrient.csv, food_attribute_type.csv, measure_unit.csv, ...) alongside its
dataset-specific table (branded_food.csv, survey_fndds_food.csv, sr_legacy_food.csv,
foundation_food.csv). This step converts 1:1, preserving dataset origin as a
subdirectory; deduplicating/unioning the shared reference tables across datasets is
build_fdc_db.py's job, not this one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from agentic_matching.config import FDC_DATASETS, PARQUET_FDC_DIR, RAW_FDC_DIR, configure_logging

log = logging.getLogger(__name__)


def _find_dataset_dir(slug: str) -> Path:
    """Each raw/fdc/<slug>/ contains one nested dir (the unzipped release folder)."""
    base = RAW_FDC_DIR / slug
    nested = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    if len(nested) == 1:
        return nested[0]
    if base.exists() and any(base.glob("*.csv")):
        return base
    raise FileNotFoundError(
        f"Could not find extracted CSVs for dataset '{slug}' under {base}. "
        "Run download_fdc.py first."
    )


def convert_dataset(slug: str, con: duckdb.DuckDBPyConnection, force: bool = False) -> list[Path]:
    src_dir = _find_dataset_dir(slug)
    out_dir = PARQUET_FDC_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for csv_path in sorted(src_dir.glob("*.csv")):
        out_path = out_dir / f"{csv_path.stem}.parquet"
        if out_path.exists() and not force:
            log.info("Skip (exists): %s", out_path)
            written.append(out_path)
            continue
        log.info("Converting %s -> %s", csv_path, out_path)
        # all_varchar avoids duckdb mis-typing sparse/inconsistent columns (common in
        # these CSVs, e.g. numeric-looking id columns with occasional blanks); downstream
        # consumers cast what they need. Paths are local, code-controlled (not user
        # input), so safe to inline after escaping single quotes.
        csv_sql = str(csv_path).replace("'", "''")
        out_sql = str(out_path).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_csv('{csv_sql}', header=true, all_varchar=true, sample_size=-1)
            ) TO '{out_sql}' (FORMAT PARQUET)
            """
        )
        written.append(out_path)
    return written


def convert_all(force: bool = False) -> dict[str, list[Path]]:
    con = duckdb.connect()
    result: dict[str, list[Path]] = {}
    for slug in FDC_DATASETS.values():
        result[slug] = convert_dataset(slug, con, force=force)
    con.close()
    return result


if __name__ == "__main__":
    configure_logging()
    out = convert_all()
    for slug, paths in out.items():
        print(f"{slug}: {len(paths)} tables -> {PARQUET_FDC_DIR / slug}")

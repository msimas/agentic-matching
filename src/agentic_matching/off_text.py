"""Flatten Open Food Facts' struct/array text fields into plain strings once, so the
blocking, matching-attribute, and linking stages don't repeatedly pay the
struct-extraction cost over OFF's ~4.66M rows.
"""

from __future__ import annotations

import logging

import duckdb

from agentic_matching.config import OFF_PARQUET, OFF_SEARCH_TEXT_PARQUET, configure_logging

log = logging.getLogger(__name__)


def build(force: bool = False) -> None:
    if OFF_SEARCH_TEXT_PARQUET.exists() and not force:
        log.info("Already built: %s", OFF_SEARCH_TEXT_PARQUET)
        return
    OFF_SEARCH_TEXT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    off_path = str(OFF_PARQUET).replace("'", "''")
    out_path = str(OFF_SEARCH_TEXT_PARQUET).replace("'", "''")
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT
                code,
                (list_transform(product_name, x -> x.text))[1] AS product_name,
                (list_transform(ingredients_text, x -> x.text))[1] AS ingredients_text,
                categories_tags,
                array_to_string(categories_tags, ' ') AS categories_joined,
                brands,
                quantity,
                lower(coalesce((list_transform(product_name, x -> x.text))[1], '') || ' ' ||
                      coalesce(array_to_string(categories_tags, ' '), '') || ' ' ||
                      coalesce(brands, '')) AS search_text
            FROM read_parquet('{off_path}')
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    )
    con.close()
    log.info("Built %s", OFF_SEARCH_TEXT_PARQUET)


if __name__ == "__main__":
    configure_logging()
    build(force=True)

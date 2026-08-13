"""Flatten Open Food Facts' struct/array text fields into plain strings once, so the
blocking, matching-attribute, and linking stages don't repeatedly pay the
struct-extraction cost over OFF's ~4.66M rows.
"""

from __future__ import annotations

import logging

import duckdb

from agentic_matching.config import OFF_PARQUET, OFF_SEARCH_TEXT_PARQUET, configure_logging

log = logging.getLogger(__name__)


def flatten(raw_path: str, out_path: str) -> None:
    """The struct/array-flattening step itself, as a pure `(raw_path, out_path) ->
    None` function -- this is what `catalog_source.OFF_SOURCE.flatten_sql` points at.
    Kept free of any `config`/pathlib dependency so it composes cleanly with a
    `CatalogSource` built elsewhere; `build()` below is the config-driven convenience
    wrapper scripts actually call."""
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
            FROM read_parquet('{raw_path}')
        ) TO '{out_path}' (FORMAT PARQUET)
        """
    )
    con.close()


def build(force: bool = False) -> None:
    if OFF_SEARCH_TEXT_PARQUET.exists() and not force:
        log.info("Already built: %s", OFF_SEARCH_TEXT_PARQUET)
        return
    OFF_SEARCH_TEXT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    off_path = str(OFF_PARQUET).replace("'", "''")
    out_path = str(OFF_SEARCH_TEXT_PARQUET).replace("'", "''")
    flatten(off_path, out_path)
    log.info("Built %s", OFF_SEARCH_TEXT_PARQUET)


if __name__ == "__main__":
    configure_logging()
    build(force=True)

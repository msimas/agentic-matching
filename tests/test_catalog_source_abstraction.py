"""Validates that catalog_source.CatalogSource actually generalizes -- exercised
against a SYNTHETIC second catalog with a deliberately different shape from OFF
(different column names, "exact" category matching instead of "array_contains"), not
just OFF-with-new-names. If blocking/rules.py's catalog_predicate_sql (or any other
consumer) secretly assumed OFF's specific column names/category semantics, this is
where that would show up.
"""

import duckdb

from agentic_matching.blocking.rules import catalog_predicate_sql
from agentic_matching.catalog_source import ACTIVE_CATALOG_SOURCE, CatalogSource

# A stand-in for a hypothetical proprietary retail catalog (e.g. "Circana", per
# PLAN.md's original framing of OFF as its stand-in) -- single scalar category column
# ("exact" match, like FNDDS's own WWEIA category), different column names throughout,
# no struct/array flattening needed (flatten_sql is a no-op copy).
def _noop_flatten(raw_path: str, out_path: str) -> None:
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM read_parquet('{raw_path}')) TO '{out_path}' (FORMAT PARQUET)")
    con.close()


FAKE_SOURCE = CatalogSource(
    name="fakecat",
    display_name="FakeCat Retail Catalog",
    raw_parquet=None,  # unused by this test -- only the shape/kind fields matter here
    search_text_parquet=None,
    id_col="sku",
    product_name_col="item_name",
    category_col="dept",
    category_kind="exact",
    brand_col="manufacturer",
    search_text_col="item_text",
    generic_term_min_doc_count=None,
    flatten_sql=_noop_flatten,
)


def test_synthetic_source_is_a_real_catalogsource_instance():
    # Sanity check the fixture itself is well-formed before using it below.
    assert FAKE_SOURCE.category_kind == "exact"
    assert FAKE_SOURCE.name != ACTIVE_CATALOG_SOURCE.name


def test_catalog_predicate_sql_honors_a_different_category_kind():
    # OFF's "array_contains" semantics use list_contains(); a catalog declaring "exact"
    # (like FNDDS's own WWEIA category) should get plain equality instead -- this is
    # the one axis blocking/rules.py's category_kind parameter exists to make pluggable.
    rule = {"catalog": {"keywords": ["widget"], "categories": ["Hardware"]}}
    sql = catalog_predicate_sql(
        rule,
        text_col=FAKE_SOURCE.search_text_col,
        category_col=FAKE_SOURCE.category_col,
        category_kind=FAKE_SOURCE.category_kind,
    )
    assert "list_contains" not in sql
    assert f"lower({FAKE_SOURCE.category_col})" in sql
    assert FAKE_SOURCE.search_text_col in sql


def test_catalog_predicate_sql_honors_different_column_names():
    rule = {"catalog": {"keywords": ["gadget"]}}
    sql = catalog_predicate_sql(rule, text_col=FAKE_SOURCE.search_text_col, category_col=None)
    assert f"lower({FAKE_SOURCE.search_text_col})" in sql
    # Confirms this isn't just silently falling back to OFF's own column name.
    assert "search_text" not in sql


def test_off_source_still_uses_array_contains_by_default():
    # The real, currently-active instantiation is untouched by the synthetic one above
    # -- confirms the two coexist as independent CatalogSource values, not global state
    # one mutates for the other.
    rule = {"catalog": {"keywords": ["yogurt"], "categories": ["en:yogurts"]}}
    sql = catalog_predicate_sql(rule)
    assert "list_contains" in sql

"""The pluggable "second side" of the FNDDS<->X linkage: a `CatalogSource` describes
whichever food-product database FNDDS is being connected to (Open Food Facts today;
eventually a proprietary retail catalog, per README's `best_match_per_catalog` note that
"nothing about this assumes OFF specifically").

FNDDS itself is NOT given a matching adapter -- that's a deliberate asymmetry, not an
oversight: FNDDS is the fixed anchor side in every real use case this project targets
(~5.4K rows, no struct/array flattening needed, no scale-tuned constants), while the
"other side" is what's expected to vary. Forcing FNDDS into the same abstraction would
be symmetry for its own sake, adding indirection with no current benefit.

Every blocking/attributes/linking/profiling/calibration consumer that used to hardcode
an OFF-specific path, column name, or scale constant should instead read it from
`ACTIVE_CATALOG_SOURCE` (module-level, swappable in tests via a different
`CatalogSource` instance -- see tests/test_catalog_source_abstraction.py for a
synthetic second instantiation that validates this actually generalizes, not just that
OFF-with-new-names still works).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


@dataclass(frozen=True)
class CatalogSource:
    # Short slug for logging/config, e.g. "off". Not shown to the LLM.
    name: str
    # Human-readable name shown in LLM-facing prompt text, e.g. "Open Food Facts (OFF)".
    display_name: str
    # Raw, as-downloaded parquet export for this catalog.
    raw_parquet: Path
    # Precomputed flat-text view (see `flatten_sql`) -- built once so blocking/matching
    # stages don't repeatedly pay a struct/array-extraction cost over a large catalog.
    search_text_parquet: Path
    # Column names as they appear in `search_text_parquet` (post-flatten).
    id_col: str
    product_name_col: str
    category_col: str
    # "exact": category_col is a single scalar string, matched by equality (FNDDS's own
    # WWEIA category is this shape). "array_contains": category_col is an array of tags,
    # matched by list_contains (OFF's categories_tags is this shape). See
    # blocking/rules.py's module docstring for why a rule can specify either kind.
    category_kind: Literal["exact", "array_contains"]
    brand_col: str | None
    search_text_col: str
    # Absolute doc-count floor for "too generic a term to be a useful standalone
    # keyword" (see profiling.py's high_frequency_terms) -- None means the caller
    # should derive one from this catalog's own row count instead of using a fixed
    # number tuned for a different catalog's scale (see
    # profiling.GENERIC_TERM_MIN_DOC_COUNT_FRACTION).
    generic_term_min_doc_count: int | None
    # Builds this catalog's `search_text_parquet` from `raw_parquet` -- the struct/array
    # flattening step, schema-specific per catalog (OFF's implementation is
    # `off_text.flatten`; a differently-shaped catalog supplies its own). Signature:
    # flatten_sql(raw_parquet_path, out_parquet_path) -> None.
    flatten_sql: Callable[[str, str], None]


# The one real instantiation this project actually uses today. Imports config/off_text
# here (not the other way around) specifically to avoid a circular import -- config.py
# stays the source of truth for filesystem paths, off_text.py stays the source of truth
# for OFF's specific flattening SQL, and this module just wires the two together into
# one `CatalogSource` value everything else imports.
from agentic_matching.config import OFF_PARQUET, OFF_SEARCH_TEXT_PARQUET  # noqa: E402
from agentic_matching.off_text import flatten as _off_flatten  # noqa: E402

OFF_SOURCE = CatalogSource(
    name="off",
    display_name="Open Food Facts (OFF)",
    raw_parquet=OFF_PARQUET,
    search_text_parquet=OFF_SEARCH_TEXT_PARQUET,
    id_col="code",
    product_name_col="product_name",
    category_col="categories_tags",
    category_kind="array_contains",
    brand_col="brands",
    search_text_col="search_text",
    generic_term_min_doc_count=None,  # derived from row count -- see profiling.py
    flatten_sql=_off_flatten,
)

# Every consumer that used to hardcode an OFF-specific path/column/constant should read
# it from here instead -- swap this to a different CatalogSource to point the whole
# pipeline at a different food-product database, with no other code changes required.
ACTIVE_CATALOG_SOURCE = OFF_SOURCE

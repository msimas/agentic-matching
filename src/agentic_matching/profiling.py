"""Corpus-wide profiling: token document-frequency and categorical field distributions
computed once over the *full* FNDDS and OFF datasets.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Literal

import duckdb

from agentic_matching.config import (
    FDC_DUCKDB_PATH,
    OFF_PARQUET,
    OFF_SEARCH_TEXT_PARQUET,
    PROFILING_DIR,
    configure_logging,
)

log = logging.getLogger(__name__)

Side = Literal["fndds", "off"]

_TOKEN_DOC_FREQ_PATHS: dict[Side, str] = {
    "fndds": "fndds_token_doc_freq.parquet",
    "off": "off_token_doc_freq.parquet",
}
_META_PATH = PROFILING_DIR / "meta.json"
_OFF_CATEGORY_TAG_FREQ_PATH = PROFILING_DIR / "off_category_tag_freq.parquet"
_FNDDS_WWEIA_CATEGORY_FREQ_PATH = PROFILING_DIR / "fndds_wweia_category_freq.parquet"
_BRANDED_CATEGORY_FREQ_PATH = PROFILING_DIR / "branded_food_category_freq.parquet"

# Tokens appearing in fewer than this many records aren't worth persisting -- they
# can't blow up a block on their own, and dropping them keeps the artifact small.
_MIN_DOC_COUNT = 10


def _token_doc_freq_sql(text_expr: str) -> str:
    return f"""
        WITH toks AS (
            SELECT unique_id, unnest(regexp_extract_all(lower(coalesce({text_expr}, '')), '[a-z]+')) AS token
            FROM texts
        ),
        distinct_toks AS (
            SELECT DISTINCT unique_id, token FROM toks WHERE length(token) >= 4
        )
        SELECT token, count(*) AS doc_count
        FROM distinct_toks
        GROUP BY token
        HAVING count(*) >= {_MIN_DOC_COUNT}
    """


def build(force: bool = False) -> None:
    if _META_PATH.exists() and not force:
        log.info("Profiling artifacts already exist under %s (use force=True to rebuild)", PROFILING_DIR)
        return
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)

    off_path = str(OFF_SEARCH_TEXT_PARQUET).replace("'", "''")
    off_raw_path = str(OFF_PARQUET).replace("'", "''")
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW texts AS SELECT code AS unique_id, product_name FROM read_parquet('{off_path}')")
    n_off = con.execute("SELECT count(*) FROM texts").fetchone()[0]
    con.execute(
        f"COPY ({_token_doc_freq_sql('product_name')}) TO '{PROFILING_DIR / _TOKEN_DOC_FREQ_PATHS['off']}' (FORMAT PARQUET)"
    )
    con.execute(
        f"""
        COPY (
            SELECT tag, count(*) AS doc_count
            FROM (SELECT unnest(categories_tags) AS tag FROM read_parquet('{off_raw_path}'))
            WHERE tag IS NOT NULL
            GROUP BY tag
            HAVING count(*) >= {_MIN_DOC_COUNT}
        ) TO '{_OFF_CATEGORY_TAG_FREQ_PATH}' (FORMAT PARQUET)
        """
    )
    con.close()

    con = duckdb.connect(str(FDC_DUCKDB_PATH), read_only=True)
    con.execute(
        "CREATE OR REPLACE TEMP VIEW texts AS SELECT fdc_id AS unique_id, description AS product_name FROM v_fndds"
    )
    n_fndds = con.execute("SELECT count(*) FROM texts").fetchone()[0]
    con.execute(
        f"COPY ({_token_doc_freq_sql('product_name')}) TO '{PROFILING_DIR / _TOKEN_DOC_FREQ_PATHS['fndds']}' (FORMAT PARQUET)"
    )
    con.execute(
        f"""
        COPY (
            SELECT wweia_food_category_description AS value, count(*) AS doc_count
            FROM v_fndds
            WHERE wweia_food_category_description IS NOT NULL
            GROUP BY 1
        ) TO '{_FNDDS_WWEIA_CATEGORY_FREQ_PATH}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT branded_food_category AS value, count(*) AS doc_count
            FROM v_branded
            WHERE branded_food_category IS NOT NULL
            GROUP BY 1
        ) TO '{_BRANDED_CATEGORY_FREQ_PATH}' (FORMAT PARQUET)
        """
    )
    con.close()

    _META_PATH.write_text(json.dumps({"n_fndds": n_fndds, "n_off": n_off}, indent=2))
    log.info("Built profiling artifacts under %s (n_fndds=%d, n_off=%d)", PROFILING_DIR, n_fndds, n_off)


@functools.lru_cache(maxsize=1)
def _meta() -> dict[str, int]:
    if not _META_PATH.exists():
        raise FileNotFoundError(f"{_META_PATH} not found; run agentic_matching.profiling.build() first.")
    return json.loads(_META_PATH.read_text())


def n_records(side: Side) -> int:
    return _meta()[f"n_{side}"]


@functools.lru_cache(maxsize=2)
def _token_doc_freq(side: Side) -> dict[str, int]:
    path = PROFILING_DIR / _TOKEN_DOC_FREQ_PATHS[side]
    con = duckdb.connect()
    rows = con.execute(f"SELECT token, doc_count FROM read_parquet('{path}')").fetchall()
    con.close()
    return dict(rows)


@functools.lru_cache(maxsize=1)
def _category_freq(kind: Literal["off_categories_tags", "fndds_wweia", "branded_food_category"]) -> dict[str, int]:
    path = {
        "off_categories_tags": _OFF_CATEGORY_TAG_FREQ_PATH,
        "fndds_wweia": _FNDDS_WWEIA_CATEGORY_FREQ_PATH,
        "branded_food_category": _BRANDED_CATEGORY_FREQ_PATH,
    }[kind]
    con = duckdb.connect()
    rows = con.execute(f"SELECT value, doc_count FROM read_parquet('{path}')").fetchall()
    con.close()
    return dict(rows)


def catalog_doc_count(side: Side, token: str) -> int:
    """Exact number of `side` records whose text contains `token` (>= 4 chars,
    case-insensitive), or 0 if it wasn't frequent enough to have been persisted."""
    return _token_doc_freq(side).get(token.lower(), 0)


# Empirically-tuned absolute doc-count floor for treating an OFF-side token as "too
# generic to be a useful standalone keyword": at OFF's ~4.66M-row scale a token needs a
# *count*, not just a top-K rank, to reliably catch generic words -- a fixed top-K list
# let some equally-broad terms (e.g. "green"/"black", each matching ~19-23K records)
# rank just outside the cutoff while still being just as harmful to block size as the
# ones that made the list. There is deliberately no equivalent for FNDDS: FNDDS only has
# ~5.4K rows total, so even a keyword matching a large share of it is not a memory risk,
# and applying an OFF-scale genericness bar there only rejects perfectly good FNDDS
# keywords (e.g. "cooked", "canned") for no safety benefit.
OFF_GENERIC_TERM_MIN_DOC_COUNT = 15_000


def _rank_terms(
    freq: dict[str, int], n: int, k: int, min_len: int, min_doc_count: int | None, max_terms: int
) -> list[dict[str, Any]]:
    """Pure filter/sort/format logic, split out from `high_frequency_terms` so it's
    testable against a small synthetic frequency table without needing the built
    profiling artifact (which requires the full data pipeline to have been run)."""
    candidates = [(tok, c) for tok, c in freq.items() if len(tok) >= min_len]
    if min_doc_count is not None:
        candidates = [(tok, c) for tok, c in candidates if c >= min_doc_count]
        top = sorted(candidates, key=lambda kv: -kv[1])[:max_terms]
    else:
        top = sorted(candidates, key=lambda kv: -kv[1])[:k]
    return [{"term": tok, "doc_count": c, "catalog_fraction": round(c / n, 4) if n else 0.0} for tok, c in top]


def high_frequency_terms(
    side: Side, k: int = 40, min_len: int = 4, min_doc_count: int | None = None, max_terms: int = 200
) -> list[dict[str, Any]]:
    """Catalog-wide common tokens on `side`, with their document count and fraction of
    the catalog -- meant to be shown to the LLM (and, for `side="off"`, used by
    llm/mock.py) as terms that are too generic to usefully narrow a block down on their
    own, whatever block is being proposed.

    With `min_doc_count` given, returns every token at or above that absolute count
    (capped at `max_terms`) rather than a fixed top-`k` -- see `OFF_GENERIC_TERM_MIN_DOC_COUNT`.
    Without it, simply the top-`k` most frequent tokens (informational; not a safety bar).
    """
    return _rank_terms(_token_doc_freq(side), n_records(side), k, min_len, min_doc_count, max_terms)


def top_category_values(
    kind: Literal["off_categories_tags", "fndds_wweia", "branded_food_category"], k: int = 20
) -> list[dict[str, Any]]:
    """The `k` most common values of a categorical field, for grounding matching
    attribute proposals in values that actually occur in the data."""
    freq = _category_freq(kind)
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:k]
    return [{"value": v, "doc_count": c} for v, c in top]


if __name__ == "__main__":
    configure_logging()
    build(force=True)
    print(f"n_fndds={n_records('fndds')} n_off={n_records('off')}")
    print("off high-frequency terms:", [t["term"] for t in high_frequency_terms("off", k=15)])
    print("fndds high-frequency terms:", [t["term"] for t in high_frequency_terms("fndds", k=15)])

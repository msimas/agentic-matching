"""Offline stand-in for the real vLLM-backed client, selected via `LLM_DEVICE=mock`.

This project's real LLM path (llm/server.py + llm/client.py) is fully implemented and
is what should be used for actual runs — see README.md for install instructions. vLLM's
CPU build is not a regular PyPI wheel (the default `vllm` wheel bundles multi-gigabyte
CUDA runtime libraries and assumes an NVIDIA GPU), so it isn't installed in every
environment this code might run in (e.g. this dev sandbox has no usable GPU and no
pre-built CPU wheel available). MockChatClient implements the exact same `ChatClient`
interface with deterministic, keyword-frequency-based heuristics so the blocking,
attribute, and linking agent loops can be exercised, tested, and demoed end-to-end
without a running LLM server. It is intentionally simple and clearly not a substitute
for real LLM reasoning — swap `LLM_DEVICE` away from `mock` to use it for real.
"""

from __future__ import annotations

import functools
import json
import re
from collections import Counter
from typing import Any, Literal

from agentic_matching.attributes.library import SEED_ATTRIBUTES as _LIBRARY_SEED_ATTRIBUTES
from agentic_matching.llm.client import ChatClient

_STOPWORDS = {
    "the", "and", "or", "of", "with", "a", "an", "in", "to", "for", "as", "is", "on",
    "not", "no", "ns", "nfs", "type", "milk", "from", "added", "fat", "other", "than",
    # Generic sensory/marketing descriptors that show up across nearly every food
    # category (not specific to any one block) -- promoting these as blocking keywords
    # is what let the yogurt/beans rules balloon to include unrelated products (e.g.
    # "vanilla"/"cream" pulled in vanilla ice cream and cream cheese as "beans"). A real
    # LLM wouldn't propose these; this mock's naive top-frequency mining needs the
    # explicit stopword to avoid the same mistake.
    "cream", "creamy", "vanilla", "cool", "sweet", "fresh", "chocolate", "flavor",
    "flavors", "flavored", "sauce", "mix", "classic", "light", "lite", "style",
    "select", "value", "premium", "great", "brand", "original", "natural", "organic",
}

_TOKEN_RE = re.compile(r"[a-z]+")


def _doc_freq(texts: list[str], min_len: int = 4) -> tuple[Counter, int]:
    counts: Counter[str] = Counter()
    n = 0
    for t in texts:
        if not t:
            continue
        n += 1
        for tok in {tok for tok in _TOKEN_RE.findall(t.lower()) if len(tok) >= min_len}:
            counts[tok] += 1
    return counts, n


@functools.lru_cache(maxsize=1)
def _background_doc_freq(side: Literal["fndds", "off"]) -> tuple[Counter, int, int]:
    """Token document-frequency over a fixed, block-agnostic sample of the given side's
    whole catalog, plus that side's true total row count (for scaling the sample-based
    frequency up to an estimated catalog-wide match count). Used to reject mined tokens
    that would match a huge absolute number of records catalog-wide (e.g. "protein",
    "rice", "black") rather than actually being specific to the block being mined -- a
    within-block top-frequency count, or even a plain *relative* background frequency,
    can't tell this apart: at OFF's ~4.66M-row scale, even a token that's rare in a
    background sample (a percent or less) still implies tens of thousands of matches,
    which is what kept letting generic ingredient/descriptor words slip into proposed
    keyword lists and blow up block size. Cached (module-lifetime) since it's the same
    regardless of which block is currently being mined.
    """
    import duckdb

    from agentic_matching.config import FDC_DUCKDB_PATH, OFF_SEARCH_TEXT_PARQUET

    if side == "off":
        path = str(OFF_SEARCH_TEXT_PARQUET).replace("'", "''")
        con = duckdb.connect()
        # A deterministic ~1/1000 sample (~4-5K rows) is plenty to estimate catalog-wide
        # frequency and far cheaper than scanning all ~4.66M OFF rows.
        rows = con.execute(
            f"SELECT product_name FROM read_parquet('{path}') WHERE hash(code) % 1000 = 0"
        ).fetchall()
        total = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    else:
        # Matches the (read-write, default-config) connection style the rest of the
        # codebase already holds open against this same file concurrently -- DuckDB
        # refuses a second connection to one file with a *different* configuration
        # (e.g. read_only=True here vs. the caller's read-write connection).
        con = duckdb.connect(str(FDC_DUCKDB_PATH))
        # FNDDS is small enough (~5.4K rows total) to use in full -- sample == total.
        rows = con.execute("SELECT description FROM v_fndds").fetchall()
        total = len(rows)
    con.close()
    counts, n = _doc_freq([r[0] for r in rows])
    return counts, n, total


def _top_tokens(
    texts: list[str],
    side: Literal["fndds", "off"],
    k: int = 6,
    min_len: int = 4,
    max_estimated_matches: int = 15_000,
) -> list[str]:
    counts, _ = _doc_freq(texts, min_len=min_len)
    bg_counts, bg_n, bg_total = _background_doc_freq(side)
    candidates = [tok for tok in counts if tok not in _STOPWORDS]
    candidates.sort(key=lambda t: -counts[t])
    keywords: list[str] = []
    for tok in candidates:
        if len(keywords) >= k:
            break
        estimated_matches = (bg_counts.get(tok, 0) / bg_n) * bg_total if bg_n else 0.0
        if estimated_matches > max_estimated_matches:
            continue  # would match too large a share of the catalog to be a useful keyword
        keywords.append(tok)
    return keywords


# Seed vocab for the two in-scope blocks, used as a floor so the mock's proposals are
# reasonable even on the first round before any text-mining has happened.
_SEED_KEYWORDS = {
    "yogurt": ["yogurt", "yoghurt", "yogourt"],
    "beans": ["bean", "beans", "legume"],
}

_BEANS_ATTRIBUTES = [
    {
        "name": "bean_type",
        "kind": "categorical",
        "description": "Variety of bean",
        "categories": {
            "black": {"fndds_keywords": ["black bean"], "off_keywords": ["black bean", "haricot noir"]},
            "pinto": {"fndds_keywords": ["pinto"], "off_keywords": ["pinto"]},
            "kidney": {"fndds_keywords": ["kidney"], "off_keywords": ["kidney"]},
            "navy": {"fndds_keywords": ["navy bean", "white bean"], "off_keywords": ["navy bean", "white bean"]},
            "garbanzo": {"fndds_keywords": ["garbanzo", "chickpea"], "off_keywords": ["garbanzo", "chickpea", "pois chiche"]},
            "lima": {"fndds_keywords": ["lima"], "off_keywords": ["lima"]},
            "refried": {"fndds_keywords": ["refried"], "off_keywords": ["refried", "refritos"]},
            "lentil": {"fndds_keywords": ["lentil"], "off_keywords": ["lentil", "lentille"]},
        },
    },
    {
        "name": "is_canned",
        "kind": "boolean",
        "description": "Is this a canned/shelf-stable prepared product?",
        "fndds_keywords": ["canned", "from canned"],
        "off_keywords": ["canned", "can ", "boite", "tin"],
    },
    {
        "name": "is_dried",
        "kind": "boolean",
        "description": "Is this a dried/dry bean product (vs. prepared/ready-to-eat)?",
        "fndds_keywords": ["dried", "from dried", "dry"],
        "off_keywords": ["dried", "dry", "sec"],
    },
    {
        "name": "is_seasoned",
        "kind": "boolean",
        "description": "Is this seasoned/flavored (vs. plain)?",
        "fndds_keywords": ["seasoned", "spicy", "chili", "flavored"],
        "off_keywords": ["seasoned", "spicy", "chili", "flavored", "epice"],
    },
    {
        "name": "sodium_level",
        "kind": "categorical",
        "description": "Sodium content tier",
        "categories": {
            "low_sodium": {
                "fndds_keywords": ["low sodium", "no salt", "reduced sodium"],
                "off_keywords": ["low sodium", "no salt", "reduced sodium", "sans sel"],
            },
            "regular": {"fndds_keywords": [], "off_keywords": []},
        },
    },
    {
        "name": "is_organic",
        "kind": "boolean",
        "description": "Is this labeled organic?",
        "fndds_keywords": ["organic"],
        "off_keywords": ["organic", "bio"],
    },
]

_SEED_ATTRIBUTES = {"yogurt": _LIBRARY_SEED_ATTRIBUTES["yogurt"], "beans": _BEANS_ATTRIBUTES}


class MockChatClient(ChatClient):
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload = json.loads(user)
        if "fndds_sample_descriptions" in payload:
            return self._blocking_response(payload)
        if "sample_candidate_pairs" in payload:
            return self._attribute_response(payload)
        raise ValueError(f"MockChatClient: unrecognized prompt payload keys: {list(payload)}")

    # -- blocking -----------------------------------------------------------------

    def _blocking_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        block = payload["block_name"]
        seed = _SEED_KEYWORDS.get(block, [block])
        fndds_mined = _top_tokens(payload.get("fndds_sample_descriptions", []), side="fndds")
        off_mined = _top_tokens(payload.get("off_sample_product_names", []), side="off")

        prev = payload.get("previous_rule")
        metrics = payload.get("previous_round_metrics")

        fndds_kw = sorted(set(seed) | set(fndds_mined))
        off_kw = sorted(set(seed) | set(off_mined))

        if prev and metrics:
            # Widen if pair completeness is low; otherwise keep stable (simulates
            # convergence so the bounded loop can stop early).
            pc = metrics.get("pair_completeness", 1.0)
            if pc < 0.9:
                fndds_kw = sorted(set(prev.get("fndds", {}).get("keywords", [])) | set(fndds_kw))
                off_kw = sorted(set(prev.get("off", {}).get("keywords", [])) | set(off_kw))
            else:
                fndds_kw = prev.get("fndds", {}).get("keywords", fndds_kw)
                off_kw = prev.get("off", {}).get("keywords", off_kw)

        return {
            "fndds": {"keywords": fndds_kw, "exclude_keywords": []},
            "off": {"keywords": off_kw, "exclude_keywords": []},
            "rationale": (
                f"[mock] seed vocabulary + top frequent tokens mined from the '{block}' "
                "sample descriptions on each side."
            ),
        }

    # -- attributes -----------------------------------------------------------------

    def _attribute_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        block = payload["block_name"]
        existing = payload.get("existing_attributes")
        correlation_flags = payload.get("correlation_flags") or []

        if existing is None:
            attrs = _SEED_ATTRIBUTES.get(block, [])
            return {
                "attributes": attrs,
                "rationale": f"[mock] seed attribute set for block '{block}'.",
            }

        # Revision: drop the second attribute in each correlated pair (simple
        # deterministic redundancy resolution), otherwise keep unchanged.
        to_drop = {flag["attribute_b"] for flag in correlation_flags if "attribute_b" in flag}
        attrs = [a for a in existing if a["name"] not in to_drop]
        return {
            "attributes": attrs,
            "rationale": (
                f"[mock] dropped {sorted(to_drop)} for redundancy with a more informative "
                "attribute." if to_drop else "[mock] attribute set looks stable; no changes."
            ),
        }

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

import json
import re
from collections import Counter
from typing import Any

from agentic_matching.attributes.seed_rules import SEED_ATTRIBUTES as _LIBRARY_SEED_ATTRIBUTES
from agentic_matching.blocking.rules import NEVER_USEFUL_KEYWORDS as _STOPWORDS
from agentic_matching.llm.client import ChatClient

# _STOPWORDS (aliased from blocking/rules.py's NEVER_USEFUL_KEYWORDS -- one shared
# source of truth, since it's now also hard-enforced there on every proposed rule
# regardless of source) filters out plain function/connector words -- these need no
# data, they're never useful keywords regardless of catalog frequency. Domain
# genericness beyond that (e.g. "protein", "rice", "black" each matching tens of
# thousands of unrelated OFF records) is NOT handled by a hand-curated list -- that was
# tried and is whack-a-mole (some new generic word always slips through). Instead it's
# handled by `_reject_terms`, which reads the same `corpus_stats` (see profiling.py /
# llm/prompts.py) a real LLM would be shown, so this mock's judgment about what's "too
# generic" is grounded in the same real catalog data rather than a list someone has to
# keep extending by hand.

_TOKEN_RE = re.compile(r"[a-z]+")


def _doc_freq(texts: list[str], min_len: int = 4) -> Counter:
    counts: Counter[str] = Counter()
    for t in texts:
        if not t:
            continue
        for tok in {tok for tok in _TOKEN_RE.findall(t.lower()) if len(tok) >= min_len}:
            counts[tok] += 1
    return counts


def _reject_terms(corpus_stats: dict[str, Any] | None, side: str) -> set[str]:
    """Terms profiling.py flagged as catalog-wide common for `side` -- i.e. too generic
    to usefully narrow any block down on their own (see llm/prompts.py's
    corpus_stats/catalog_wide_common_terms docs)."""
    if not corpus_stats:
        return set()
    return {t["term"] for t in corpus_stats.get(side, {}).get("catalog_wide_common_terms", [])}


def _top_tokens(texts: list[str], reject: set[str], k: int = 6, min_len: int = 4) -> list[str]:
    counts = _doc_freq(texts, min_len=min_len)
    candidates = [tok for tok in counts if tok not in _STOPWORDS and tok not in reject]
    candidates.sort(key=lambda t: -counts[t])
    return candidates[:k]


def _matching_categories(category_options: list[dict[str, Any]], block: str, k: int = 5) -> list[str]:
    """Category values (from blocking/agent_loop.py's _category_options) whose own
    name contains the block's name -- e.g. "Yogurt, regular" / "en:yogurts" for block
    "yogurt". A simple substring check is enough here precisely because these are
    short, clean, human-curated labels (unlike free text), so it doesn't need the
    genericity machinery keyword mining does."""
    block_singular = block[:-1] if block.endswith("s") else block
    needles = {block.lower(), block_singular.lower()}
    matches = [
        c["value"]
        for c in sorted(category_options, key=lambda c: -c["count"])
        if any(n in c["value"].lower() for n in needles)
    ]
    return matches[:k]


# Seed vocab for the in-scope blocks, used as a floor so the mock's proposals are
# reasonable even on the first round before any text-mining has happened. No single
# word anchors "breaded_vegetables" well (see blocking/metrics.py's
# CANONICAL_BLOCK_TERMS docstring), so its seed is several specific dish names instead.
_SEED_KEYWORDS = {
    "yogurt": ["yogurt", "yoghurt", "yogourt"],
    "beans": ["bean", "beans", "legume"],
    "breaded_vegetables": ["onion ring", "fried okra", "fried mushroom", "vegetable tempura", "pakora"],
}

# Categorical attributes requiring domain-synonym grouping (e.g. "garbanzo"/"chickpea"/
# "pois chiche" are all the same bean variety; "low sodium"/"no salt"/"sans sel" are all
# the same sodium tier) -- recognizing that is exactly the kind of world-knowledge
# reasoning a real LLM should do better than frequency counting, so this mock keeps a
# small hand-curated exception for it per from-scratch block (only "beans" currently).
# Every *boolean* attribute, in contrast, is now mined from the block's own data (see
# _mined_boolean_attributes) rather than hand-curated -- this is the part any new
# from-scratch block gets "for free" without someone having to write it by hand.
_CATEGORICAL_EXCEPTIONS: dict[str, list[dict[str, Any]]] = {
    "beans": [
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
    ],
}

_MAX_MINED_ATTRIBUTES = 6


def _mined_boolean_attributes(candidate_terms: list[dict[str, Any]], k: int = _MAX_MINED_ATTRIBUTES) -> list[dict[str, Any]]:
    """Turn the top `k` mined candidate terms (see
    attributes/agent_loop.py::_candidate_boolean_terms) into boolean attribute
    definitions -- the literal token, applied identically on both sides, since this
    mock has no way to recognize a cross-language synonym for it (see that function's
    docstring)."""
    attrs = []
    for c in candidate_terms[:k]:
        term = c["term"]
        attrs.append(
            {
                "name": f"has_{term}",
                "kind": "boolean",
                "description": f"Does the text mention '{term}'?",
                "fndds_keywords": [term],
                "off_keywords": [term],
            }
        )
    return attrs


_SEED_ATTRIBUTES = {"yogurt": _LIBRARY_SEED_ATTRIBUTES["yogurt"]}


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
        corpus_stats = payload.get("corpus_stats")
        # Only the OFF side (~4.66M rows) carries real memory risk from an overly-broad
        # keyword -- FNDDS (~5.4K rows total) doesn't need (and shouldn't get) the same
        # genericness bar; see profiling.OFF_GENERIC_TERM_MIN_DOC_COUNT.
        fndds_mined = _top_tokens(payload.get("fndds_sample_descriptions", []), reject=set())
        off_mined = _top_tokens(
            payload.get("off_sample_product_names", []), reject=_reject_terms(corpus_stats, "off")
        )

        # Categories whose own (short, human-curated) name contains the block's name --
        # a structured membership signal, far more precise than keyword substring
        # matching against free text, and immune to the boilerplate-annotation problem
        # keywords have (see rules.py's module docstring). Doesn't use the withheld
        # calibration ground-truth term (CANONICAL_BLOCK_TERMS) -- block_name itself is
        # already given in every prompt regardless.
        category_options = payload.get("category_options") or {}
        fndds_categories = set(_matching_categories(category_options.get("fndds", []), block))
        off_categories = set(_matching_categories(category_options.get("off", []), block))

        prev = payload.get("previous_rule")
        metrics = payload.get("previous_round_metrics")
        # Carried forward unconditionally (not gated on `metrics` existing): a seed rule
        # (see blocking/seed_rules.py) or a prior round's own additions are things
        # already known to be needed/useful, not something this mock has any mechanism
        # to propose on its own, so the only sensible default is "don't drop what's
        # already there" -- this matters most for round 0 *with* a seed rule but no
        # metrics yet, which the keyword-widening logic below (gated on `metrics`)
        # doesn't otherwise account for.
        fndds_exclude = (prev or {}).get("fndds", {}).get("exclude_keywords", [])
        off_exclude = (prev or {}).get("off", {}).get("exclude_keywords", [])
        fndds_categories |= set((prev or {}).get("fndds", {}).get("categories", []))
        off_categories |= set((prev or {}).get("off", {}).get("categories", []))
        seed_fndds_kw = set(seed) | set((prev or {}).get("fndds", {}).get("keywords", []))
        seed_off_kw = set(seed) | set((prev or {}).get("off", {}).get("keywords", []))

        # When a clean matching category exists, trust it and don't widen beyond the
        # seed vocabulary via speculative keyword mining: mined candidates are drawn
        # from real in-block sample text, but words that are common there can *also* be
        # common in totally unrelated foods' own descriptions (e.g. "whole"/"fruit"/
        # "plain" show up in "whole wheat muffins", "fruit salad", "plain pretzels" --
        # not just yogurt), and this mock has no way to distinguish that from a
        # genuinely block-specific term the way a real LLM's world knowledge could.
        # Verified: for yogurt, seed+category alone is 62 FNDDS records with zero false
        # positives, vs. 402 records (85% false positives) once "flavors"/"fruit"/
        # "plain"/"whole" get mined in on top. Falls back to mining when no matching
        # category was found (e.g. a block with no clean corresponding taxonomy entry).
        fndds_kw = sorted(seed_fndds_kw) if fndds_categories else sorted(seed_fndds_kw | set(fndds_mined))
        off_kw = sorted(seed_off_kw) if off_categories else sorted(seed_off_kw | set(off_mined))
        fndds_categories = sorted(fndds_categories)
        off_categories = sorted(off_categories)

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
            fndds_categories = prev.get("fndds", {}).get("categories", fndds_categories)
            off_categories = prev.get("off", {}).get("categories", off_categories)

        return {
            "fndds": {"keywords": fndds_kw, "categories": fndds_categories, "exclude_keywords": fndds_exclude},
            "off": {"keywords": off_kw, "categories": off_categories, "exclude_keywords": off_exclude},
            "rationale": (
                f"[mock] seed vocabulary + top frequent tokens mined from the '{block}' "
                "sample descriptions on each side, plus any on-topic category found "
                "among plausibly in-block records."
            ),
        }

    # -- attributes -----------------------------------------------------------------

    def _attribute_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        block = payload["block_name"]
        existing = payload.get("existing_attributes")
        correlation_flags = payload.get("correlation_flags") or []

        if existing is None:
            if block in _SEED_ATTRIBUTES:
                return {
                    "attributes": _SEED_ATTRIBUTES[block],
                    "rationale": f"[mock] seed attribute set for block '{block}'.",
                }
            # No seed (from-scratch block, e.g. beans): a small hand-curated exception
            # for categorical attributes needing domain-synonym grouping, plus boolean
            # attributes mined from this block's own data (see _mined_boolean_attributes).
            categorical = _CATEGORICAL_EXCEPTIONS.get(block, [])
            mined = _mined_boolean_attributes(payload.get("candidate_terms") or [])
            return {
                "attributes": categorical + mined,
                "rationale": (
                    f"[mock] {len(categorical)} hand-curated categorical attribute(s) + "
                    f"{len(mined)} boolean attribute(s) mined from this block's own text."
                ),
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

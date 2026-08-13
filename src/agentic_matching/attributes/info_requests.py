"""Fulfills a "need_more_info" response's `requested` list (see llm/prompts.py's
_NEED_MORE_INFO_NOTE) using nothing but plain in-memory counting/filtering over text
already loaded for the round -- no new query capability, no LLM call, no database
round-trip. Deliberately narrow: exactly the two request "kind"s the prompt itself
documents (term_frequency, sample_records), both backed by data every caller of this
module already has in hand (fndds_texts/catalog_texts), consistent with this being a
lightweight, bounded mechanism rather than open-ended tool use -- see
attributes/revision.py's module docstring for the fuller rationale.
"""

from __future__ import annotations

from typing import Any

MAX_REQUESTS = 3  # matches the cap stated in the prompt itself
MAX_SAMPLE_RECORDS = 5


def _term_frequency(term: str, fndds_texts: list[Any], catalog_texts: list[Any]) -> dict[str, Any]:
    term_lower = term.lower()
    fndds_hits = sum(1 for t in fndds_texts if isinstance(t, str) and term_lower in t.lower())
    catalog_hits = sum(1 for t in catalog_texts if isinstance(t, str) and term_lower in t.lower())
    return {
        "kind": "term_frequency",
        "term": term,
        "fndds_count": fndds_hits,
        "fndds_fraction": round(fndds_hits / len(fndds_texts), 4) if fndds_texts else 0.0,
        "catalog_count": catalog_hits,
        "catalog_fraction": round(catalog_hits / len(catalog_texts), 4) if catalog_texts else 0.0,
    }


def _sample_records(term: str, side: str, fndds_texts: list[Any], catalog_texts: list[Any]) -> dict[str, Any]:
    side = side if side in ("fndds", "catalog") else "fndds"
    texts = fndds_texts if side == "fndds" else catalog_texts
    term_lower = term.lower()
    matches = [t for t in texts if isinstance(t, str) and term_lower in t.lower()][:MAX_SAMPLE_RECORDS]
    return {"kind": "sample_records", "term": term, "side": side, "examples": matches}


def fulfill_requests(requested: list[dict[str, Any]], fndds_texts: list[Any], catalog_texts: list[Any]) -> list[dict[str, Any]]:
    """Returns one answer dict per request in `requested` (capped at MAX_REQUESTS,
    matching the prompt's own stated limit -- a request beyond that is silently
    dropped, not an error, since it's the model exceeding a documented bound, not a
    caller mistake)."""
    answers = []
    for req in (requested or [])[:MAX_REQUESTS]:
        kind = req.get("kind")
        term = req.get("term")
        if kind == "term_frequency" and term:
            answers.append(_term_frequency(term, fndds_texts, catalog_texts))
        elif kind == "sample_records" and term:
            answers.append(_sample_records(term, req.get("side", "fndds"), fndds_texts, catalog_texts))
        else:
            answers.append({"kind": kind, "error": "unrecognized or incomplete request, ignored"})
    return answers

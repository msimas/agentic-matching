"""A matching attribute is a keyword-based extraction rule, computed identically on
both the FNDDS side and the OFF side -- the attribute-stage counterpart to
blocking/rules.py's blocking-rule predicates.

An attribute definition (as produced by llm/prompts.py's attribute schema) is either:
  - boolean:      {"kind": "boolean", "fndds_keywords": [...], "off_keywords": [...]}
  - categorical:  {"kind": "categorical", "categories": {cat_name: {"fndds_keywords":
                   [...], "off_keywords": [...]}, ...}}

This module is the single place that turns a definition + a side's raw text into a
value, so attributes/metrics.py's correlation check and linking/splink_model.py's
comparison construction stay consistent.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

log = logging.getLogger(__name__)

Side = Literal["fndds", "off"]


def validate_attribute(attr: dict[str, Any]) -> str | None:
    """Returns a reason string if `attr` is structurally malformed -- would evaluate to
    the exact same value for EVERY record on one side, providing zero matching signal
    -- or None if it looks usable.

    A boolean attribute with an empty (or missing) keyword list on a side isn't just
    weak, it's a silent no-op: apply_attribute's `bool(kws) and any(...)` is False
    whenever `kws` is empty, so the attribute is `False` for every record on that side
    regardless of what the text actually says -- verified real case: an LLM revision
    for this project's "beans" block returned `contains_meat` with its
    "description" field updated but `fndds_keywords`/`off_keywords` entirely absent
    from the response, silently turning a real (if incomplete) attribute into one that
    could never be True again. This is checked on the DEFINITION itself, not computed
    values, so it catches the problem immediately after the LLM response is parsed --
    before it's persisted to an artifact as if it were a real, usable attribute, and
    before splink_model ever builds comparisons from it.

    Deliberately does NOT flag an attribute whose keywords are well-formed but simply
    never match in this particular block's data (e.g. a legitimately rare, real
    distinction) -- that's a data/frequency question, not a structural one, and
    splink_model._drop_unobservable_attrs (a separate, computed-value-based check)
    already handles the analogous "always null" case for categorical attributes.
    """
    name = attr.get("name", "<unnamed>")
    kind = attr.get("kind")
    if kind == "boolean":
        for side in ("fndds", "off"):
            if not attr.get(f"{side}_keywords"):
                return f"{name!r}: boolean attribute has no {side}_keywords (would always evaluate False on {side})"
    elif kind == "categorical":
        categories = attr.get("categories") or {}
        if len(categories) < 2:
            return f"{name!r}: categorical attribute has fewer than 2 categories"
        for side in ("fndds", "off"):
            if not any(cat_def.get(f"{side}_keywords") for cat_def in categories.values()):
                return f"{name!r}: no category has any {side}_keywords (would always evaluate null on {side})"
    return None


def filter_valid_attributes(attrs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop structurally malformed attributes from an LLM response (see
    validate_attribute), logging why each one was dropped. Call this immediately after
    parsing `response["attributes"]`, in every agent loop that accepts an LLM-proposed
    or -revised attribute set."""
    kept = []
    for attr in attrs:
        reason = validate_attribute(attr)
        if reason:
            log.warning("Dropping malformed attribute from LLM response: %s", reason)
            continue
        kept.append(attr)
    return kept


def apply_attribute(attr: dict[str, Any], text: str | None, side: Side) -> Any:
    """Compute one attribute's value for one record's text on one side."""
    text_l = (text or "").lower()
    key = f"{side}_keywords"
    if attr["kind"] == "boolean":
        kws = attr.get(key, [])
        return bool(kws) and any(kw.lower() in text_l for kw in kws if kw)
    if attr["kind"] == "categorical":
        fallback_category = None
        for cat_name, cat_def in attr.get("categories", {}).items():
            kws = cat_def.get(key, [])
            if kws and any(kw.lower() in text_l for kw in kws if kw):
                return cat_name
            if not cat_def.get("fndds_keywords") and not cat_def.get("off_keywords"):
                fallback_category = cat_name
        return fallback_category
    raise ValueError(f"Unknown attribute kind: {attr['kind']!r}")


def compute_attribute_values(
    attrs: list[dict[str, Any]], texts: list[str | None], side: Side
) -> dict[str, list[Any]]:
    """Vectorized (well, list-comprehension) computation of every attribute for a list
    of records' text on one side. Returns {attribute_name: [values...]}."""
    return {attr["name"]: [apply_attribute(attr, t, side) for t in texts] for attr in attrs}

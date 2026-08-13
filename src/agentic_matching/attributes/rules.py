"""A matching attribute is a keyword-based extraction rule, computed identically on
both the FNDDS side and the OFF side -- the attribute-stage counterpart to
blocking/rules.py's blocking-rule predicates.

An attribute definition (as produced by llm/prompts.py's attribute schema) is either:
  - boolean:      {"kind": "boolean", "fndds_keywords": [...], "catalog_keywords": [...]}
  - categorical:  {"kind": "categorical", "categories": {cat_name: {"fndds_keywords":
                   [...], "catalog_keywords": [...]}, ...}}

This module is the single place that turns a definition + a side's raw text into a
value, so attributes/metrics.py's correlation check and linking/splink_model.py's
comparison construction stay consistent.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

log = logging.getLogger(__name__)

Side = Literal["fndds", "catalog"]


def validate_attribute(attr: dict[str, Any]) -> str | None:
    """Returns a reason string if `attr` is structurally malformed -- would evaluate to
    the exact same value for EVERY record on one side, providing zero matching signal
    -- or None if it looks usable.

    A boolean attribute with an empty (or missing) keyword list on a side isn't just
    weak, it's a silent no-op: apply_attribute's `bool(kws) and any(...)` is False
    whenever `kws` is empty, so the attribute is `False` for every record on that side
    regardless of what the text actually says -- verified real case: an LLM revision
    for this project's "beans" block returned `contains_meat` with its
    "description" field updated but `fndds_keywords`/`catalog_keywords` entirely absent
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
        for side in ("fndds", "catalog"):
            if not attr.get(f"{side}_keywords"):
                return f"{name!r}: boolean attribute has no {side}_keywords (would always evaluate False on {side})"
    elif kind == "categorical":
        categories = attr.get("categories") or {}
        if len(categories) < 2:
            return f"{name!r}: categorical attribute has fewer than 2 categories"
        for side in ("fndds", "catalog"):
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
            if not cat_def.get("fndds_keywords") and not cat_def.get("catalog_keywords"):
                fallback_category = cat_name
        return fallback_category
    raise ValueError(f"Unknown attribute kind: {attr['kind']!r}")


def compute_attribute_values(
    attrs: list[dict[str, Any]], texts: list[str | None], side: Side
) -> dict[str, list[Any]]:
    """Vectorized (well, list-comprehension) computation of every attribute for a list
    of records' text on one side. Returns {attribute_name: [values...]}."""
    return {attr["name"]: [apply_attribute(attr, t, side) for t in texts] for attr in attrs}


def attribute_set_signature(attrs: list[dict[str, Any]]) -> frozenset[str]:
    """A content-sensitive fingerprint of an attribute set, for "did anything actually
    change" checks -- comparing only `{a["name"] for a in attrs}` (as both agent loops
    used to) is wrong: a "redefine" that keeps an attribute's NAME but changes its
    keywords/categories looks identical under a name-only comparison, so the loop
    reports "no changes" and stops even though real content changed. Verified real
    case (beans, LLM_DEVICE=databricks, decomposed revision -- see attributes/
    revision.py): a round that redefined beans_bean_type's and beans_preparation_
    method's keywords (same names, new content) was logged as "LLM proposed no
    attribute changes" and stopped the loop after round 0, discarding a real
    revision. Order-independent (a set of per-attribute signatures, not a single
    signature of the whole list) since attribute order was never meaningful."""
    return frozenset(json.dumps(a, sort_keys=True, default=str) for a in attrs)


def filter_reintroduced_attributes(
    new_attrs: list[dict[str, Any]], auto_dropped: list[str], dropped_definitions: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Backstop against an LLM re-proposing an attribute name already proven
    non-discriminating this run -- content-aware, not name-only, for the same reason
    attribute_set_signature (above) isn't name-only: comparing purely by name silently
    discards a genuinely DIFFERENT redefinition just because it reused a burned name.

    Verified real case (beans, LLM_DEVICE=databricks): a gap-identification call
    correctly re-identified a "preparation method" gap (now with concept_history
    explaining what failed before -- see attributes/revision.py), the definition step
    proposed a fresh `beans_preparation_method` definition for it, and a name-only
    filter threw it away unseen purely because that name had failed once before -- in
    the same round `beans_bean_type` was also dropped with nothing replacing it,
    collapsing the round to 2 weak attributes and cratering recall (candidate pairs
    18,743 -> 2,409).

    `dropped_definitions` maps a name to the exact attribute dict it had AT THE MOMENT
    it was auto-dropped (the caller must capture this at drop time, before any later
    round's proposals overwrite what that name currently means). Only a reintroduction
    matching that frozen definition exactly is filtered; anything else -- a real
    redefinition reusing the name -- passes through to be retrained and judged on its
    own merits, same as any other new attribute (the caller's own retrain-on-collapse
    loop still catches it if it's still bad, at the cost of one round -- just no longer
    pre-judged unseen).

    Returns (filtered_new_attrs, same_as_before_names, new_definition_names) -- the
    caller logs the two name lists differently (a real problem vs. an FYI)."""
    same_as_before = [
        a["name"]
        for a in new_attrs
        if a["name"] in auto_dropped
        and a["name"] in dropped_definitions
        and attribute_set_signature([a]) == attribute_set_signature([dropped_definitions[a["name"]]])
    ]
    filtered = [a for a in new_attrs if a["name"] not in same_as_before]
    reintroduced_new_definition = [a["name"] for a in filtered if a["name"] in auto_dropped]
    return filtered, same_as_before, reintroduced_new_definition

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

from typing import Any, Literal

Side = Literal["fndds", "off"]


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

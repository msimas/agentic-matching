"""Matching-attribute definitions and the (shared) keyword-based extraction logic used
to compute them on both the FNDDS side and the OFF side.

An attribute definition (as produced by llm/prompts.py's attribute schema) is either:
  - boolean:      {"kind": "boolean", "fndds_keywords": [...], "off_keywords": [...]}
  - categorical:  {"kind": "categorical", "categories": {cat_name: {"fndds_keywords":
                   [...], "off_keywords": [...]}, ...}}

This module is the single place that turns a definition + a side's raw text into a
value, so blocking's correlation check (attributes/correlation_check.py) and the
splink comparison construction (linking/splink_model.py) stay consistent.

`SEED_ATTRIBUTES["yogurt"]` is the attribute set given verbatim in the project's vision
document (is_greek, is_plain, is_drink, is_baby, fruit_flavored, contains_cereal,
fat_level); no seed is provided for "beans" — that attribute set is left for the LLM
agent loop to propose from scratch, as intended.
"""

from __future__ import annotations

from typing import Any, Literal

Side = Literal["fndds", "off"]

SEED_ATTRIBUTES: dict[str, list[dict[str, Any]]] = {
    "yogurt": [
        {
            "name": "is_greek",
            "kind": "boolean",
            "description": "Is this a Greek-style yogurt?",
            "fndds_keywords": ["greek"],
            "off_keywords": ["greek", "grec"],
        },
        {
            "name": "is_plain",
            "kind": "boolean",
            "description": "Is this a plain (unflavored) yogurt?",
            "fndds_keywords": ["plain"],
            "off_keywords": ["plain", "nature", "natural"],
        },
        {
            "name": "is_drink",
            "kind": "boolean",
            "description": "Is this a drinkable yogurt product?",
            "fndds_keywords": ["drink", "smoothie", "kefir"],
            "off_keywords": ["drink", "smoothie", "kefir", "a boire"],
        },
        {
            "name": "is_baby",
            "kind": "boolean",
            "description": "Is this targeted at infants/toddlers?",
            "fndds_keywords": ["baby food", "infant"],
            "off_keywords": ["baby", "infant", "toddler"],
        },
        {
            "name": "fruit_flavored",
            "kind": "boolean",
            "description": "Is this flavored using fruit?",
            "fndds_keywords": ["fruit", "strawberry", "blueberry", "peach", "berry"],
            "off_keywords": ["fruit", "strawberry", "blueberry", "peach", "berry", "fraise"],
        },
        {
            "name": "contains_cereal",
            "kind": "boolean",
            "description": "Does this include cereal (granola, oat bran, corn flakes, etc.)?",
            "fndds_keywords": ["granola", "cereal", "oat", "muesli"],
            "off_keywords": ["granola", "cereal", "oat", "muesli"],
        },
        {
            "name": "fat_level",
            "kind": "categorical",
            "description": "Fat content tier",
            "categories": {
                "fat_free": {
                    "fndds_keywords": ["nonfat", "fat free", "skim"],
                    "off_keywords": ["fat free", "0% fat", "skimmed", "nonfat"],
                },
                "low_fat": {
                    "fndds_keywords": ["low fat", "lowfat", "low-fat"],
                    "off_keywords": ["low fat", "light", "1%", "2%"],
                },
                "full_fat": {
                    "fndds_keywords": ["whole milk"],
                    "off_keywords": ["whole milk", "full fat", "entier"],
                },
            },
        },
    ],
    # "beans": intentionally no seed -- proposed by the LLM agent loop from scratch.
}


def get_seed_attributes(block_name: str) -> list[dict[str, Any]] | None:
    return SEED_ATTRIBUTES.get(block_name)


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

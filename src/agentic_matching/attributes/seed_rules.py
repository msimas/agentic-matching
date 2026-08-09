"""Hand-authored seed data for the attribute-generation agent loop -- the
attribute-stage counterpart to blocking/seed_rules.py.

Two independent things live here, both loaded from seed_rules.json so they can be
edited by a domain expert without touching Python:

  - SEED_ATTRIBUTES["yogurt"]: the *complete, immutable* attribute set given verbatim
    in the project's vision document (is_greek, is_plain, is_drink, is_baby,
    fruit_flavored, contains_cereal, fat_level); no seed is provided for "beans" or
    "breaded_vegetables" -- those attribute sets are left for the LLM agent loop to
    propose from scratch, as intended.

  - SEED_ATTRIBUTE_NOTES: freeform prompt fragments for blocks that need domain
    guidance the structured {name, kind, fndds_keywords/off_keywords} attribute schema
    can't express on its own -- e.g. "these two attributes must agree on WHICH specific
    thing, not just that both have *a* thing of that general kind". Unlike
    SEED_ATTRIBUTES (yogurt's complete, hand-authored attribute set), a note here
    doesn't replace the LLM's own proposal -- it is echoed alongside it (see
    agent_loop.build_attribute_prompt's "guidance" parameter) on every round, as
    persistent context, not something the LLM's revision iterates away.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SEED_RULES_PATH = Path(__file__).resolve().parent / "seed_rules.json"
_seed_data: dict[str, Any] = json.loads(_SEED_RULES_PATH.read_text())

SEED_ATTRIBUTES: dict[str, list[dict[str, Any]]] = _seed_data["seed_attributes"]
SEED_ATTRIBUTE_NOTES: dict[str, str] = _seed_data["notes"]


def get_seed_attributes(block_name: str) -> list[dict[str, Any]] | None:
    return SEED_ATTRIBUTES.get(block_name)


def get_seed_attribute_notes(block_name: str) -> str | None:
    return SEED_ATTRIBUTE_NOTES.get(block_name)

"""Optional hand-authored starting point for a block's blocking rule -- the blocking-
stage counterpart to attributes/seed_rules.py's SEED_ATTRIBUTES (yogurt's fixed
vision-specified attribute set). Where SEED_ATTRIBUTES is a *complete, immutable*
attribute set for yogurt, a seed rule here is just a *starting point* for round 0: the
agent loop (mock or real LLM) is free to -- and expected to -- refine it using the
samples, corpus stats, and category options it's shown, same as any other round's
revision (see llm/prompts.py's "starting point" instruction wording for a seeded round).

Most useful for domain knowledge that isn't derivable from a handful of samples (e.g. an
SME already knows "breaded" as a keyword will pull in seafood/cheese products alongside
vegetables, before ever running the loop and seeing that in a materialized block) --
this is a place to encode that up front rather than rediscovering it every run.

The actual seed data lives in seed_rules.json, not here, so it can be edited by a
domain expert without touching Python -- this module is just the loader plus the two
accessor functions the rest of the codebase calls. A block absent from the JSON has no
seed (yogurt and beans both originally proposed blocking rules fully from scratch, as
described in PLAN.md) -- add an entry only when you have real domain knowledge worth
seeding.

An entry may also carry a "notes" key: a freeform prompt fragment (not a structured
keyword/category/exclude_keywords list) for domain knowledge that doesn't fit that
schema -- e.g. "these two dish names must refer to the same specific vegetable, not just
both be 'a fried vegetable'". Unlike "fndds"/"off", which only seed round 0 (the LLM's
own revised rule replaces them from round 1 on -- see run_blocking_agent), "notes" is
echoed to the LLM on *every* round via build_blocking_prompt, since it's persistent
guidance about the block's domain, not part of the rule state being iterated on.

A side's rule dict may also carry an "exclude_keywords_rationale" key: a freeform note
explaining *why* those specific exclude_keywords were chosen (e.g. verified
false-positive/false-negative counts from a real materialized block) -- purely
documentation for whoever edits the JSON next, never read by any agent loop or prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SEED_RULES_PATH = Path(__file__).resolve().parent / "seed_rules.json"

SEED_BLOCKING_RULES: dict[str, dict[str, Any]] = json.loads(_SEED_RULES_PATH.read_text())


def get_seed_rule(block_name: str) -> dict[str, Any] | None:
    return SEED_BLOCKING_RULES.get(block_name)


def get_seed_notes(block_name: str) -> str | None:
    return SEED_BLOCKING_RULES.get(block_name, {}).get("notes")

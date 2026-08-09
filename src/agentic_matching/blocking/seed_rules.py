"""Optional hand-authored starting point for a block's blocking rule -- the blocking-
stage counterpart to attributes/library.py's SEED_ATTRIBUTES (yogurt's fixed
vision-specified attribute set). Where SEED_ATTRIBUTES is a *complete, immutable*
attribute set for yogurt, a seed rule here is just a *starting point* for round 0: the
agent loop (mock or real LLM) is free to -- and expected to -- refine it using the
samples, corpus stats, and category options it's shown, same as any other round's
revision (see llm/prompts.py's "starting point" instruction wording for a seeded round).

Most useful for domain knowledge that isn't derivable from a handful of samples (e.g. an
SME already knows "breaded" as a keyword will pull in seafood/cheese products alongside
vegetables, before ever running the loop and seeing that in a materialized block) --
this is a place to encode that up front rather than rediscovering it every run.

Unlike attributes/library.py's SEED_ATTRIBUTES, this dict starts empty by default for
every block (yogurt and beans both propose blocking rules fully from scratch, as
described in PLAN.md) -- add an entry only when you have real domain knowledge worth
seeding.

An entry may also carry a "notes" key: a freeform prompt fragment (not a structured
keyword/category/exclude_keywords list) for domain knowledge that doesn't fit that
schema -- e.g. "these two dish names must refer to the same specific vegetable, not just
both be 'a fried vegetable'". Unlike "fndds"/"off", which only seed round 0 (the LLM's
own revised rule replaces them from round 1 on -- see run_blocking_agent), "notes" is
echoed to the LLM on *every* round via build_blocking_prompt, since it's persistent
guidance about the block's domain, not part of the rule state being iterated on.
"""

from __future__ import annotations

from typing import Any

SEED_BLOCKING_RULES: dict[str, dict[str, Any]] = {
    "breaded_vegetables": {
        "fndds": {
            "keywords": ["fried vegetable", "breaded vegetable"],
            "categories": ["Fried vegetables"],
            "exclude_keywords": ["fry", "fries", "french fries", "fish", "shrimp", "prawn", "tilapia", "haddock", "flounder", "cod", "cheese", "chicken"],
        },
        "off": {
            "keywords": ["breaded", "onion ring", "pakora", "tempura"],
            "categories": ["en:breaded-onion-rings", "en:pakora"],
            # "breaded" alone pulls in breaded seafood/cheese products too (shrimp,
            # tilapia, haddock, flounder, mozzarella sticks, fish sticks) -- verified
            # against a real materialized block (LLM_DEVICE=ollama): 6,753 OFF records,
            # a meaningful share of them breaded fish/shrimp/cheese, not vegetables.
            "exclude_keywords": ["fish", "shrimp", "prawn", "tilapia", "haddock", "flounder", "cod", "cheese", "chicken", "fry", "fries", "french fries"],
        },
        "notes": (
            "This block covers several distinct vegetables prepared fried/breaded "
            "(onion, mushroom, cauliflower, eggplant, squash, green tomato, green bean, "
            "broccoli, pickle, sweet potato, ...), not one single dish -- a record is "
            "in-block if it's ANY of these, so don't narrow the keyword/category lists "
            "down to only the vegetable(s) that happen to dominate the samples you're "
            "shown."
        ),
    },
}


def get_seed_rule(block_name: str) -> dict[str, Any] | None:
    return SEED_BLOCKING_RULES.get(block_name)


def get_seed_notes(block_name: str) -> str | None:
    return SEED_BLOCKING_RULES.get(block_name, {}).get("notes")

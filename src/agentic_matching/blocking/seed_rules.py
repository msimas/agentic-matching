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
    },
}


def get_seed_rule(block_name: str) -> dict[str, Any] | None:
    return SEED_BLOCKING_RULES.get(block_name)

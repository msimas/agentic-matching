"""Prompt templates for the two agentic loops: blocking-rule proposal/revision and
matching-attribute proposal/revision. Each builder returns (system, user) message
strings; the LLM is always instructed to reply with a single JSON object matching a
documented schema, since `ChatClient.complete_json` parses the reply as JSON.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------

_BLOCKING_SYSTEM = """\
You are a subject-matter-expert-in-the-loop assistant helping construct record-linkage \
blocking rules between USDA FoodData Central FNDDS records and Open Food Facts (OFF) \
records, for a single product category ("block").

A blocking rule for a side (fndds or off) is a simple boolean predicate over that \
side's text fields, expressed as:
  - "keywords": a list of lowercase substrings; a record is IN the block if its \
    description/category text contains ANY of them.
  - "exclude_keywords": lowercase substrings that, if present, take a record OUT of \
    the block even if a keyword matched (use sparingly, only for clear false-positive \
    patterns you observe in the samples).

You will be shown: the block name, a few dozen sample records from each side (some \
in-block, some plausibly-confusable out-of-block), and, on revision rounds, the pair \
completeness / reduction ratio achieved by the current rule against a calibration \
sample plus block sizes. Propose keyword lists that are inclusive enough to keep pair \
completeness high while excluding obviously unrelated products (protect reduction \
ratio). Reply with ONLY a JSON object of this shape:

{
  "fndds": {"keywords": [...], "exclude_keywords": [...]},
  "off": {"keywords": [...], "exclude_keywords": [...]},
  "rationale": "one or two sentences"
}
"""


def build_blocking_prompt(
    block_name: str,
    fndds_samples: list[str],
    off_samples: list[str],
    previous_rule: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> tuple[str, str]:
    payload = {
        "block_name": block_name,
        "fndds_sample_descriptions": fndds_samples[:40],
        "off_sample_product_names": off_samples[:40],
    }
    if previous_rule is not None:
        payload["previous_rule"] = previous_rule
    if metrics is not None:
        payload["previous_round_metrics"] = metrics
        payload["instruction"] = (
            "Revise the previous rule to improve pair completeness and/or reduction "
            "ratio based on these metrics, or keep it unchanged if it looks optimal."
        )
    else:
        payload["instruction"] = f"Propose an initial blocking rule for the '{block_name}' block."
    return _BLOCKING_SYSTEM, json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Matching attributes
# ---------------------------------------------------------------------------

_ATTRIBUTE_SYSTEM = """\
You are a subject-matter-expert-in-the-loop assistant proposing MATCHING ATTRIBUTES for \
probabilistic record linkage (Fellegi-Sunter / splink) between USDA FNDDS records and \
Open Food Facts (OFF) records within a single product block.

A matching attribute is a derived boolean or categorical field, computed independently \
on each side from its own text fields, that should agree for true matches and disagree \
for non-matches (e.g. is_greek, fat_level). For each attribute, specify how to compute \
it on each side as a keyword-based rule (the runtime uses simple case-insensitive \
substring matching against the block's text fields — you are proposing the keyword \
lists and (for categorical attributes) the category values, not writing code).

Attributes MUST be conceptually distinct from each other (avoid proposing two \
attributes that are near-synonyms — this violates the Fellegi-Sunter conditional \
independence assumption); prefer a small number of high-signal attributes over many \
redundant ones. On revision rounds you'll be shown a correlation-check report flagging \
attribute pairs that are too correlated, and/or linkage evaluation results — drop or \
redefine attributes accordingly.

Reply with ONLY a JSON object of this shape:

{
  "attributes": [
    {
      "name": "is_greek",
      "kind": "boolean",
      "description": "Is this a Greek-style yogurt?",
      "fndds_keywords": ["greek"],
      "off_keywords": ["greek", "grec"]
    },
    {
      "name": "fat_level",
      "kind": "categorical",
      "description": "Fat content tier",
      "categories": {
        "fat_free": {"fndds_keywords": ["fat free", "nonfat", "skim"], "off_keywords": ["fat free", "0% fat", "skimmed"]},
        "low_fat":  {"fndds_keywords": ["low fat", "lowfat"], "off_keywords": ["low fat", "1%", "2%"]},
        "full_fat": {"fndds_keywords": ["whole milk"], "off_keywords": ["whole milk", "full fat"]}
      }
    }
  ],
  "rationale": "one or two sentences"
}
"""


def build_attribute_prompt(
    block_name: str,
    sample_pairs: list[dict[str, Any]],
    existing_attributes: list[dict[str, Any]] | None = None,
    correlation_flags: list[dict[str, Any]] | None = None,
    evaluation: dict[str, Any] | None = None,
) -> tuple[str, str]:
    payload: dict[str, Any] = {
        "block_name": block_name,
        "sample_candidate_pairs": sample_pairs[:30],
    }
    if existing_attributes is not None:
        payload["existing_attributes"] = existing_attributes
    if correlation_flags:
        payload["correlation_flags"] = correlation_flags
    if evaluation is not None:
        payload["previous_round_evaluation"] = evaluation
    if existing_attributes is None:
        payload["instruction"] = (
            f"Propose an initial set of 4-8 matching attributes for the '{block_name}' block."
        )
    else:
        payload["instruction"] = (
            "Revise the attribute set: drop/merge correlated attributes, add attributes "
            "that would help distinguish the evaluation errors, or keep unchanged if it "
            "looks optimal."
        )
    return _ATTRIBUTE_SYSTEM, json.dumps(payload, indent=2, default=str)

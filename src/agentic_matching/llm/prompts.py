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

A blocking rule for a side (fndds or off) is a boolean predicate over that side's data:
  - "keywords": lowercase substrings; a record is IN the block if its raw \
    description/product-name text contains ANY of them. Tested against the record's \
    own name/description ONLY, never category or annotation text.
  - "categories": exact category values (see "category_options" below) -- a record is \
    IN the block if its category field equals (fndds) or contains (off) any of them. \
    Prefer this over keywords whenever a clearly on-topic category exists: FNDDS's \
    WWEIA food category and OFF's categories_tags are clean, human-curated labels, far \
    more precise than a keyword guess.
  - "exclude_keywords": lowercase substrings that, if present, take a record OUT of \
    the block even if a keyword/category matched (use sparingly, only for clear \
    false-positive patterns you observe in the samples).

Two keyword pitfalls to avoid, both verified to actually happen with this prompt:
  - Boilerplate false positives: FNDDS's "Additional Description" field is full of \
    variant-annotations ("all flavors", "multigrain, whole grain, whole wheat") shared \
    across many unrelated categories -- a keyword that looks narrow in a sample can \
    still match wildly unrelated records if it happens to also appear in that \
    boilerplate. "flavors", "whole", "fruit", "plain" are BAD standalone keywords for \
    exactly this reason (they recur in whole wheat muffins, fruit salad, plain \
    pretzels, ...), not because they're bad words in general. Prefer keywords \
    conceptually tied to the block itself (its own name and clear synonyms) over \
    generic descriptive/modifier words, even if a modifier appears frequently in the \
    samples you're shown.
  - Catalog-wide-common terms: avoid proposing any term from "catalog_wide_common_terms" \
    (below) as a standalone keyword unless it's genuinely central to this specific \
    block (e.g. the block's own name) -- a term matching a large fraction of the whole \
    catalog will let in many unrelated products no matter how narrow this block seems \
    from the samples alone. This is the single most common way a proposed rule ends up \
    far larger than intended.

You will be shown:
  - The block name and a few dozen sample records from each side (some in-block, some \
    plausibly-confusable out-of-block).
  - "category_options": the most common category values among records already \
    plausibly in this block, with counts -- pick from these, don't invent category names.
  - "corpus_stats": each side's total catalog size and its "catalog_wide_common_terms" \
    (see above).
  - On revision rounds, "previous_round_metrics": pair completeness / reduction ratio \
    achieved by the current rule against a calibration sample, plus block sizes.
  - "domain_notes" (may be absent): a short freeform note from a domain expert about \
    this specific block that doesn't fit the structured keyword/category schema (e.g. \
    a block covering several distinct sub-concepts that shouldn't be narrowed down to \
    just the ones dominating the samples) -- treat it as authoritative, weighted \
    alongside everything else you're shown.
  - "prior_linking_findings" (may be absent): a factual finding from a PREVIOUS full \
    run of this block all the way through attribute-generation and linking, specifically \
    flagging that the *blocking rule* -- not the attributes -- looked like the \
    bottleneck (e.g. too few candidate pairs survived for the model to train on, or a \
    collapsed match-probability prior). This means an earlier version of the rule you're \
    revising was already tried end-to-end and came up short for a structural reason -- \
    address it directly (e.g. broaden keywords/categories, reconsider whether an \
    exclude_keyword is too aggressive), don't just repeat the same shape of rule.

Propose keyword/category lists that are inclusive enough to keep pair completeness \
high while excluding obviously unrelated products (protect reduction ratio). Reply \
with ONLY a JSON object of this shape:

{
  "fndds": {"keywords": [...], "categories": [...], "exclude_keywords": [...]},
  "off": {"keywords": [...], "categories": [...], "exclude_keywords": [...]},
  "rationale": "one or two sentences"
}
"""


def build_blocking_prompt(
    block_name: str,
    fndds_samples: list[str],
    off_samples: list[str],
    previous_rule: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    corpus_stats: dict[str, Any] | None = None,
    category_options: dict[str, Any] | None = None,
    notes: str | None = None,
    linking_feedback: str | None = None,
) -> tuple[str, str]:
    payload = {
        "block_name": block_name,
        "fndds_sample_descriptions": fndds_samples[:40],
        "off_sample_product_names": off_samples[:40],
    }
    if notes:
        # Freeform domain guidance from blocking/seed_rules.py -- not a structured
        # keyword/category list, so it wouldn't fit previous_rule's shape; echoed every
        # round (unlike previous_rule, which only seeds round 0) since it's persistent
        # knowledge about the block, not rule state being iterated on.
        payload["domain_notes"] = notes
    if linking_feedback:
        # A finding from outer_loop.diagnose_blocking_problem, echoed every round of a
        # re-blocking attempt (see run_blocking_agent's `linking_feedback` parameter) --
        # unlike `notes`, this isn't hand-authored domain knowledge, it's an empirical
        # observation from a previous end-to-end run of this exact block.
        payload["prior_linking_findings"] = linking_feedback
    if corpus_stats is not None:
        payload["corpus_stats"] = corpus_stats
    if category_options is not None:
        payload["category_options"] = category_options
    if previous_rule is not None:
        payload["previous_rule"] = previous_rule
    if metrics is not None:
        payload["previous_round_metrics"] = metrics
        payload["instruction"] = (
            "Revise the previous rule to improve pair completeness and/or reduction "
            "ratio based on these metrics, or keep it unchanged if it looks optimal."
        )
    elif previous_rule is not None:
        # A seed rule (see blocking/seed_rules.py) with no metrics yet -- an SME- or
        # config-provided starting point (e.g. hand-picked exclude_keywords) for round
        # 0, not yet evaluated. Frame it as a starting point to refine, not something
        # to discard in favor of proposing from scratch.
        payload["instruction"] = (
            "A domain expert provided the above as a starting point for this block "
            "(e.g. specific exclude_keywords already known to be needed) -- refine it "
            "using the samples below (add/adjust keywords or categories, keep the "
            "given exclude_keywords unless you have a specific reason to change them), "
            "rather than proposing an unrelated rule from scratch."
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
for non-matches (e.g., for a yogurt block, "is this Greek-style?" or "what's the fat \
tier?"; for a different block, entirely different attributes would make sense — always \
tie every attribute you propose to the CURRENT block's own actual products, never to \
another block's). For each attribute, specify how to compute it on each side as a \
keyword-based rule (the runtime uses simple case-insensitive substring matching \
against the block's text fields — you are proposing the keyword lists and (for \
categorical attributes) the category values, not writing code).

Attributes MUST be conceptually distinct from each other (avoid proposing two \
attributes that are near-synonyms — this violates the Fellegi-Sunter conditional \
independence assumption); prefer a small number of high-signal attributes over many \
redundant ones. This also means: if a real-world property has several mutually \
exclusive values (e.g. a product's bean type is kidney OR pinto OR black, never \
several at once), model it as ONE categorical attribute with those values as \
categories, not as several independent booleans (is_kidney, is_pinto, is_black, ...) \
— independent booleans for mutually-exclusive facts violate the same conditional \
independence assumption just as badly as near-synonym attributes do.

You will be shown:
  - "sample_candidate_pairs": a few dozen (FNDDS description, OFF product name) pairs \
    from this block.
  - "field_stats" (may be absent): the most common real values of each side's \
    categorical fields (e.g. OFF's categories_tags, FNDDS's WWEIA category, Branded's \
    category) within this block's population -- ground any categorical attribute's \
    category values in these observed values rather than inventing category names \
    that may not occur in the actual data.
  - "candidate_terms" (may be absent): tokens mined directly from this block's own \
    free text that split its population into a meaningful minority/majority on at \
    least one side (not near-0% or near-100%, which wouldn't discriminate anything), \
    with the fraction of each side's records containing them. These are a floor, not \
    a ceiling — consider proposing a simple boolean attribute for a salient one (e.g. \
    a term like "meat" or "rice" suggests has_meat/with_rice), but you are not limited \
    to only these; your own domain knowledge may suggest attributes, synonyms, or \
    translations (e.g. recognizing "pork" and "porc" as the same concept) that pure \
    frequency counting can't.
  - "domain_guidance" (may be absent): a short freeform note from a domain expert \
    about this specific block that doesn't fit the structured attribute schema (e.g. \
    that matches must agree on which specific sub-concept is involved, not just some \
    shared general property) -- treat it as authoritative and let it directly shape \
    which attributes you propose, alongside everything else you're shown.
  - On revision rounds, "existing_attributes" (the current set), "correlation_flags" \
    (a Cramer's V report flagging attribute pairs that are too correlated -- drop or \
    redefine them), and "previous_round_evaluation" (linkage results, including \
    "attribute_discriminative_power": each existing attribute's agreement rate on \
    known-true pairs vs. random non-match pairs from a calibration sample -- an \
    attribute agreeing at nearly the same rate on both, e.g. true=0.90/decoy=0.88, \
    isn't discriminating matches from non-matches at all and is a strong drop/redefine \
    candidate regardless of how it correlates with other attributes; a big gap, e.g. \
    true=0.95/decoy=0.12, means it's doing real work and should be kept).

Reply with ONLY a JSON object matching this SHAPE (this example's names/keywords are \
placeholders illustrating the schema for an unrelated example block, "widgets" -- do \
NOT reuse "example_is_x_widget"/"example_size_tier"/or their keywords for the actual \
block you're proposing attributes for; every real attribute you output must be built \
from the block name, samples, field_stats, and candidate_terms you were actually shown \
above, not copied from this template):

{
  "attributes": [
    {
      "name": "example_is_x_widget",
      "kind": "boolean",
      "description": "(placeholder -- replace with a real boolean attribute for the actual block above)",
      "fndds_keywords": ["<a term from THIS block's own samples>"],
      "off_keywords": ["<a term from THIS block's own samples>"]
    },
    {
      "name": "example_size_tier",
      "kind": "categorical",
      "description": "(placeholder -- replace with a real categorical attribute for the actual block above)",
      "categories": {
        "example_small": {"fndds_keywords": ["<...>"], "off_keywords": ["<...>"]},
        "example_large":  {"fndds_keywords": ["<...>"], "off_keywords": ["<...>"]}
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
    field_stats: dict[str, Any] | None = None,
    candidate_terms: list[dict[str, Any]] | None = None,
    guidance: str | None = None,
) -> tuple[str, str]:
    payload: dict[str, Any] = {
        "block_name": block_name,
        "sample_candidate_pairs": sample_pairs[:30],
    }
    if guidance:
        # Freeform domain guidance from attributes/seed_rules.py's SEED_ATTRIBUTE_NOTES --
        # echoed every round (not just round 0), same rationale as build_blocking_
        # prompt's "notes" parameter: persistent context about the block, not state the
        # LLM's revision is expected to iterate away.
        payload["domain_guidance"] = guidance
    if field_stats is not None:
        payload["field_stats"] = field_stats
    if candidate_terms:
        payload["candidate_terms"] = candidate_terms
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
            "Revise the attribute set: drop/merge correlated attributes, drop/redefine "
            "non-discriminating attributes (see attribute_discriminative_power), add "
            "attributes that would help distinguish the evaluation errors, or keep "
            "unchanged if it looks optimal."
        )
    return _ATTRIBUTE_SYSTEM, json.dumps(payload, indent=2, default=str)

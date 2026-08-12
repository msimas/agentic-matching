"""Decomposed attribute revision: three narrower LLM calls replacing the single
"revise the whole set at once" call (llm/prompts.py's build_attribute_prompt, used
only for the "propose from scratch" case now) --

  1. decide_keep_drop -- for each EXISTING attribute, keep / drop / redefine, and why.
  2. identify_gap -- given concrete false-positive/false-negative examples, is a
     genuinely NEW attribute needed, and what concept should it capture.
  3. define_attributes -- given exactly what (1) and (2) flagged, produce the actual
     keyword/category definitions -- nothing else, and "keep" attributes are carried
     forward programmatically, never re-emitted by the LLM.

Motivation, grounded in this project's own real agent-loop runs, not a generic best
practice: a single big revision call reliably got SOME of several simultaneous asks
right and silently dropped others -- adopting only 6 of 25 offered blocking
categories, noticing a degenerate attribute's bad discriminative-power signal but
never actually dropping it across 6 real rounds. That's a scope/attention problem, not
a knowledge gap (the model already had everything it needed in each of those cases) --
so the fix is narrowing what's asked per call, not giving the model more autonomy
(tool use) or more context (which this project already does extensively, e.g.
holdout_error_examples, and which alone didn't fix the keep/drop pattern above).

Stages 1 and 2 support a bounded "need more info before I can decide" round-trip (see
llm/prompts.py's _NEED_MORE_INFO_NOTE and info_requests.py) -- deliberately not stage
3, which is closer to pure extraction (given exactly what to define, write the
keywords) than judgment, and not given tool use / open-ended requests, for the same
reason the comparison set stayed a fixed, backend-agnostic mechanism elsewhere in this
project: cheap, bounded, auditable, and reproducible across every LLM backend this
project supports (mock/Ollama/Databricks), unlike function-calling reliability, which
varies a lot by model/backend.

Shared by attributes/agent_loop.py (standalone loop -- correlation_flags only, no
trained-model evaluation) and linking/agent_loop.py (richer evaluation +
discriminative_power + error_examples, since it trains a real model every round) --
called with whatever signals each caller actually has; a caller with weaker signals
just gets an emptier keep_drop/gap-identification payload, not a different code path.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentic_matching.attributes.info_requests import fulfill_requests
from agentic_matching.attributes.rules import filter_valid_attributes
from agentic_matching.llm.prompts import build_definition_prompt, build_gap_identification_prompt, build_keep_drop_prompt

log = logging.getLogger(__name__)


def ask_with_followup(client, system: str, user: str, temperature: float, fndds_texts: list[Any], off_texts: list[Any]) -> dict[str, Any]:
    """Calls `client.complete_json`; if the response is a "need_more_info" request
    (see llm/prompts.py's _NEED_MORE_INFO_NOTE), fulfills it from data already in
    memory (info_requests.fulfill_requests -- no new query capability, no extra LLM
    call to gather it) and asks exactly once more with the answers appended. Never
    loops beyond that one follow-up -- bounded cost, not an open-ended negotiation."""
    response = client.complete_json(system, user, temperature=temperature)
    if isinstance(response, dict) and response.get("status") == "need_more_info":
        requested = response.get("requested") or []
        answers = fulfill_requests(requested, fndds_texts, off_texts)
        log.info("LLM requested more info (%d item(s)) before deciding; re-asking with it included: %s", len(requested), requested)
        followup_user = (
            user
            + "\n\nYou requested more information before deciding; here it is:\n"
            + json.dumps(answers, indent=2, default=str)
            + "\n\nNow answer the original question, in the original required JSON shape "
            "(not another need_more_info request)."
        )
        response = client.complete_json(system, followup_user, temperature=temperature)
    return response


def decide_keep_drop(
    client,
    block_name: str,
    existing_attributes: list[dict[str, Any]],
    correlation_flags: list[dict[str, Any]] | None,
    evaluation: dict[str, Any] | None,
    guidance: str | None,
    temperature: float,
    fndds_texts: list[Any],
    off_texts: list[Any],
) -> dict[str, str]:
    """Returns {attribute_name: "keep"|"drop"|"redefine"}. Any attribute the response
    doesn't mention (shouldn't happen -- the prompt asks for every one by name)
    defaults to "keep", the conservative choice -- never silently dropped by omission."""
    if not existing_attributes:
        return {}
    sys_p, user_p = build_keep_drop_prompt(block_name, existing_attributes, correlation_flags, evaluation, guidance)
    response = ask_with_followup(client, sys_p, user_p, temperature, fndds_texts, off_texts)
    decisions: dict[str, str] = {}
    for d in response.get("decisions", []) or []:
        name, action = d.get("name"), d.get("action")
        if name and action in ("keep", "drop", "redefine"):
            decisions[name] = action
    for attr in existing_attributes:
        decisions.setdefault(attr["name"], "keep")
    return decisions


def identify_gap(
    client,
    block_name: str,
    error_examples: dict[str, list[dict[str, Any]]] | None,
    existing_attributes: list[dict[str, Any]],
    guidance: str | None,
    temperature: float,
    fndds_texts: list[Any],
    off_texts: list[Any],
    concept_history: dict[str, list[dict[str, Any]]] | None = None,
) -> str | None:
    """Returns a short concept phrase for a new attribute if one's warranted, else
    None. Skipped entirely (no LLM call) when there are no concrete error examples to
    reason from -- the standalone attribute loop (attributes/agent_loop.py) never has
    these, since it never trains a real model; only linking/agent_loop.py's revision
    calls do.

    `concept_history` (linking/agent_loop.py only -- see its docstring there) is every
    concept identified so far THIS RUN, whatever attribute(s) got defined for it, and
    why each was auto-dropped once trained -- passed straight through to the prompt so
    the model can recognize "I already tried this and it failed" itself instead of
    that recognition depending on string-matching a guidance note (verified real case:
    the standalone auto_dropped-name guidance note didn't stop "preparation method"
    from being re-identified 3 rounds running under 3 different attribute names, each
    auto-dropped in turn -- the concept-level history was real and available, just
    never actually shown to this specific call)."""
    if not error_examples or not (error_examples.get("false_positives") or error_examples.get("false_negatives")):
        return None
    sys_p, user_p = build_gap_identification_prompt(block_name, error_examples, existing_attributes, guidance, concept_history)
    response = ask_with_followup(client, sys_p, user_p, temperature, fndds_texts, off_texts)
    if response.get("needed") and response.get("concept"):
        return str(response["concept"])
    return None


def define_attributes(
    client,
    block_name: str,
    existing_attributes: list[dict[str, Any]],
    decisions: dict[str, str],
    gap_concept: str | None,
    sample_pairs: list[dict[str, Any]],
    field_stats: dict[str, Any] | None,
    candidate_terms: list[dict[str, Any]] | None,
    guidance: str | None,
    temperature: float,
    concept_history: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Applies "keep" decisions programmatically (carried forward unchanged, never
    re-emitted by the LLM -- an unchanged attribute has no reason to risk the LLM
    corrupting it, e.g. the verified real case earlier this session where a revision
    dropped an attribute's keywords while only meaning to update its description).
    Asks the LLM only for "redefine"/new-from-gap definitions -- nothing else.

    Guards against a verified real collision (beans, LLM_DEVICE=databricks): the gap
    concept's "to_define" entry has no name (that's the whole point -- it's asking the
    LLM to invent one), and the definition call used to have no visibility into which
    names were already taken. The LLM's "new" attribute for a "specific bean type" gap
    came back named identically to the kept `beans_contains_meat` (near-duplicate
    keywords, nothing to do with bean type), and nothing caught it -- both survived
    into the merged list, and splink silently double-trained that one comparison.
    Telling the LLM the taken names (build_definition_prompt's existing_names) fixes
    the common case; the post-hoc filter below is the backstop for when it ignores
    that anyway -- never trust a single layer for something this cheap to double-check."""
    by_name = {a["name"]: a for a in existing_attributes}
    kept = [by_name[name] for name, action in decisions.items() if action == "keep" and name in by_name]
    redefine_names = [name for name, action in decisions.items() if action == "redefine" and name in by_name]
    to_define: list[dict[str, Any]] = [
        {"name": name, "reason": "flagged for redefinition -- see keep/drop decision"} for name in redefine_names
    ]
    if gap_concept:
        gap_entry = {"reason": f"new attribute needed: {gap_concept}"}
        prior = (concept_history or {}).get(gap_concept.strip().lower())
        if prior:
            # This exact concept was already tried and failed before this run's
            # definition step gets a chance to repeat the same keyword shape blind --
            # see revise_attributes' docstring for the verified case this fixes.
            gap_entry["prior_attempts_for_this_concept"] = prior
        to_define.append(gap_entry)
    if not to_define:
        return kept
    existing_names = [a["name"] for a in existing_attributes]
    sys_p, user_p = build_definition_prompt(
        block_name, sample_pairs, field_stats, candidate_terms, guidance, to_define, existing_names
    )
    response = client.complete_json(sys_p, user_p, temperature=temperature)
    new_defs = filter_valid_attributes(response.get("attributes", []) or [])

    kept_names = {a["name"] for a in kept}
    allowed_new_names = set(redefine_names)  # a redefinition is allowed to keep its name; a gap attribute is not
    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for a in new_defs:
        name = a["name"]
        if name in kept_names and name not in allowed_new_names:
            log.warning(
                "block '%s': LLM's new attribute for gap concept %r collided with kept attribute name %r "
                "(despite being told it was taken) -- dropping the collision instead of silently duplicating it.",
                block_name,
                gap_concept,
                name,
            )
            continue
        if name in seen_names:
            log.warning("block '%s': LLM returned duplicate definitions for %r; keeping the first.", block_name, name)
            continue
        seen_names.add(name)
        deduped.append(a)
    return kept + deduped


def revise_attributes(
    client,
    block_name: str,
    existing_attributes: list[dict[str, Any]],
    *,
    correlation_flags: list[dict[str, Any]] | None = None,
    evaluation: dict[str, Any] | None = None,
    error_examples: dict[str, list[dict[str, Any]]] | None = None,
    sample_pairs: list[dict[str, Any]],
    field_stats: dict[str, Any] | None,
    candidate_terms: list[dict[str, Any]] | None,
    guidance: str | None,
    fndds_texts: list[Any],
    off_texts: list[Any],
    temperature: float,
    concept_history: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Top-level orchestration of the three stages. Returns (attrs, rationale,
    gap_concept) -- the decomposed flow doesn't produce one "rationale" string the way
    the old single-call prompt did, so one is synthesized here from the stage
    decisions, for the same artifact-logging purpose (see attributes/agent_loop.py's
    AttributeRound.rationale). gap_concept is also returned on its own (not just
    embedded in the rationale string) so a caller that trains a real model across
    rounds (linking/agent_loop.py) can track it across rounds to detect thrashing --
    the same concept being proposed, auto-dropped for no discriminating power, and
    proposed again under a new attribute name -- without resorting to parsing it back
    out of the rationale text.

    `concept_history` (see identify_gap's docstring) is forwarded to both identify_gap
    (so it can recognize a concept it's about to re-propose already failed) and
    define_attributes (so a redefinition attempt sees exactly what keyword shape
    already failed for that concept, not just that it should "try something else")."""
    decisions = decide_keep_drop(
        client, block_name, existing_attributes, correlation_flags, evaluation, guidance, temperature, fndds_texts, off_texts
    )
    gap_concept = identify_gap(
        client, block_name, error_examples, existing_attributes, guidance, temperature, fndds_texts, off_texts, concept_history
    )
    attrs = define_attributes(
        client,
        block_name,
        existing_attributes,
        decisions,
        gap_concept,
        sample_pairs,
        field_stats,
        candidate_terms,
        guidance,
        temperature,
        concept_history,
    )
    dropped = [n for n, a in decisions.items() if a == "drop"]
    redefined = [n for n, a in decisions.items() if a == "redefine"]
    rationale_parts = []
    if dropped:
        rationale_parts.append(f"dropped {dropped}")
    if redefined:
        rationale_parts.append(f"redefined {redefined}")
    if gap_concept:
        rationale_parts.append(f"added new attribute for: {gap_concept}")
    rationale = "; ".join(rationale_parts) or "kept the attribute set unchanged"
    log.info("block '%s': decomposed revision -- %s", block_name, rationale)
    return attrs, rationale, gap_concept

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
) -> str | None:
    """Returns a short concept phrase for a new attribute if one's warranted, else
    None. Skipped entirely (no LLM call) when there are no concrete error examples to
    reason from -- the standalone attribute loop (attributes/agent_loop.py) never has
    these, since it never trains a real model; only linking/agent_loop.py's revision
    calls do."""
    if not error_examples or not (error_examples.get("false_positives") or error_examples.get("false_negatives")):
        return None
    sys_p, user_p = build_gap_identification_prompt(block_name, error_examples, existing_attributes, guidance)
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
) -> list[dict[str, Any]]:
    """Applies "keep" decisions programmatically (carried forward unchanged, never
    re-emitted by the LLM -- an unchanged attribute has no reason to risk the LLM
    corrupting it, e.g. the verified real case earlier this session where a revision
    dropped an attribute's keywords while only meaning to update its description).
    Asks the LLM only for "redefine"/new-from-gap definitions -- nothing else."""
    by_name = {a["name"]: a for a in existing_attributes}
    kept = [by_name[name] for name, action in decisions.items() if action == "keep" and name in by_name]
    to_define: list[dict[str, Any]] = [
        {"name": name, "reason": "flagged for redefinition -- see keep/drop decision"}
        for name, action in decisions.items()
        if action == "redefine" and name in by_name
    ]
    if gap_concept:
        to_define.append({"reason": f"new attribute needed: {gap_concept}"})
    if not to_define:
        return kept
    sys_p, user_p = build_definition_prompt(block_name, sample_pairs, field_stats, candidate_terms, guidance, to_define)
    response = client.complete_json(sys_p, user_p, temperature=temperature)
    new_defs = filter_valid_attributes(response.get("attributes", []) or [])
    return kept + new_defs


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
) -> tuple[list[dict[str, Any]], str]:
    """Top-level orchestration of the three stages. Returns (attrs, rationale) -- the
    decomposed flow doesn't produce one "rationale" string the way the old single-call
    prompt did, so one is synthesized here from the stage decisions, for the same
    artifact-logging purpose (see attributes/agent_loop.py's AttributeRound.rationale)."""
    decisions = decide_keep_drop(
        client, block_name, existing_attributes, correlation_flags, evaluation, guidance, temperature, fndds_texts, off_texts
    )
    gap_concept = identify_gap(
        client, block_name, error_examples, existing_attributes, guidance, temperature, fndds_texts, off_texts
    )
    attrs = define_attributes(
        client, block_name, existing_attributes, decisions, gap_concept, sample_pairs, field_stats, candidate_terms, guidance, temperature
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
    return attrs, rationale

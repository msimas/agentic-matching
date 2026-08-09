"""Bounded (max_outer_rounds, default 2) outer loop that closes the feedback gap the
three inner agent loops leave open: blocking/agent_loop.py never sees anything from
attributes or linking, and linking/agent_loop.py only ever revises the ATTRIBUTE set,
never the blocking rule that determined which records were even candidates in the
first place. A weak result after linking can be caused by either -- but only attribute
revision was previously automated; a genuinely too-narrow (or wrongly-excluding)
blocking rule required a human to notice and manually re-run blocking with a hint (as
happened for this project's own "breaded_vegetables" and "yogurt" blocks, more than
once, before this module existed).

Each outer round runs the full blocking -> attributes -> linking pipeline once, then
`diagnose_blocking_problem` inspects the LAST linking round for a small, deliberately
narrow set of symptoms that specifically implicate blocking (not attributes -- those are
already the inner linking loop's job to fix). If one fires, the finding is fed back into
another blocking round via `run_blocking_agent`'s `linking_feedback` parameter and the
whole pipeline runs again; if not, the outer loop stops -- success or otherwise, this is
NOT a search for a "perfect" block, just a check for a structural blocking problem that
attribute revision can't resolve on its own.

Bounded low (2, not the inner loops' 3) because each outer round is a full cycle of
real-LLM calls across all three stages -- this can add up to a long wall-clock time
depending on the LLM backend in use (see e.g. the yogurt/breaded_vegetables re-runs
earlier in this project's history) -- so this is "give re-blocking one chance," not an
open-ended search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from agentic_matching.attributes.agent_loop import run_attribute_agent
from agentic_matching.blocking.agent_loop import run_blocking_agent
from agentic_matching.config import ARTIFACTS_DIR, agent_loop_settings
from agentic_matching.linking.agent_loop import LinkingRound, run_linking_agent
from agentic_matching.llm.client import ChatClient, get_llm_client

log = logging.getLogger(__name__)

# Every real linking round this project has produced, even for its thinnest/most
# degenerate block ("breaded_vegetables" at its narrowest), has cleared several hundred
# raw candidate pairs -- this floor is set well below that, as a "clearly too few to
# plausibly contain a working model" line, not a tuned optimum. See LinkingRound's
# n_candidate_pairs docstring for why this checks the raw count, not plausibility's
# confident-match count.
MIN_CANDIDATE_PAIRS = 100


@dataclass
class OuterRound:
    round: int
    trigger: str | None  # None means no blocking-shaped problem was found (stop here)
    linking_rounds_completed: int
    final_n_candidate_pairs: int
    final_holdout_f1: float | None


def diagnose_blocking_problem(rounds: list[LinkingRound]) -> str | None:
    """Inspect a completed linking loop's rounds for symptoms that specifically point
    at the BLOCKING rule, not the attribute set -- attribute-shaped weaknesses (poor
    holdout f1 with plenty of candidate pairs and a non-degenerate model, low
    attribute_discriminative_power on individual attributes) are exactly what the inner
    linking loop's own attribute-revision step already exists to fix, so they're
    deliberately NOT reasons to re-block here; only checks the LAST round, since that's
    the one attribute revision had every opportunity to already fix.

    Returns a concise, factual finding string (fed to build_blocking_prompt as
    `linking_feedback`) if triggered, else None.
    """
    if not rounds:
        return None
    last = rounds[-1]
    reasons = []
    if last.n_candidate_pairs < MIN_CANDIDATE_PAIRS:
        reasons.append(
            f"only {last.n_candidate_pairs} raw candidate pairs were generated even "
            f"after {len(rounds)} round(s) of attribute revision, well below a usable "
            f"threshold ({MIN_CANDIDATE_PAIRS})"
        )
    collapsed = [f for f in last.degeneracy_flags if f.get("kind") == "collapsed"]
    if collapsed:
        cols = ", ".join(f.get("column", "?") for f in collapsed)
        reasons.append(
            f"the trained model still shows a 'collapsed' degeneracy flag (column(s): "
            f"{cols}) after attribute revision, meaning EM had too little variation in "
            f"the candidate pairs to estimate real parameters from"
        )
    if not reasons:
        return None
    return (
        f"A previous end-to-end run of this block (through {len(rounds)} round(s) of "
        f"attribute revision) still showed: {'; '.join(reasons)}. This looks like a "
        "blocking problem, not an attribute problem -- attribute revision already had "
        "its chance to fix it and couldn't. Reconsider the rule itself: it may be too "
        "narrow (missing keyword/category variants that would surface more candidates), "
        "or an exclude_keyword may be too aggressive."
    )


def run_outer_loop(block_name: str, client: ChatClient | None = None) -> list[OuterRound]:
    client = client or get_llm_client()
    rounds: list[OuterRound] = []
    linking_feedback: str | None = None

    for round_idx in range(agent_loop_settings.max_outer_rounds):
        log.info("=== outer round %d for block '%s': blocking ===", round_idx, block_name)
        run_blocking_agent(block_name, client=client, linking_feedback=linking_feedback)

        log.info("=== outer round %d for block '%s': attributes ===", round_idx, block_name)
        run_attribute_agent(block_name, client=client)

        log.info("=== outer round %d for block '%s': linking ===", round_idx, block_name)
        linking_rounds = run_linking_agent(block_name, client=client)

        trigger = diagnose_blocking_problem(linking_rounds)
        last = linking_rounds[-1] if linking_rounds else None
        outer_round = OuterRound(
            round=round_idx,
            trigger=trigger,
            linking_rounds_completed=len(linking_rounds),
            final_n_candidate_pairs=last.n_candidate_pairs if last else 0,
            final_holdout_f1=last.holdout_evaluation.get("f1") if last else None,
        )
        rounds.append(outer_round)
        _write_artifact(block_name, outer_round)

        if trigger is None:
            log.info(
                "Outer loop for '%s' found no blocking-shaped problem after outer "
                "round %d; stopping.",
                block_name,
                round_idx,
            )
            break
        if round_idx == agent_loop_settings.max_outer_rounds - 1:
            log.info(
                "Outer loop for '%s' reached max_outer_rounds (%d) with a blocking "
                "problem still flagged; stopping anyway -- see the last outer_loop "
                "artifact for the finding.",
                block_name,
                agent_loop_settings.max_outer_rounds,
            )
            break
        log.warning("Outer round %d flagged a blocking problem for '%s': %s", round_idx, block_name, trigger)
        linking_feedback = trigger

    return rounds


def _write_artifact(block_name: str, r: OuterRound) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"outer_loop_{block_name}_round{r.round}.json"
    path.write_text(json.dumps(asdict(r), indent=2))
    log.info("Wrote %s", path)

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
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from agentic_matching.attributes.agent_loop import run_attribute_agent
from agentic_matching.blocking.agent_loop import run_blocking_agent
from agentic_matching.config import ARTIFACTS_DIR, agent_loop_settings, new_run_id, latest_run_dir, run_artifacts_dir
from agentic_matching.linking.agent_loop import LinkingRound, run_linking_agent, select_best_round
from agentic_matching.linking.degeneracy_check import degenerate_attribute_columns
from agentic_matching.llm.client import ChatClient, get_llm_client

log = logging.getLogger(__name__)

# Every stage `run_outer_loop`'s `steps` parameter can select -- in pipeline order,
# which is also the order they run in within a round regardless of the order `steps`
# lists them.
ALL_STEPS = ("blocking", "attributes", "linking")

# The artifact filename glob(s) each stage is expected to have left behind in a run
# directory -- used by _copy_forward_skipped_stage (below) both to find what to copy
# forward from the most recent previous run when a stage is skipped this run, and to
# recognize when that previous run doesn't actually have them either (also skipped, or
# never completed) so that's a clear, loud error instead of a silently incomplete
# folder.
STAGE_ARTIFACT_GLOBS: dict[str, tuple[str, ...]] = {
    "blocking": ("blocking_round*.json",),
    "attributes": ("attributes_round*.json",),
    "linking": ("linking_round*.json", "matches_round*.csv", "final_matches.csv"),
}


def _copy_forward_skipped_stage(block_name: str, run_dir: Path, stage: str) -> None:
    """Copy `stage`'s artifacts from the most recent PREVIOUS run into `run_dir`, so a
    run that skips a stage (see run_outer_loop's `steps`) still leaves a complete,
    self-contained folder behind -- one that looks like every other run's, not one
    silently missing the files a skipped stage would normally produce.

    Raises FileNotFoundError -- not a silent no-op -- if there's no previous run for
    this block at all, or the most recent previous run doesn't actually have this
    stage's expected artifacts either (e.g. it also skipped that stage, or failed
    before completing it). An incomplete copy would be worse than an incomplete
    folder: it would look complete without being one, whereas this fails loudly at the
    moment the gap would first cause a problem, not later against best_round_so_far,
    diagnose_blocking_problem, or an SME who assumes something a run's folder claims
    to have but doesn't."""
    previous = latest_run_dir(block_name, before=run_dir.name)
    if previous is None:
        raise FileNotFoundError(
            f"--steps skips '{stage}' for block '{block_name}', but no previous run "
            f"exists under {ARTIFACTS_DIR / block_name} to copy its artifacts forward "
            f"from. Run with '{stage}' included at least once before skipping it."
        )
    # Resolve every pattern BEFORE creating run_dir or copying anything -- a partial
    # failure partway through must leave no trace (no empty/half-populated run_dir
    # behind), not just fail loudly. Verified real case while building this: an early
    # version created run_dir eagerly, then raised on the first missing pattern,
    # leaving a stray empty directory on disk that a subsequent latest_run_dir lookup
    # could itself mistake for a "previous run."
    matches_by_pattern: dict[str, list[Path]] = {}
    for pattern in STAGE_ARTIFACT_GLOBS[stage]:
        matches = list(previous.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"--steps skips '{stage}' for block '{block_name}', but the most recent "
                f"previous run ({previous.name}) has no '{pattern}' artifact(s) to copy "
                f"forward -- it likely also skipped '{stage}' or never completed it. "
                f"Run with '{stage}' included at least once before skipping it."
            )
        matches_by_pattern[pattern] = matches

    run_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for matches in matches_by_pattern.values():
        for f in matches:
            shutil.copy2(f, run_dir / f.name)
            copied += 1
    log.info(
        "block '%s': '%s' skipped this run -- copied %d artifact(s) forward from "
        "previous run %s into %s.",
        block_name,
        stage,
        copied,
        previous.name,
        run_dir.name,
    )

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
    deliberately NOT reasons to re-block here; only checks the BEST round (see
    linking.agent_loop.select_best_round), not just the chronologically last one --
    that's what actually ends up as the deliverable, so it's what the diagnosis (and
    outer_loop's own reported final_n_candidate_pairs/final_holdout_f1) should reflect,
    not whatever a later, possibly-regressed round happened to leave behind.

    Returns a concise, factual finding string (fed to build_blocking_prompt as
    `linking_feedback`) if triggered, else None.
    """
    if not rounds:
        return None
    best = select_best_round(rounds)
    reasons = []
    if best.n_candidate_pairs < MIN_CANDIDATE_PAIRS:
        reasons.append(
            f"only {best.n_candidate_pairs} raw candidate pairs were generated even "
            f"after {len(rounds)} round(s) of attribute revision, well below a usable "
            f"threshold ({MIN_CANDIDATE_PAIRS})"
        )
    # Only a collapsed NON-attribute comparison implicates blocking -- a collapsed
    # *attribute* comparison means that one derived attribute carries no discriminating
    # signal in this block, which is an attribute-shaped problem, not a blocking one
    # (see degenerate_attribute_columns's docstring). The inner linking loop now
    # proactively drops those itself before a round is even recorded (see
    # run_linking_agent's retrain-on-degeneracy loop), so this should rarely fire on an
    # attribute name in practice -- excluded here regardless, as a correctness
    # guarantee rather than something that merely happens to hold today. Used to
    # reliably mean `description` specifically (its own fuzzy text comparison showed
    # this in effectively every trained round across every block tested, and
    # re-blocking never actually fixed it) -- `description` is no longer a comparison
    # at all (see splink_model.build_comparisons' docstring), so this condition is now
    # expected to rarely if ever fire; kept as a generic guarantee, not deleted,
    # in case a future comparison develops the same failure mode.
    # kinds={"collapsed"} only, matching this function's own narrower scope (see its
    # docstring) -- label_switching/untrained attribute flags are the inner loop's
    # concern (it checks all three), not a reason to trigger re-blocking here.
    attr_names = {a["name"] for a in best.attributes}
    collapsed_attr_cols = set(
        degenerate_attribute_columns(best.degeneracy_flags, attr_names, kinds=frozenset({"collapsed"}))
    )
    collapsed = [
        f
        for f in best.degeneracy_flags
        if f.get("kind") == "collapsed" and f.get("column") not in collapsed_attr_cols
    ]
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


def run_outer_loop(
    block_name: str,
    client: ChatClient | None = None,
    steps: Sequence[str] = ALL_STEPS,
    run_dir: Path | None = None,
) -> list[OuterRound]:
    """Run the pipeline stages named in `steps` (any non-empty subset of
    "blocking"/"attributes"/"linking", pipeline order regardless of how `steps` orders
    them) for `block_name`, once per outer round.

    Skipping a stage means exactly that -- it doesn't run at all, so e.g. skipping
    "attributes" leaves whatever's already persisted in attributes/generated/<block>/
    latest.json untouched and used as-is by "linking" if that's included. This is the
    tool for "just redo attribute selection against the existing block" (steps=
    ("attributes", "linking")) or "just redo linking against the existing attributes"
    (steps=("linking",)) without paying for a stage you don't want re-run.

    Every stage this run actually includes writes its artifacts into ONE shared
    `run_dir` (data/artifacts/<block_name>/<run_id>/, see config.run_artifacts_dir) --
    a fresh one per call by default, so each `run_outer_loop` invocation leaves behind
    its own timestamped, comparable-over-time record instead of overwriting the
    previous run's files. A SKIPPED stage's artifacts are instead COPIED FORWARD into
    this same `run_dir` from the most recent previous run (see
    _copy_forward_skipped_stage) -- so a folder produced by steps=("linking",) still
    looks like a complete run, with blocking/attributes artifacts included, just
    carried over rather than freshly produced. If no previous run has those artifacts
    to copy (first run ever for this block, or the previous run also skipped that
    stage), this raises FileNotFoundError rather than silently leaving `run_dir`
    incomplete -- a run "reusing" attributes/blocking results that don't actually
    exist anywhere is a real bug to surface immediately, not a gap to paper over.

    The re-blocking feedback loop (see this module's docstring) only makes sense when
    "blocking" is actually a step that can run again in response to a diagnosed
    problem -- if "blocking" isn't in `steps`, or "linking" isn't (so there's nothing
    to diagnose from), this runs a single outer round and stops regardless of what
    diagnose_blocking_problem would have said, logging a warning if it *would* have
    flagged something so that's not silently lost.
    """
    client = client or get_llm_client()
    steps_set = set(steps)
    invalid = steps_set - set(ALL_STEPS)
    if invalid:
        raise ValueError(f"Unknown step(s) {sorted(invalid)!r}; valid steps are {ALL_STEPS}")
    if not steps_set:
        raise ValueError(f"steps must be a non-empty subset of {ALL_STEPS}")
    can_loop = "blocking" in steps_set and "linking" in steps_set

    run_dir = run_dir or run_artifacts_dir(block_name, new_run_id())
    for stage in ALL_STEPS:
        if stage not in steps_set:
            _copy_forward_skipped_stage(block_name, run_dir, stage)

    rounds: list[OuterRound] = []
    linking_feedback: str | None = None

    for round_idx in range(agent_loop_settings.max_outer_rounds):
        if "blocking" in steps_set:
            log.info("=== outer round %d for block '%s': blocking ===", round_idx, block_name)
            run_blocking_agent(block_name, client=client, linking_feedback=linking_feedback, run_dir=run_dir)
        else:
            log.info("=== outer round %d for block '%s': skipping blocking (not in steps) ===", round_idx, block_name)

        if "attributes" in steps_set:
            log.info("=== outer round %d for block '%s': attributes ===", round_idx, block_name)
            run_attribute_agent(block_name, client=client, run_dir=run_dir)
        else:
            log.info("=== outer round %d for block '%s': skipping attributes (not in steps) ===", round_idx, block_name)

        if "linking" in steps_set:
            log.info("=== outer round %d for block '%s': linking ===", round_idx, block_name)
            linking_rounds = run_linking_agent(block_name, client=client, run_dir=run_dir)
        else:
            log.info("=== outer round %d for block '%s': skipping linking (not in steps) ===", round_idx, block_name)
            linking_rounds = []

        trigger = diagnose_blocking_problem(linking_rounds)
        # select_best_round, not linking_rounds[-1] -- see its docstring; keeps what
        # this reports consistent with what run_linking_agent actually left as the
        # deliverable (final_matches_<block>.csv), not a possibly-regressed last round.
        best = select_best_round(linking_rounds) if linking_rounds else None
        outer_round = OuterRound(
            round=round_idx,
            trigger=trigger,
            linking_rounds_completed=len(linking_rounds),
            final_n_candidate_pairs=best.n_candidate_pairs if best else 0,
            final_holdout_f1=best.holdout_evaluation.get("f1") if best else None,
        )
        rounds.append(outer_round)
        _write_artifact(run_dir, outer_round)

        if trigger is None:
            log.info(
                "Outer loop for '%s' found no blocking-shaped problem after outer "
                "round %d; stopping.",
                block_name,
                round_idx,
            )
            break
        if not can_loop:
            log.warning(
                "Outer round %d flagged a blocking problem for '%s' (%s), but "
                "re-blocking isn't possible this run ('blocking' and/or 'linking' not "
                "in steps=%s) -- stopping after one round instead of looping with "
                "nothing new to try.",
                round_idx,
                block_name,
                trigger,
                sorted(steps_set),
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

    # Grand total across every stage this run actually included -- the same client
    # instance (see `client = client or get_llm_client()` above) is reused for
    # blocking/attributes/linking, so this accumulator already reflects the whole
    # outer-loop run's real API cost, not just one stage's.
    client.log_usage_summary(label=f"{block_name} FULL OUTER LOOP, grand total")
    return rounds


def _write_artifact(run_dir: Path, r: OuterRound) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"outer_loop_round{r.round}.json"
    path.write_text(json.dumps(asdict(r), indent=2))
    log.info("Wrote %s", path)

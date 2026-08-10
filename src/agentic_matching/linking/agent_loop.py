"""Bounded (N=3 by default) agentic loop for the linking stage: train -> score ->
LLM revises the attribute set -> retrain, stopping early once the (proxy) F1 against
the calibration holdout stabilizes or no further revision is proposed. Each round's
predictions summary, degeneracy flags, and holdout evaluation are logged to
data/artifacts/linking_<block>_round<N>.json for SME review, and the full set of
predicted pairs (not just the JSON's top/bottom-N examples) is written to
data/artifacts/matches_<block>_round<N>.csv for direct inspection.

data/artifacts/final_matches_<block>.csv (no round number -- overwritten each round, so
it always reflects the latest run, same pattern as attributes/generated/<block>/
latest.json) is the actual deliverable: one best FNDDS record per OFF/commercial-product
record (see evaluate.best_match_per_off), not every candidate pair.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from agentic_matching.attributes.agent_loop import (
    _candidate_boolean_terms,
    _field_stats,
    _load_block,
    _pooled_values,
    _sample_pairs,
    load_latest_attributes,
)
from agentic_matching.attributes.metrics import check_correlations
from agentic_matching.attributes.rules import filter_valid_attributes
from agentic_matching.attributes.seed_rules import get_seed_attribute_notes
from agentic_matching.config import ARTIFACTS_DIR, agent_loop_settings
from agentic_matching.linking import splink_model
from agentic_matching.linking.degeneracy_check import check_degeneracy, export_trained_settings
from agentic_matching.linking.evaluate import (
    attribute_discriminative_power,
    best_match_per_off,
    export_predictions_csv,
    holdout_error_examples,
    plausibility_report,
    score_against_holdout,
)
from agentic_matching.llm.client import ChatClient, get_llm_client
from agentic_matching.llm.prompts import build_attribute_prompt

log = logging.getLogger(__name__)


@dataclass
class LinkingRound:
    round: int
    attributes: list[dict[str, Any]]
    degeneracy_flags: list[dict[str, Any]]
    holdout_evaluation: dict[str, Any]
    attribute_discriminative_power: list[dict[str, Any]]
    holdout_error_examples: dict[str, list[dict[str, Any]]]
    plausibility: dict[str, Any]
    # Raw candidate-pair count at threshold=0.0, *before* the 0.5 "confident match"
    # filter `plausibility` is computed from -- outer_loop.diagnose_blocking_problem
    # uses this specifically (not plausibility["n_pairs"]) because a low confident-match
    # count can be an attribute-quality problem the inner revision loop already handles,
    # while a low *raw* candidate count means the blocking rule itself left the model
    # with too little to work with regardless of attributes.
    n_candidate_pairs: int
    n_final_matches: int
    rationale: str
    matches_csv: str
    final_matches_csv: str


def _stabilized(prev_f1: float | None, cur_f1: float | None) -> bool:
    if prev_f1 is None or cur_f1 is None:
        return False
    return abs(prev_f1 - cur_f1) < agent_loop_settings.stabilization_delta


def run_linking_agent(block_name: str, client: ChatClient | None = None) -> list[LinkingRound]:
    client = client or get_llm_client()
    attrs = load_latest_attributes(block_name)

    # Grounding for this loop's attribute-revision calls (below) -- the block doesn't
    # change across rounds, so this is computed once, from the raw materialized block
    # (data/blocks/<block>_{fndds,off}.parquet), not from splink_model.build_linker's
    # prepared frames: those carry only unique_id/description/search_text/attribute
    # columns, not the raw category/brand fields _field_stats needs. Same functions,
    # same block-scoped data, the standalone attribute agent loop already uses -- see
    # attributes/agent_loop.py::run_attribute_agent for the identical pattern.
    raw_fndds_df, raw_off_df = _load_block(block_name)
    sample_pairs = _sample_pairs(raw_fndds_df, raw_off_df)
    field_stats = _field_stats(raw_fndds_df, raw_off_df)
    candidate_terms = _candidate_boolean_terms(raw_fndds_df, raw_off_df, block_name)

    rounds: list[LinkingRound] = []
    prev_f1: float | None = None

    for round_idx in range(agent_loop_settings.max_rounds):
        log.info("=== linking round %d for block '%s': training ===", round_idx, block_name)
        # build_linker may drop attributes that turned out unobservable on one side
        # (see splink_model._drop_unobservable_attrs) -- reassign `attrs` to what it
        # actually returns so every downstream use this round (train, holdout scoring,
        # CSV export, the artifact, next round's LLM prompt) stays consistent with what
        # the trained linker's settings actually contain.
        linker, fndds_df, off_df, attrs = splink_model.build_linker(block_name, attrs)
        splink_model.train(linker, attrs)

        trained_settings = export_trained_settings(linker)
        degeneracy_flags = check_degeneracy(trained_settings)

        predictions = splink_model.predict(linker, threshold=0.5)
        plausibility = plausibility_report(predictions)
        holdout_eval = score_against_holdout(block_name, attrs, trained_settings)
        # Pure value-agreement comparison against the same calibration holdout, not a
        # model prediction -- cheap enough to compute every round regardless of block
        # size (see its docstring for why this is a different, complementary signal to
        # both holdout_eval's aggregate f1 and correlation_flags below).
        discriminative_power = attribute_discriminative_power(block_name, attrs)
        # Concrete wrong pairs, not just aggregate numbers -- see its docstring for why
        # this is what actually lets the revision LLM (below) reason about what NEW
        # attribute would fix a real mistake, not just which existing ones are weak.
        error_examples = holdout_error_examples(block_name, attrs, trained_settings)

        # Exported separately from `predictions` (not gated on the 0.5 "confident
        # match" threshold above): the CSV is a review artifact, not a decision, so it
        # should always show the best available candidates -- confidence is conveyed by
        # the match_probability column itself, not a hard cutoff. Otherwise a weak
        # block/attribute combination can produce an empty, unhelpful file even when
        # real (if unconfident) candidates exist -- verified case: yogurt's
        # threshold=0.5 predictions were empty despite 304K real candidate pairs
        # topping out at match_probability 0.21. export_predictions_csv's own
        # `top_n` cap (default 5000) keeps the file a manageable size regardless.
        all_predictions = splink_model.predict(linker, threshold=0.0)
        matches_csv_path = ARTIFACTS_DIR / f"matches_{block_name}_round{round_idx}.csv"
        n_written = export_predictions_csv(all_predictions, attrs, matches_csv_path)
        log.info(
            "Wrote %s (top %d of %d candidate pairs by match_probability)",
            matches_csv_path,
            n_written,
            len(all_predictions),
        )

        # The actual deliverable (see this module's docstring): one best FNDDS record
        # per OFF/commercial-product record, not every candidate. No round number --
        # always reflects this (latest) round, so a downstream consumer always reads
        # the current best output without needing to know which round number "won".
        final_matches = best_match_per_off(all_predictions)
        final_matches_csv_path = ARTIFACTS_DIR / f"final_matches_{block_name}.csv"
        n_final_written = export_predictions_csv(final_matches, attrs, final_matches_csv_path, top_n=None)
        log.info(
            "Wrote %s (%d OFF records, each with its single best FNDDS match)",
            final_matches_csv_path,
            n_final_written,
        )

        rationale = f"round {round_idx} trained with {len(attrs)} attributes"
        rounds.append(
            LinkingRound(
                round=round_idx,
                attributes=attrs,
                degeneracy_flags=degeneracy_flags,
                holdout_evaluation=holdout_eval,
                attribute_discriminative_power=discriminative_power,
                holdout_error_examples=error_examples,
                plausibility=plausibility,
                n_candidate_pairs=len(all_predictions),
                n_final_matches=n_final_written,
                rationale=rationale,
                matches_csv=str(matches_csv_path),
                final_matches_csv=str(final_matches_csv_path),
            )
        )
        _write_artifact(block_name, rounds[-1])

        cur_f1 = holdout_eval.get("f1")
        log.info(
            "block=%s round=%d degeneracy_flags=%d holdout_precision=%s holdout_recall=%s holdout_f1=%s",
            block_name,
            round_idx,
            len(degeneracy_flags),
            holdout_eval.get("precision"),
            holdout_eval.get("recall"),
            cur_f1,
        )

        if round_idx == agent_loop_settings.max_rounds - 1:
            break
        if not degeneracy_flags and _stabilized(prev_f1, cur_f1):
            log.info("Linking loop for '%s' stabilized after round %d", block_name, round_idx)
            break

        # Ask the LLM to revise the attribute set given this round's evaluation +
        # a fresh correlation check, same as the standalone attribute agent loop --
        # including the same block-grounded sample_pairs/field_stats/candidate_terms
        # (computed once, above), not just correlation/evaluation numbers in isolation.
        values_df = _pooled_values(attrs, fndds_df.rename(columns={"search_text": "fndds_search_text"}), off_df)
        correlation_flags = check_correlations(values_df)
        sys_p, user_p = build_attribute_prompt(
            block_name,
            sample_pairs=sample_pairs,
            existing_attributes=attrs,
            correlation_flags=correlation_flags or None,
            evaluation={
                **holdout_eval,
                "degeneracy_flags": degeneracy_flags,
                "attribute_discriminative_power": discriminative_power,
                **error_examples,
            },
            field_stats=field_stats,
            candidate_terms=candidate_terms,
            guidance=get_seed_attribute_notes(block_name),
        )
        log.info(
            "block '%s' round %d: asking the LLM to revise matching attributes based "
            "on this round's linking results -- this is the slow step, everything "
            "else in this loop (splink training, scoring) finishes in seconds.",
            block_name,
            round_idx,
        )
        try:
            response = client.complete_json(sys_p, user_p)
        except Exception:
            # See blocking/agent_loop.py's identical handling for why: this round's own
            # training/evaluation already succeeded and is already appended to `rounds`
            # -- only the *next* round's revision is lost, so just stop here rather
            # than losing everything to an uncaught exception.
            log.exception(
                "Attribute-revision call failed after round %d for block '%s'; "
                "stopping here with the rounds completed so far.",
                round_idx,
                block_name,
            )
            break
        new_attrs = filter_valid_attributes(response["attributes"])
        if {a["name"] for a in new_attrs} == {a["name"] for a in attrs}:
            log.info("LLM proposed no attribute changes for '%s'; stopping.", block_name)
            attrs = new_attrs
            break
        attrs = new_attrs
        prev_f1 = cur_f1

    return rounds


def _write_artifact(block_name: str, r: LinkingRound) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"linking_{block_name}_round{r.round}.json"
    payload = asdict(r)
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Wrote %s", path)

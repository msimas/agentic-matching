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
from dataclasses import asdict, dataclass, field
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
from agentic_matching.attributes.revision import revise_attributes
from agentic_matching.attributes.seed_rules import get_seed_attribute_notes
from agentic_matching.config import ARTIFACTS_DIR, agent_loop_settings, round_temperature
from agentic_matching.linking import splink_model
from agentic_matching.linking.degeneracy_check import (
    check_degeneracy,
    degenerate_attribute_columns,
    export_trained_settings,
)
from agentic_matching.linking.evaluate import (
    attribute_discriminative_power,
    best_match_per_off,
    export_predictions_csv,
    holdout_error_examples,
    plausibility_report,
    score_against_holdout,
)
from agentic_matching.linking.nutrition_priors import apply_nutrition_priors, dampen_description_weight
from agentic_matching.llm.client import ChatClient, get_llm_client

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
    # Attribute names proactively dropped *before* this round's training because their
    # own comparison was degenerate (collapsed/label_switching/untrained) in a prior
    # same-round attempt -- see degenerate_attribute_columns and the retrain loop in
    # run_linking_agent. Empty for a round that never needed this. Kept on the record
    # (not just logged) so it's visible in the artifact JSON without needing to grep logs.
    dropped_attributes: list[str] = field(default_factory=list)


def _stabilized(prev_f1: float | None, cur_f1: float | None) -> bool:
    if prev_f1 is None or cur_f1 is None:
        return False
    return abs(prev_f1 - cur_f1) < agent_loop_settings.stabilization_delta


def select_best_round(rounds: list[LinkingRound]) -> LinkingRound:
    """Pick which round's results are the actual final deliverable -- NOT just the
    last round, since a later revision can regress. Verified real case (yogurt,
    LLM_DEVICE=ollama, qwen3:8b): round 3 reached holdout f1=0.048 with 55,516 real
    confident matches; round 4's revision collapsed real-world matching to ZERO
    confident matches (out of 296K candidates) while its holdout f1 (0.024) merely
    looked "back to round 0's baseline," not obviously catastrophic -- the
    calibration-holdout f1 metric alone does not reliably catch this class of
    regression, only `plausibility` (the actual target-domain prediction count) does.
    Public (not `_`-prefixed) since outer_loop.py needs the exact same selection, not
    a re-implementation of it, to keep what it reports consistent with what
    run_linking_agent actually leaves on disk as the deliverable.

    Scored as (not collapsed, no degeneracy flags, holdout f1) in that priority order:
    a round with zero confident real-world matches is disqualified outright regardless
    of how good its calibration-proxy f1 looks; among rounds that aren't collapsed,
    prefer no degeneracy flags, then the highest f1.
    """

    def _score(r: LinkingRound) -> tuple[bool, bool, float]:
        collapsed = r.plausibility.get("n_pairs", 0) == 0
        f1 = r.holdout_evaluation.get("f1") or 0.0
        return (not collapsed, not r.degeneracy_flags, f1)

    return max(rounds, key=_score)


def _train_and_evaluate_round(
    block_name: str, round_idx: int, attrs: list[dict[str, Any]]
) -> tuple[LinkingRound, list[dict[str, Any]], Any, Any]:
    """Train + evaluate one round with the given attribute set, writing this round's
    matches CSV / final_matches CSV as a side effect. Returns the round record, the
    attribute set actually used (build_linker may drop attributes that turned out
    unobservable on one side -- see splink_model._drop_unobservable_attrs), and the
    prepared fndds/off frames (needed by the caller's correlation check on the way to
    building the next round's revision prompt).

    Called more than once for the same `round_idx` when run_linking_agent's retrain
    loop (see its docstring) proactively drops a degenerate attribute -- each call
    fully overwrites that round's matches/final_matches CSVs with the latest retrain,
    so what's on disk always reflects the attribute set that was actually kept, never
    a stale attempt that included an attribute since dropped.
    """
    log.info("=== linking round %d for block '%s': training ===", round_idx, block_name)
    # build_linker may drop attributes that turned out unobservable on one side --
    # reassign `attrs` to what it actually returns so every downstream use this round
    # (train, holdout scoring, CSV export, the artifact, next round's LLM prompt) stays
    # consistent with what the trained linker's settings actually contain.
    linker, fndds_df, off_df, attrs = splink_model.build_linker(block_name, attrs)
    # Concise, always-INFO summary of what this round is actually training with --
    # distinct from the full prompt/response dump at DEBUG (see llm/client.py) and from
    # _write_artifact's full-fidelity JSON below: this is the one line to scan when
    # skimming a run's log to see how the attribute set evolved round over round
    # without opening every artifact file. Reflects what build_linker actually kept,
    # not just what was proposed, so it's never misleading about what got trained.
    log.info(
        "block '%s' round %d selected linking attributes: %s",
        block_name,
        round_idx,
        [f"{a['name']} ({a['kind']})" for a in attrs],
    )
    splink_model.train(linker, attrs)

    raw_trained_settings = export_trained_settings(linker)
    # Pure value-agreement comparison against the same calibration holdout, not a
    # model prediction -- cheap enough to compute every round regardless of block
    # size (see its docstring for why this is a different, complementary signal to
    # both holdout_eval's aggregate f1 and correlation_flags below). Computed here,
    # before the nutrition-prior blend below, because apply_nutrition_priors needs
    # each attribute's n_true_pairs to weight how much its EM estimate vs. the domain
    # prior gets trusted -- see nutrition_priors.py's docstring.
    discriminative_power = attribute_discriminative_power(block_name, attrs)
    # A nutrition-significant attribute (meat/dairy/egg/sugar/molasses -- see
    # nutrition_priors.py) has its EM-trained m/u blended toward a domain prior
    # rather than trusted as-is; `description`'s own comparison is separately damped
    # toward "no information" (repeatedly unstable at this project's block scale --
    # see nutrition_priors.py's module docstring). Every other comparison passes
    # through unchanged. This produces a NEW settings dict each step -- rebuild the
    # linker from the final one so every downstream prediction/degeneracy check
    # reflects the actual, adjusted weights, not EM's untouched estimate.
    trained_settings = apply_nutrition_priors(raw_trained_settings, attrs, discriminative_power)
    trained_settings = dampen_description_weight(trained_settings)
    if trained_settings != raw_trained_settings:
        linker = splink_model.linker_from_settings(fndds_df, off_df, trained_settings)

    degeneracy_flags = check_degeneracy(trained_settings)

    predictions = splink_model.predict(linker, threshold=0.5)
    plausibility = plausibility_report(predictions)
    holdout_eval = score_against_holdout(block_name, attrs, trained_settings)
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
    round_obj = LinkingRound(
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
    return round_obj, attrs, fndds_df, off_df


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
    # Names dropped by the retrain-on-collapse loop below, across all rounds so far --
    # echoed to the revision LLM (via the evaluation payload) so it doesn't propose the
    # exact same non-discriminating attribute right back, and also used to filter its
    # response defensively in case it does anyway.
    auto_dropped: list[str] = []

    for round_idx in range(agent_loop_settings.max_rounds):
        round_result, attrs, fndds_df, off_df = _train_and_evaluate_round(block_name, round_idx, attrs)
        # Proactively drop any attribute whose OWN comparison is degenerate
        # (collapsed/label_switching/untrained -- see degenerate_attribute_columns's
        # docstring for why all three mean "no usable signal" for an attribute
        # comparison specifically) in this round's trained model, then retrain the
        # SAME round without it -- rather than only handing it to the LLM as feedback
        # and hoping a future revision notices. Real cases this fixes (beans,
        # LLM_DEVICE=ollama/databricks): `beans_is_bean_dip` stayed collapsed through 6
        # rounds of attribute revision, and `beans_preparation_method` showed
        # label_switching in EVERY round of the block's entire history (true pairs
        # agreed on it 0% of the time vs 0.2% for random decoy pairs -- see
        # attribute_discriminative_power) -- in both cases the LLM kept revising other
        # attributes instead of ever dropping the broken one, and the outer loop then
        # misdiagnosed the survivor as a blocking problem. Looped (not a single pass)
        # since dropping one attribute can occasionally unmask another; bounded
        # naturally because `attrs` strictly shrinks each pass.
        dropped_this_round: list[str] = []
        while attrs:
            degenerate = degenerate_attribute_columns(round_result.degeneracy_flags, {a["name"] for a in attrs})
            if not degenerate:
                break
            kinds_by_name = {
                f["column"]: f["kind"] for f in round_result.degeneracy_flags if f.get("column") in degenerate
            }
            log.warning(
                "block '%s' round %d: proactively dropping attribute(s) %s -- "
                "degeneracy flag(s) %s in this round's own trained model (see "
                "degenerate_attribute_columns's docstring for what each flag kind "
                "means for an attribute comparison); retraining round %d without them "
                "instead of waiting on attribute revision to notice.",
                block_name,
                round_idx,
                degenerate,
                kinds_by_name,
                round_idx,
            )
            dropped_this_round.extend(degenerate)
            attrs = [a for a in attrs if a["name"] not in degenerate]
            round_result, attrs, fndds_df, off_df = _train_and_evaluate_round(block_name, round_idx, attrs)
        if dropped_this_round:
            round_result.dropped_attributes = dropped_this_round
            auto_dropped.extend(dropped_this_round)

        rounds.append(round_result)
        _write_artifact(block_name, rounds[-1])

        degeneracy_flags = round_result.degeneracy_flags
        holdout_eval = round_result.holdout_evaluation
        discriminative_power = round_result.attribute_discriminative_power
        error_examples = round_result.holdout_error_examples
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
        # Decomposed into keep/drop -> gap-identification -> definition (see
        # attributes/revision.py) rather than one big "revise everything" call --
        # degeneracy_flags deliberately excluded from what's shown here: a degenerate
        # attribute-column flag is now auto-remediated above (proactive drop, before
        # this point) rather than something the LLM needs to review/act on itself.
        values_df = _pooled_values(attrs, fndds_df.rename(columns={"search_text": "fndds_search_text"}), off_df)
        correlation_flags = check_correlations(values_df)
        evaluation: dict[str, Any] = {**holdout_eval, "attribute_discriminative_power": discriminative_power}
        guidance = get_seed_attribute_notes(block_name)
        if auto_dropped:
            note = (
                "Do not propose these already-tried, proactively-removed attributes "
                "again with the same definition -- their comparison showed zero "
                f"discriminating power in a trained model: {auto_dropped}."
            )
            guidance = f"{guidance}\n\n{note}" if guidance else note
        # has_prior_state=True unconditionally -- unlike blocking/attributes' round 0,
        # this loop never lacks prior state: it always starts from an already-persisted
        # attribute set (see load_latest_attributes above), so every round here has
        # real evidence/an existing set to stay faithful to, never a from-scratch
        # "exploration" call.
        temperature = round_temperature(round_idx, has_prior_state=True)
        log.info(
            "Linking and revising matching attributes for block '%s' round %d: asking the LLM to revise matching attributes based "
            "on this round's linking results (temperature=%.2f).",
            block_name,
            round_idx,
            temperature,
        )
        try:
            new_attrs, revision_rationale = revise_attributes(
                client,
                block_name,
                attrs,
                correlation_flags=correlation_flags or None,
                evaluation=evaluation,
                error_examples=error_examples,
                sample_pairs=sample_pairs,
                field_stats=field_stats,
                candidate_terms=candidate_terms,
                guidance=guidance,
                fndds_texts=raw_fndds_df["description"].tolist(),
                off_texts=raw_off_df["search_text"].tolist(),
                temperature=temperature,
            )
            log.info("block '%s' round %d: %s", block_name, round_idx, revision_rationale)
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
        if auto_dropped:
            # Defensive backstop against the LLM re-proposing an attribute already
            # proven non-discriminating this run, despite the evaluation payload
            # telling it not to (see `auto_dropped_attributes` above) -- the retrain
            # loop next round would just drop it again anyway, so save that wasted
            # round here instead.
            reintroduced = [a["name"] for a in new_attrs if a["name"] in auto_dropped]
            if reintroduced:
                log.warning(
                    "block '%s' round %d: LLM re-proposed already-dropped attribute(s) "
                    "%s despite being told not to; filtering them out again.",
                    block_name,
                    round_idx,
                    reintroduced,
                )
                new_attrs = [a for a in new_attrs if a["name"] not in auto_dropped]
        if {a["name"] for a in new_attrs} == {a["name"] for a in attrs}:
            log.info("LLM proposed no attribute changes for '%s'; stopping.", block_name)
            attrs = new_attrs
            break
        attrs = new_attrs
        prev_f1 = cur_f1

    # If the round the loop actually ended on isn't the best one produced (see
    # select_best_round's docstring for the verified regression case), the "always
    # reflects the latest run" final_matches_<block>.csv would otherwise silently ship
    # a worse result than what this run already found. Retraining costs nothing
    # LLM-related (splink training/prediction on these block sizes is seconds, not
    # minutes) so this is cheap insurance, run regardless of how many rounds happened.
    best = select_best_round(rounds)
    if best.round != rounds[-1].round:
        log.warning(
            "block '%s': round %d regressed (plausibility n_pairs=%s, holdout f1=%s) "
            "relative to round %d (n_pairs=%s, f1=%s) -- restoring round %d's "
            "attributes as the final deliverable instead of leaving the regressed "
            "round's results in final_matches_%s.csv.",
            block_name,
            rounds[-1].round,
            rounds[-1].plausibility.get("n_pairs"),
            rounds[-1].holdout_evaluation.get("f1"),
            best.round,
            best.plausibility.get("n_pairs"),
            best.holdout_evaluation.get("f1"),
            best.round,
            block_name,
        )
        linker, _fndds_df, _off_df, _attrs = splink_model.build_linker(block_name, best.attributes)
        splink_model.train(linker, best.attributes)
        best_predictions = splink_model.predict(linker, threshold=0.0)
        final_matches = best_match_per_off(best_predictions)
        final_matches_csv_path = ARTIFACTS_DIR / f"final_matches_{block_name}.csv"
        n_final_written = export_predictions_csv(final_matches, best.attributes, final_matches_csv_path, top_n=None)
        log.info(
            "Restored %s from round %d (%d OFF records, each with its single best FNDDS match)",
            final_matches_csv_path,
            best.round,
            n_final_written,
        )

    return rounds


def _write_artifact(block_name: str, r: LinkingRound) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"linking_{block_name}_round{r.round}.json"
    payload = asdict(r)
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Wrote %s", path)

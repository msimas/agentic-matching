from agentic_matching.blocking.agent_loop import BlockingRound, _select_final_rule, select_best_blocking_round

# Real numbers from the case this was written to fix: LLM_DEVICE=ollama, yogurt block.
# Round 0 -> round 1 moved pair_completeness by +0.007 (well under the 0.01 default
# stabilization delta) while round 1's rule (added "plain" as an FNDDS keyword) pulled
# in 69 false positives the proxy metric has no way to see.
ROUND_0 = BlockingRound(
    round=0,
    rule={"fndds": {"keywords": ["yogurt", "Greek"]}},
    metrics={"pair_completeness": 0.372, "reduction_ratio": 0.9999},
    rationale="initial",
)
ROUND_1_NEGLIGIBLE = BlockingRound(
    round=1,
    rule={"fndds": {"keywords": ["yogurt", "Greek", "plain"]}},
    metrics={"pair_completeness": 0.379, "reduction_ratio": 0.9997},
    rationale="added plain",
)
ROUND_1_REAL_GAIN = BlockingRound(
    round=1,
    rule={"fndds": {"keywords": ["yogurt", "Greek", "yoghurt"]}},
    metrics={"pair_completeness": 0.6, "reduction_ratio": 0.999},
    rationale="added yoghurt spelling",
)


def test_negligible_change_keeps_earlier_rule():
    final = _select_final_rule([ROUND_0, ROUND_1_NEGLIGIBLE])
    assert final is ROUND_0.rule


def test_real_gain_keeps_latest_rule():
    final = _select_final_rule([ROUND_0, ROUND_1_REAL_GAIN])
    assert final is ROUND_1_REAL_GAIN.rule


def test_single_round_has_no_earlier_round_to_prefer():
    final = _select_final_rule([ROUND_0])
    assert final is ROUND_0.rule


def test_three_rounds_negligible_change_only_at_the_end():
    # round0 -> round1 is a real gain, round1 -> round2 is negligible: should keep
    # round1 (the real gain), not round0 or round2.
    round_2_negligible = BlockingRound(
        round=2,
        rule={"fndds": {"keywords": ["yogurt", "Greek", "yoghurt", "plain"]}},
        metrics={"pair_completeness": 0.605, "reduction_ratio": 0.9989},
        rationale="added plain",
    )
    final = _select_final_rule([ROUND_0, ROUND_1_REAL_GAIN, round_2_negligible])
    assert final is ROUND_1_REAL_GAIN.rule


def test_never_stabilized_keeps_last_round():
    # Every round shows a real gain (exhausted max_rounds without stabilizing) --
    # should keep the last (most-refined) round, matching prior behavior.
    round_2_real_gain = BlockingRound(
        round=2,
        rule={"fndds": {"keywords": ["yogurt", "Greek", "yoghurt", "yogourt"]}},
        metrics={"pair_completeness": 0.85, "reduction_ratio": 0.998},
        rationale="added yogourt spelling",
    )
    final = _select_final_rule([ROUND_0, ROUND_1_REAL_GAIN, round_2_real_gain])
    assert final is round_2_real_gain.rule


# -- select_best_blocking_round: the regression guard nothing previously provided ----


def test_never_stabilized_but_last_round_is_a_real_regression_does_not_win():
    # The exact gap this function was added to close: a big (non-negligible) change
    # that is ALSO a bad one. Old behavior would have shipped round_2 unconditionally
    # just because the loop exhausted max_rounds without ever stabilizing.
    round_2_regression = BlockingRound(
        round=2,
        rule={"fndds": {"keywords": ["yogurt", "plain"]}},  # "plain" catches everything
        metrics={"pair_completeness": 0.4, "reduction_ratio": 0.95},  # big, real drop
        rationale="broadened with a generic term",
    )
    final = _select_final_rule([ROUND_0, ROUND_1_REAL_GAIN, round_2_regression])
    assert final is ROUND_1_REAL_GAIN.rule  # not the regressed last round


def test_pair_completeness_below_floor_disqualifies_regardless_of_reduction_ratio():
    unusable = BlockingRound(
        round=0, rule={"fndds": {"keywords": ["x"]}}, metrics={"pair_completeness": 0.0, "reduction_ratio": 0.99999}, rationale=""
    )
    usable = BlockingRound(
        round=1, rule={"fndds": {"keywords": ["y"]}}, metrics={"pair_completeness": 0.2, "reduction_ratio": 0.95}, rationale=""
    )
    assert select_best_blocking_round([unusable, usable]) is usable


def test_reduction_ratio_below_floor_disqualifies_regardless_of_pair_completeness():
    unusable = BlockingRound(
        round=0, rule={"fndds": {"keywords": ["x"]}}, metrics={"pair_completeness": 0.9, "reduction_ratio": 0.5}, rationale=""
    )
    usable = BlockingRound(
        round=1, rule={"fndds": {"keywords": ["y"]}}, metrics={"pair_completeness": 0.3, "reduction_ratio": 0.995}, rationale=""
    )
    assert select_best_blocking_round([unusable, usable]) is usable


def test_balances_pair_completeness_and_reduction_ratio_not_either_alone():
    # High pair_completeness but middling reduction_ratio vs. the reverse -- the
    # harmonic mean should prefer whichever is more BALANCED, not just the one with
    # the single highest raw number on either axis.
    lopsided = BlockingRound(
        round=0, rule={"fndds": {"keywords": ["x"]}}, metrics={"pair_completeness": 0.99, "reduction_ratio": 0.99}, rationale=""
    )
    balanced = BlockingRound(
        round=1, rule={"fndds": {"keywords": ["y"]}}, metrics={"pair_completeness": 0.9, "reduction_ratio": 0.999}, rationale=""
    )
    # balanced: 2*0.9*0.999/(0.9+0.999) ≈ 0.9474 vs lopsided: 2*0.99*0.99/(0.99+0.99) = 0.99
    # (lopsided actually wins here since both its numbers are high -- this test documents
    # the harmonic-mean behavior concretely rather than asserting a vague notion of "balance").
    assert select_best_blocking_round([lopsided, balanced]) is lopsided


def test_single_round_is_returned_even_below_floors():
    # No alternative exists -- returning the only round beats crashing on an empty max().
    only = BlockingRound(
        round=0, rule={"fndds": {"keywords": ["x"]}}, metrics={"pair_completeness": 0.0, "reduction_ratio": 0.0}, rationale=""
    )
    assert select_best_blocking_round([only]) is only


# -- seed_round as a scored baseline candidate ---------------------------------------
#
# Real case this covers (breaded_vegetables, LLM_DEVICE=databricks): the seed rule
# scored n_fndds_block=16/n_off_block=89/pair_completeness=0.030, and the LLM's own
# round 0 came back WORSE on every axis (10/53/0.006, having dropped every
# exclude_keyword) -- round 0 was only ever compared against other LLM rounds before
# this, never against the seed it was supposedly refining, so nothing caught a
# regression right out of the gate.

SEED_STRONG = BlockingRound(
    round=-1,
    rule={"fndds": {"keywords": ["seed"]}},
    metrics={"pair_completeness": 0.030, "reduction_ratio": 0.9999},
    rationale="unmodified seed rule (baseline)",
)
ROUND_0_REGRESSION = BlockingRound(
    round=0,
    rule={"fndds": {"keywords": ["seed", "worse"]}},
    metrics={"pair_completeness": 0.006, "reduction_ratio": 0.9995},
    rationale="dropped exclude_keywords",
)


def test_seed_beats_a_regressed_round_0():
    final = _select_final_rule([ROUND_0_REGRESSION], SEED_STRONG)
    assert final is SEED_STRONG.rule


def test_genuinely_better_round_0_still_beats_the_seed():
    final = _select_final_rule([ROUND_1_REAL_GAIN], SEED_STRONG)
    assert final is ROUND_1_REAL_GAIN.rule


def test_no_seed_behaves_exactly_as_before():
    # seed_round defaults to None -- same result as calling with one argument.
    assert _select_final_rule([ROUND_0, ROUND_1_REAL_GAIN]) is _select_final_rule(
        [ROUND_0, ROUND_1_REAL_GAIN], None
    )


def test_seed_is_not_exempt_from_the_stabilization_check_on_the_last_real_round():
    # The seed is a candidate alongside the rounds, not a veto over the stabilization
    # rule: if the last real round stabilized against the one before it, it's still
    # dropped from consideration -- the seed only gets to compete with what's left.
    round_2_negligible = BlockingRound(
        round=2,
        rule={"fndds": {"keywords": ["yogurt", "Greek", "yoghurt", "plain"]}},
        metrics={"pair_completeness": 0.605, "reduction_ratio": 0.9989},
        rationale="added plain",
    )
    final = _select_final_rule([ROUND_0, ROUND_1_REAL_GAIN, round_2_negligible], SEED_STRONG)
    # round_2_negligible dropped for stabilizing against round 1; seed (0.030) loses to
    # round 1's real gain (0.6) on the merits.
    assert final is ROUND_1_REAL_GAIN.rule

from agentic_matching.blocking.agent_loop import BlockingRound, _select_final_rule

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

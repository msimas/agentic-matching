from agentic_matching.linking.agent_loop import _derived_evaluation_signals


def test_f1_pct_of_ceiling_computed_and_rounded():
    result = _derived_evaluation_signals(0.068, 0.5528, None)
    assert result["f1_pct_of_ceiling"] == 12  # 0.068 / 0.5528 * 100 = 12.3.. -> 12


def test_f1_pct_of_ceiling_absent_when_max_achievable_missing():
    result = _derived_evaluation_signals(0.068, None, None)
    assert "f1_pct_of_ceiling" not in result


def test_f1_pct_of_ceiling_absent_when_cur_f1_missing():
    result = _derived_evaluation_signals(None, 0.5528, None)
    assert "f1_pct_of_ceiling" not in result


def test_f1_pct_of_ceiling_absent_when_ceiling_is_zero():
    # A holdout sample with zero resolvable true OFF items -- division would be
    # meaningless, not just a crash to dodge.
    result = _derived_evaluation_signals(0.0, 0.0, None)
    assert "f1_pct_of_ceiling" not in result


def test_regressed_from_best_round_delta_is_negative_when_below_best():
    result = _derived_evaluation_signals(0.05, 0.5528, 0.065)
    assert result["regressed_from_best_round"] == {"f1_delta": -0.015}


def test_regressed_from_best_round_absent_when_no_best_given():
    result = _derived_evaluation_signals(0.05, 0.5528, None)
    assert "regressed_from_best_round" not in result


def test_both_signals_present_together():
    result = _derived_evaluation_signals(0.05, 0.5528, 0.065)
    assert set(result.keys()) == {"f1_pct_of_ceiling", "regressed_from_best_round"}


def test_empty_when_only_cur_f1_given():
    result = _derived_evaluation_signals(0.05, None, None)
    assert result == {}

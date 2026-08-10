from agentic_matching.linking.agent_loop import LinkingRound, select_best_round


def _round(round_idx, f1, n_pairs, degeneracy_flags=None):
    return LinkingRound(
        round=round_idx,
        attributes=[],
        degeneracy_flags=degeneracy_flags or [],
        holdout_evaluation={"f1": f1},
        attribute_discriminative_power=[],
        holdout_error_examples={"false_positives": [], "false_negatives": []},
        plausibility={"n_pairs": n_pairs},
        n_candidate_pairs=1000,
        n_final_matches=0,
        rationale="test",
        matches_csv="unused.csv",
        final_matches_csv="unused_final.csv",
    )


def test_single_round_is_selected():
    r = _round(0, f1=0.1, n_pairs=100)
    assert select_best_round([r]) is r


def test_prefers_higher_f1_when_neither_collapsed():
    r0 = _round(0, f1=0.1, n_pairs=100)
    r1 = _round(1, f1=0.2, n_pairs=100)
    assert select_best_round([r0, r1]).round == 1


def test_real_regression_case_yogurt_qwen3_8b():
    # Verified real case: round 3 reached f1=0.048 with 55,516 confident real matches;
    # round 4/5 regressed to f1=0.024 (merely "looks like round 0's baseline", not
    # obviously catastrophic on f1 alone) with ZERO confident matches -- a collapse
    # only visible via plausibility, not the holdout f1 metric. This is the case that
    # motivated select_best_round existing at all.
    rounds = [
        _round(0, f1=0.0237, n_pairs=0),
        _round(1, f1=0.0421, n_pairs=54634),
        _round(2, f1=0.0454, n_pairs=54887),
        _round(3, f1=0.0481, n_pairs=55516),
        _round(4, f1=0.0237, n_pairs=0),
        _round(5, f1=0.0237, n_pairs=0),
    ]
    best = select_best_round(rounds)
    assert best.round == 3


def test_collapsed_round_disqualified_even_with_perfect_f1():
    healthy = _round(0, f1=0.05, n_pairs=100)
    collapsed_but_higher_f1 = _round(1, f1=0.99, n_pairs=0)
    assert select_best_round([healthy, collapsed_but_higher_f1]).round == 0


def test_degeneracy_flags_break_ties_before_f1():
    clean = _round(0, f1=0.1, n_pairs=100, degeneracy_flags=[])
    flagged_higher_f1 = _round(1, f1=0.2, n_pairs=100, degeneracy_flags=[{"kind": "collapsed", "column": "x"}])
    assert select_best_round([clean, flagged_higher_f1]).round == 0


def test_all_rounds_collapsed_falls_back_to_best_f1_among_them():
    r0 = _round(0, f1=0.01, n_pairs=0)
    r1 = _round(1, f1=0.02, n_pairs=0)
    assert select_best_round([r0, r1]).round == 1


def test_missing_f1_treated_as_zero():
    r0 = _round(0, f1=None, n_pairs=100)
    r1 = _round(1, f1=0.01, n_pairs=100)
    assert select_best_round([r0, r1]).round == 1

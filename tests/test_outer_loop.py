from agentic_matching.linking.agent_loop import LinkingRound
from agentic_matching.outer_loop import MIN_CANDIDATE_PAIRS, diagnose_blocking_problem


def _round(n_candidate_pairs: int, degeneracy_flags=None, f1=0.5, round_idx=0) -> LinkingRound:
    return LinkingRound(
        round=round_idx,
        attributes=[],
        degeneracy_flags=degeneracy_flags or [],
        holdout_evaluation={"f1": f1},
        attribute_discriminative_power=[],
        plausibility={"n_pairs": 3},
        n_candidate_pairs=n_candidate_pairs,
        n_final_matches=0,
        rationale="test",
        matches_csv="unused.csv",
        final_matches_csv="unused_final.csv",
    )


def test_no_rounds_returns_none():
    assert diagnose_blocking_problem([]) is None


def test_healthy_round_returns_none():
    rounds = [_round(n_candidate_pairs=5000, degeneracy_flags=[], f1=0.4)]
    assert diagnose_blocking_problem(rounds) is None


def test_too_few_candidate_pairs_triggers():
    rounds = [_round(n_candidate_pairs=MIN_CANDIDATE_PAIRS - 1)]
    finding = diagnose_blocking_problem(rounds)
    assert finding is not None
    assert "candidate pairs" in finding


def test_exactly_at_floor_does_not_trigger():
    rounds = [_round(n_candidate_pairs=MIN_CANDIDATE_PAIRS)]
    assert diagnose_blocking_problem(rounds) is None


def test_collapsed_degeneracy_flag_triggers():
    rounds = [_round(n_candidate_pairs=5000, degeneracy_flags=[{"kind": "collapsed", "column": "description"}])]
    finding = diagnose_blocking_problem(rounds)
    assert finding is not None
    assert "collapsed" in finding
    assert "description" in finding


def test_non_collapsed_degeneracy_flag_does_not_trigger():
    # label_switching (verified elsewhere in this project to be common/benign on small
    # blocks) should NOT by itself imply a blocking problem -- only "collapsed" does.
    rounds = [_round(n_candidate_pairs=5000, degeneracy_flags=[{"kind": "label_switching", "column": "description"}])]
    assert diagnose_blocking_problem(rounds) is None


def test_only_last_round_is_inspected():
    # An early round's collapsed flag shouldn't matter if the final (most-revised)
    # round is healthy -- attribute revision already had its chance to fix it.
    rounds = [
        _round(n_candidate_pairs=10, degeneracy_flags=[{"kind": "collapsed", "column": "x"}], round_idx=0),
        _round(n_candidate_pairs=5000, degeneracy_flags=[], round_idx=1),
    ]
    assert diagnose_blocking_problem(rounds) is None


def test_both_reasons_combine_in_one_finding():
    rounds = [_round(n_candidate_pairs=5, degeneracy_flags=[{"kind": "collapsed", "column": "y"}])]
    finding = diagnose_blocking_problem(rounds)
    assert "candidate pairs" in finding
    assert "collapsed" in finding

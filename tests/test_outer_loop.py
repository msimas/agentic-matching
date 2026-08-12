from unittest.mock import Mock

import pytest

import agentic_matching.outer_loop as outer_loop
from agentic_matching.linking.agent_loop import LinkingRound
from agentic_matching.outer_loop import MIN_CANDIDATE_PAIRS, diagnose_blocking_problem, run_outer_loop


def _round(n_candidate_pairs: int, degeneracy_flags=None, f1=0.5, round_idx=0) -> LinkingRound:
    return LinkingRound(
        round=round_idx,
        attributes=[],
        degeneracy_flags=degeneracy_flags or [],
        holdout_evaluation={"f1": f1},
        attribute_discriminative_power=[],
        holdout_error_examples={"false_positives": [], "false_negatives": []},
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


# -- run_outer_loop's `steps` selection -----------------------------------------------


class _Recorder:
    """Fake for run_blocking_agent/run_attribute_agent/run_linking_agent -- records
    that it was called (and with what) instead of doing real work."""

    def __init__(self, linking_rounds=None):
        self.calls: list[dict] = []
        self._linking_rounds = linking_rounds if linking_rounds is not None else [_round(n_candidate_pairs=5000)]

    def __call__(self, block_name, **kwargs):
        self.calls.append({"block_name": block_name, **kwargs})
        return self._linking_rounds  # only meaningful for the linking fake


@pytest.fixture
def fakes(monkeypatch):
    blocking = _Recorder()
    attributes = _Recorder()
    linking = _Recorder()
    monkeypatch.setattr(outer_loop, "run_blocking_agent", blocking)
    monkeypatch.setattr(outer_loop, "run_attribute_agent", attributes)
    monkeypatch.setattr(outer_loop, "run_linking_agent", linking)
    return {"blocking": blocking, "attributes": attributes, "linking": linking}


def test_default_steps_calls_all_three(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(outer_loop, "ARTIFACTS_DIR", tmp_path)
    run_outer_loop("beans", client=Mock())
    assert len(fakes["blocking"].calls) == 1
    assert len(fakes["attributes"].calls) == 1
    assert len(fakes["linking"].calls) == 1


def test_skipping_blocking_does_not_call_it(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(outer_loop, "ARTIFACTS_DIR", tmp_path)
    run_outer_loop("beans", client=Mock(), steps=("attributes", "linking"))
    assert fakes["blocking"].calls == []
    assert len(fakes["attributes"].calls) == 1
    assert len(fakes["linking"].calls) == 1


def test_only_linking_step_skips_blocking_and_attributes(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(outer_loop, "ARTIFACTS_DIR", tmp_path)
    run_outer_loop("beans", client=Mock(), steps=("linking",))
    assert fakes["blocking"].calls == []
    assert fakes["attributes"].calls == []
    assert len(fakes["linking"].calls) == 1


def test_invalid_step_raises():
    with pytest.raises(ValueError):
        run_outer_loop("beans", client=Mock(), steps=("not_a_real_step",))


def test_empty_steps_raises():
    with pytest.raises(ValueError):
        run_outer_loop("beans", client=Mock(), steps=())


def test_cannot_loop_without_blocking_and_linking_stops_after_one_round(fakes, tmp_path, monkeypatch):
    # A triggering result (too few candidate pairs) would normally prompt a re-block --
    # but "blocking" isn't in steps, so there's nothing to loop for; should run exactly
    # once regardless of max_outer_rounds.
    fakes["linking"]._linking_rounds = [_round(n_candidate_pairs=1)]
    monkeypatch.setattr(outer_loop, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(outer_loop.agent_loop_settings, "max_outer_rounds", 5)
    rounds = run_outer_loop("beans", client=Mock(), steps=("attributes", "linking"))
    assert len(rounds) == 1
    assert rounds[0].trigger is not None
    assert len(fakes["linking"].calls) == 1

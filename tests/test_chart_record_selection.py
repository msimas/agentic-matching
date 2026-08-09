import pandas as pd
import pytest

from agentic_matching.linking.charts import select_records


def _predictions(probs: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"match_probability": probs, "id": range(len(probs))})


def test_empty_predictions_returns_empty_list():
    assert select_records(_predictions([]), n=5) == []


def test_top_mode_returns_highest_probabilities():
    df = _predictions([0.1, 0.9, 0.5, 0.7])
    records = select_records(df, n=2, mode="top")
    assert [r["match_probability"] for r in records] == [0.9, 0.7]


def test_bottom_mode_returns_lowest_probabilities():
    df = _predictions([0.1, 0.9, 0.5, 0.7])
    records = select_records(df, n=2, mode="bottom")
    assert set(r["match_probability"] for r in records) == {0.1, 0.5}


def test_borderline_mode_returns_closest_to_half():
    df = _predictions([0.01, 0.51, 0.99, 0.4])
    records = select_records(df, n=1, mode="borderline")
    assert records[0]["match_probability"] == 0.51


def test_stratified_mode_spans_the_score_range():
    df = _predictions([round(i / 100, 2) for i in range(101)])  # 0.00 .. 1.00
    records = select_records(df, n=5, mode="stratified")
    probs = sorted(r["match_probability"] for r in records)
    assert probs[0] < 0.1
    assert probs[-1] > 0.9
    assert len(probs) == 5


def test_stratified_mode_does_not_exceed_available_rows():
    df = _predictions([0.2, 0.8])
    records = select_records(df, n=10, mode="stratified")
    assert len(records) <= 2


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        select_records(_predictions([0.5]), n=1, mode="bogus")  # type: ignore[arg-type]


def test_n_zero_returns_empty():
    assert select_records(_predictions([0.5, 0.6]), n=0) == []

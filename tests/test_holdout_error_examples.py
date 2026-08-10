import pandas as pd

from agentic_matching.linking.evaluate import _holdout_error_examples

TRUE_LABELS = {"f1": "o1", "f2": "o2", "f3": "o3"}


def _preds(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_confident_wrong_match_is_a_false_positive():
    preds = _preds(
        [
            # f1's best-scoring candidate is o9, not the true o1 -- and it's confident.
            {"unique_id_l": "f1", "unique_id_r": "o9", "match_probability": 0.95, "description_l": "a", "description_r": "b"},
            {"unique_id_l": "f1", "unique_id_r": "o1", "match_probability": 0.3, "description_l": "a", "description_r": "c"},
        ]
    )
    result = _holdout_error_examples(preds, TRUE_LABELS, threshold=0.5, n=5)
    assert len(result["false_positives"]) == 1
    assert result["false_positives"][0]["unique_id_r"] == "o9"


def test_confident_correct_match_is_not_a_false_positive():
    preds = _preds([{"unique_id_l": "f1", "unique_id_r": "o1", "match_probability": 0.95, "description_l": "a", "description_r": "a"}])
    result = _holdout_error_examples(preds, TRUE_LABELS, threshold=0.5, n=5)
    assert result["false_positives"] == []


def test_true_pair_scored_below_threshold_is_a_false_negative():
    preds = _preds(
        [{"unique_id_l": "f1", "unique_id_r": "o1", "match_probability": 0.2, "description_l": "a", "description_r": "a"}]
    )
    result = _holdout_error_examples(preds, TRUE_LABELS, threshold=0.5, n=5)
    assert len(result["false_negatives"]) == 1
    assert result["false_negatives"][0]["unique_id_r"] == "o1"


def test_true_pair_scored_above_threshold_is_not_a_false_negative():
    preds = _preds(
        [{"unique_id_l": "f1", "unique_id_r": "o1", "match_probability": 0.7, "description_l": "a", "description_r": "a"}]
    )
    result = _holdout_error_examples(preds, TRUE_LABELS, threshold=0.5, n=5)
    assert result["false_negatives"] == []


def test_true_pair_never_a_candidate_is_silently_skipped():
    # No row at all for (f1, o1) -- a blocking-shaped gap, not reported here (see
    # docstring: outer_loop.diagnose_blocking_problem is where that belongs).
    preds = _preds([{"unique_id_l": "f2", "unique_id_r": "o2", "match_probability": 0.9, "description_l": "x", "description_r": "x"}])
    result = _holdout_error_examples(preds, TRUE_LABELS, threshold=0.5, n=5)
    assert result["false_negatives"] == []


def test_n_caps_both_lists():
    rows = []
    labels = {}
    for i in range(10):
        fid, oid = f"f{i}", f"o{i}"
        labels[fid] = oid
        rows.append({"unique_id_l": fid, "unique_id_r": f"wrong{i}", "match_probability": 0.9, "description_l": "x", "description_r": "y"})
    result = _holdout_error_examples(_preds(rows), labels, threshold=0.5, n=3)
    assert len(result["false_positives"]) == 3


def test_empty_preds_returns_empty_lists():
    result = _holdout_error_examples(pd.DataFrame(), TRUE_LABELS, threshold=0.5, n=5)
    assert result == {"false_positives": [], "false_negatives": []}


def test_empty_true_labels_returns_empty_lists():
    preds = _preds([{"unique_id_l": "f1", "unique_id_r": "o1", "match_probability": 0.9, "description_l": "a", "description_r": "a"}])
    result = _holdout_error_examples(preds, {}, threshold=0.5, n=5)
    assert result == {"false_positives": [], "false_negatives": []}

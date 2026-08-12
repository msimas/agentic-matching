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


# -- selection direction: best FNDDS match PER OFF RECORD, mirroring production's
# best_match_per_off (see score_against_holdout's docstring for why) --------------


def test_dedup_is_per_off_record_not_per_fndds_record():
    # Two different FNDDS candidates competing for the SAME off record -- only the
    # higher-scoring one should be considered (dedup on unique_id_r).
    preds = _preds(
        [
            {"unique_id_l": "f1", "unique_id_r": "o1", "match_probability": 0.95, "description_l": "a", "description_r": "x"},
            {"unique_id_l": "f9", "unique_id_r": "o1", "match_probability": 0.3, "description_l": "b", "description_r": "x"},
        ]
    )
    result = _holdout_error_examples(preds, {"f1": "o1"}, threshold=0.5, n=5)
    # f1 IS o1's true partner and won its own slot -- no false positive.
    assert result["false_positives"] == []


def test_landing_on_any_true_partner_is_not_a_false_positive():
    # off_id has multiple genuinely-true fdc_ids -- landing on the one NOT sampled as
    # "true_labels[fdc]" for this off_id must still not be flagged wrong.
    true_labels = {"f1": "o1", "f2": "o1"}  # both f1 and f2 are true partners of o1
    preds = _preds([{"unique_id_l": "f2", "unique_id_r": "o1", "match_probability": 0.9, "description_l": "a", "description_r": "b"}])
    result = _holdout_error_examples(preds, true_labels, threshold=0.5, n=5)
    assert result["false_positives"] == []


# -- same_text_as_true_partner tag: metric-artifact false positives (see llm/prompts.py's
# _GAP_SYSTEM, which is told to disregard these) -----------------------------------


def test_false_positive_with_identical_text_to_true_partner_is_tagged():
    # Verified real case: predicted "BLACK BEANS" for an off record whose true partner
    # (f_true) is ALSO "BLACK BEANS" -- wrong id, identical text, unfixable by any
    # attribute.
    true_labels = {"f_true": "o1"}
    preds = _preds(
        [
            {"unique_id_l": "f_wrong", "unique_id_r": "o1", "match_probability": 0.95, "description_l": "BLACK BEANS", "description_r": "x"},
            {"unique_id_l": "f_true", "unique_id_r": "o9", "match_probability": 0.1, "description_l": "black beans", "description_r": "y"},
        ]
    )
    result = _holdout_error_examples(preds, true_labels, threshold=0.5, n=5)
    assert len(result["false_positives"]) == 1
    assert result["false_positives"][0]["same_text_as_true_partner"] is True


def test_false_positive_with_genuinely_different_text_is_not_tagged():
    true_labels = {"f_true": "o1"}
    preds = _preds(
        [
            {"unique_id_l": "f_wrong", "unique_id_r": "o1", "match_probability": 0.95, "description_l": "Pinto Beans", "description_r": "x"},
            {"unique_id_l": "f_true", "unique_id_r": "o9", "match_probability": 0.1, "description_l": "Black Beans", "description_r": "y"},
        ]
    )
    result = _holdout_error_examples(preds, true_labels, threshold=0.5, n=5)
    assert len(result["false_positives"]) == 1
    assert result["false_positives"][0]["same_text_as_true_partner"] is False


def test_false_positive_tag_false_when_true_partners_text_unknown():
    # The true partner's description never appears anywhere in `preds` -- can't confirm
    # a text match, so the tag defaults to False rather than crashing or guessing True.
    true_labels = {"f_true": "o1"}
    preds = _preds(
        [{"unique_id_l": "f_wrong", "unique_id_r": "o1", "match_probability": 0.95, "description_l": "Black Beans", "description_r": "x"}]
    )
    result = _holdout_error_examples(preds, true_labels, threshold=0.5, n=5)
    assert result["false_positives"][0]["same_text_as_true_partner"] is False

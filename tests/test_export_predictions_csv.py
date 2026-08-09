import pandas as pd

from agentic_matching.linking.evaluate import export_predictions_csv

ATTRS = [{"name": "is_greek", "kind": "boolean"}]


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_probability": 0.4,
                "match_weight": -0.5,
                "unique_id_l": "1",
                "description_l": "Yogurt, plain",
                "unique_id_r": "a",
                "description_r": "Plain yogurt",
                "is_greek_l": "False",
                "is_greek_r": "False",
            },
            {
                "match_probability": 0.9,
                "match_weight": 3.2,
                "unique_id_l": "2",
                "description_l": "Yogurt, Greek",
                "unique_id_r": "b",
                "description_r": "Greek yogurt",
                "is_greek_l": "True",
                "is_greek_r": "True",
            },
        ]
    )


def test_writes_one_row_per_prediction(tmp_path):
    out_path = tmp_path / "matches.csv"
    n = export_predictions_csv(_predictions(), ATTRS, out_path)
    assert n == 2
    written = pd.read_csv(out_path)
    assert len(written) == 2


def test_sorted_by_match_probability_descending(tmp_path):
    out_path = tmp_path / "matches.csv"
    export_predictions_csv(_predictions(), ATTRS, out_path)
    written = pd.read_csv(out_path)
    assert written["match_probability"].tolist() == [0.9, 0.4]


def test_renames_id_and_description_columns(tmp_path):
    out_path = tmp_path / "matches.csv"
    export_predictions_csv(_predictions(), ATTRS, out_path)
    written = pd.read_csv(out_path)
    assert {"fndds_id", "fndds_description", "off_code", "off_product_name"} <= set(written.columns)
    assert "unique_id_l" not in written.columns


def test_includes_attribute_columns_both_sides(tmp_path):
    out_path = tmp_path / "matches.csv"
    export_predictions_csv(_predictions(), ATTRS, out_path)
    written = pd.read_csv(out_path)
    assert "is_greek_l" in written.columns
    assert "is_greek_r" in written.columns


def test_empty_predictions_writes_header_only(tmp_path):
    out_path = tmp_path / "matches.csv"
    n = export_predictions_csv(_predictions().iloc[0:0], ATTRS, out_path)
    assert n == 0
    written = pd.read_csv(out_path)
    assert len(written) == 0
    assert "match_probability" in written.columns


def test_top_n_caps_rows_keeping_highest_probability(tmp_path):
    out_path = tmp_path / "matches.csv"
    n = export_predictions_csv(_predictions(), ATTRS, out_path, top_n=1)
    assert n == 1
    written = pd.read_csv(out_path)
    assert written["match_probability"].tolist() == [0.9]


def test_top_n_none_exports_every_row_regardless_of_threshold():
    # No caller-side probability filtering required -- low-probability rows are still
    # written (confidence conveyed by the column, not a cutoff on what gets exported).
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "matches.csv"
        preds = _predictions()
        preds.loc[len(preds)] = {**preds.iloc[0].to_dict(), "match_probability": 0.001, "unique_id_l": "3"}
        n = export_predictions_csv(preds, ATTRS, out_path, top_n=None)
        assert n == 3

import pandas as pd

from agentic_matching.linking.splink_model import normalize_prediction_sides

# Real, verified regression this guards against: splink assigns a link_only
# prediction's `_l`/`_r` suffix by sorting the two tables' `source_dataset` values
# ALPHABETICALLY, not by the order passed to `Linker(...)`. Every consumer in this
# codebase assumes `_l` is always the fndds side -- true only when "fndds" sorts before
# whatever the catalog side is named. It did for "off" ("fndds" < "off") but not for
# "catalog" ("catalog" < "fndds"), which silently reversed which physical side splink
# calls `_l` vs `_r` once the catalog side was renamed -- production's own
# final_matches.csv ended up with fndds_id/catalog_code swapped, and every holdout-eval
# score silently read 0.0 because catalog ids never matched a dict built assuming
# unique_id_r was the catalog side.


def test_leaves_an_already_correct_frame_untouched():
    df = pd.DataFrame(
        {
            "source_dataset_l": ["fndds", "fndds"],
            "source_dataset_r": ["catalog", "catalog"],
            "unique_id_l": ["100", "101"],
            "unique_id_r": ["A1", "A2"],
            "match_probability": [0.9, 0.4],
        }
    )
    result = normalize_prediction_sides(df)
    assert result["unique_id_l"].tolist() == ["100", "101"]
    assert result["unique_id_r"].tolist() == ["A1", "A2"]


def test_swaps_a_reversed_frame():
    # This is splink's actual real-world output shape once "catalog" < "fndds"
    # alphabetically -- source_dataset_l is "catalog", not "fndds".
    df = pd.DataFrame(
        {
            "source_dataset_l": ["catalog", "catalog"],
            "source_dataset_r": ["fndds", "fndds"],
            "unique_id_l": ["A1", "A2"],
            "unique_id_r": ["100", "101"],
            "description_l": ["Widget A", "Widget B"],
            "description_r": ["Fndds Widget", "Fndds Gadget"],
            "match_probability": [0.9, 0.4],
        }
    )
    result = normalize_prediction_sides(df)
    assert result["source_dataset_l"].tolist() == ["fndds", "fndds"]
    assert result["source_dataset_r"].tolist() == ["catalog", "catalog"]
    assert result["unique_id_l"].tolist() == ["100", "101"]
    assert result["unique_id_r"].tolist() == ["A1", "A2"]
    assert result["description_l"].tolist() == ["Fndds Widget", "Fndds Gadget"]
    assert result["description_r"].tolist() == ["Widget A", "Widget B"]
    # match_probability has no _l/_r suffix -- untouched, not accidentally dropped.
    assert result["match_probability"].tolist() == [0.9, 0.4]


def test_missing_source_dataset_columns_is_a_no_op():
    # A frame with no source_dataset_l column at all (e.g. a single-table frame, or
    # already post-processed) -- nothing to check, pass through unchanged.
    df = pd.DataFrame({"unique_id_l": ["1"], "unique_id_r": ["2"], "match_probability": [0.5]})
    result = normalize_prediction_sides(df)
    assert result is df


def test_empty_frame_is_a_no_op():
    df = pd.DataFrame(columns=["source_dataset_l", "source_dataset_r", "unique_id_l", "unique_id_r"])
    result = normalize_prediction_sides(df)
    assert len(result) == 0


def test_mixed_source_dataset_values_raises_rather_than_guess():
    # Every real prediction frame from ONE linker call has a uniform source_dataset_l
    # value across all rows (splink's alphabetical assignment is a per-call constant,
    # not per-row) -- a mix would mean something unexpected is going on, and silently
    # guessing which rows to swap would be worse than a loud, explicit failure.
    df = pd.DataFrame(
        {
            "source_dataset_l": ["fndds", "catalog"],
            "source_dataset_r": ["catalog", "fndds"],
            "unique_id_l": ["100", "A1"],
            "unique_id_r": ["A1", "100"],
        }
    )
    import pytest

    with pytest.raises(ValueError, match="mixed/unexpected"):
        normalize_prediction_sides(df)

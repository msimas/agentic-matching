import pandas as pd

from agentic_matching.linking.evaluate import best_match_per_catalog


def _predictions() -> pd.DataFrame:
    # Two OFF records ("a", "b"); "a" has two competing FNDDS candidates ("1" and "2"),
    # "b" has only one ("1" again -- the same FNDDS record legitimately attaching to a
    # second, different commercial product).
    return pd.DataFrame(
        [
            {"unique_id_l": "1", "unique_id_r": "a", "match_probability": 0.6},
            {"unique_id_l": "2", "unique_id_r": "a", "match_probability": 0.9},
            {"unique_id_l": "1", "unique_id_r": "b", "match_probability": 0.4},
        ]
    )


def test_keeps_only_highest_probability_candidate_per_catalog_record():
    out = best_match_per_catalog(_predictions())
    row_a = out[out["unique_id_r"] == "a"]
    assert len(row_a) == 1
    assert row_a.iloc[0]["unique_id_l"] == "2"
    assert row_a.iloc[0]["match_probability"] == 0.9


def test_does_not_collapse_the_fndds_side():
    # unique_id_l="1" legitimately appears for both OFF records "a" (well, not in the
    # final output since "2" wins there) and "b" -- one FNDDS nutrition profile can
    # attach to many different commercial products.
    out = best_match_per_catalog(_predictions())
    assert set(out["unique_id_r"]) == {"a", "b"}
    assert len(out) == 2


def test_min_probability_filters_before_dedup():
    out = best_match_per_catalog(_predictions(), min_probability=0.5)
    # "b"'s only candidate (0.4) is below the floor -- "b" drops out entirely rather
    # than falling back to a worse match.
    assert set(out["unique_id_r"]) == {"a"}


def test_empty_predictions_returns_empty():
    empty = pd.DataFrame(columns=["unique_id_l", "unique_id_r", "match_probability"])
    out = best_match_per_catalog(empty)
    assert out.empty


def test_all_rows_below_threshold_returns_empty():
    out = best_match_per_catalog(_predictions(), min_probability=0.99)
    assert out.empty

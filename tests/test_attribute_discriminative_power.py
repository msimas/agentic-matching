import pandas as pd

from agentic_matching.linking.evaluate import _attribute_discriminative_power

ATTRS = [{"name": "vegetable_type", "kind": "categorical"}, {"name": "is_frozen", "kind": "boolean"}]


def _frames():
    # 3 FNDDS records, each with a true OFF match plus several decoys. vegetable_type
    # agrees only on the true pairs (a real discriminator); is_frozen is "False"
    # everywhere, so it agrees on true pairs AND decoys alike (not discriminating).
    fndds_df = pd.DataFrame(
        [
            {"unique_id": "f1", "vegetable_type": "onion", "is_frozen": "False"},
            {"unique_id": "f2", "vegetable_type": "broccoli", "is_frozen": "False"},
            {"unique_id": "f3", "vegetable_type": "cauliflower", "is_frozen": "False"},
        ]
    )
    catalog_df = pd.DataFrame(
        [
            {"unique_id": "o1", "vegetable_type": "onion", "is_frozen": "False"},  # true match for f1
            {"unique_id": "o2", "vegetable_type": "broccoli", "is_frozen": "False"},  # true match for f2
            {"unique_id": "o3", "vegetable_type": "cauliflower", "is_frozen": "False"},  # true match for f3
            {"unique_id": "d1", "vegetable_type": "broccoli", "is_frozen": "False"},  # decoy
            {"unique_id": "d2", "vegetable_type": "cauliflower", "is_frozen": "False"},  # decoy
            {"unique_id": "d3", "vegetable_type": "onion", "is_frozen": "False"},  # decoy
        ]
    )
    true_labels = {"f1": "o1", "f2": "o2", "f3": "o3"}
    return fndds_df, catalog_df, true_labels


def test_discriminating_attribute_scores_high_on_true_low_on_decoy():
    fndds_df, catalog_df, true_labels = _frames()
    result = _attribute_discriminative_power(fndds_df, catalog_df, true_labels, ATTRS, n_decoys_per_positive=5, seed=1)
    veg = next(r for r in result if r["attribute"] == "vegetable_type")
    assert veg["agreement_rate_true_pairs"] == 1.0
    assert veg["agreement_rate_decoy_pairs"] < 1.0


def test_non_discriminating_attribute_scores_similarly_on_both():
    fndds_df, catalog_df, true_labels = _frames()
    result = _attribute_discriminative_power(fndds_df, catalog_df, true_labels, ATTRS, n_decoys_per_positive=5, seed=1)
    frozen = next(r for r in result if r["attribute"] == "is_frozen")
    assert frozen["agreement_rate_true_pairs"] == 1.0
    assert frozen["agreement_rate_decoy_pairs"] == 1.0


def test_empty_true_labels_returns_empty_list():
    fndds_df, catalog_df, _ = _frames()
    assert _attribute_discriminative_power(fndds_df, catalog_df, {}, ATTRS) == []


def test_empty_frames_returns_empty_list():
    empty = pd.DataFrame()
    assert _attribute_discriminative_power(empty, empty, {"f1": "o1"}, ATTRS) == []


def test_unknown_attribute_name_skipped_not_errored():
    fndds_df, catalog_df, true_labels = _frames()
    attrs = [{"name": "does_not_exist", "kind": "boolean"}]
    assert _attribute_discriminative_power(fndds_df, catalog_df, true_labels, attrs) == []

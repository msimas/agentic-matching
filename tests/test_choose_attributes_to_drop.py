from agentic_matching.linking.agent_loop import _choose_attributes_to_drop

# Real case this was written to fix: breaded_vegetables, LLM_DEVICE=databricks -- two
# categorical attributes (vegetable_type, preparation_style) on a tiny (12-21 FNDDS)
# block both flagged label_switching in the SAME pass, an artifact of
# splink_model.train's two-attribute EM strategy (each attribute's own pass blocks on
# the OTHER attribute's exact match, shrinking the training pool enough on a block this
# small to produce an unstable fit). Dropping both at once used to hand the next
# retrain an empty attrs list, which build_comparisons correctly rejects with a
# ValueError, taking the whole round (and the run) down.


def _attrs(*names):
    return [{"name": n} for n in names]


def test_normal_case_some_but_not_all_flagged_drops_them_all():
    degenerate = ["a"]
    attrs = _attrs("a", "b")
    to_drop, kept = _choose_attributes_to_drop(degenerate, attrs, [])
    assert to_drop == ["a"]
    assert kept is None


def test_no_degenerate_attributes_is_a_no_op():
    to_drop, kept = _choose_attributes_to_drop([], _attrs("a", "b"), [])
    assert to_drop == []
    assert kept is None


def test_all_attributes_flagged_at_once_keeps_the_best_discriminating_one():
    degenerate = ["vegetable_type", "preparation_style"]
    attrs = _attrs("vegetable_type", "preparation_style")
    discriminative_power = [
        {"attribute": "vegetable_type", "agreement_rate_true_pairs": 0.714, "agreement_rate_decoy_pairs": 0.139},
        {"attribute": "preparation_style", "agreement_rate_true_pairs": 0.2, "agreement_rate_decoy_pairs": 0.18},
    ]
    to_drop, kept = _choose_attributes_to_drop(degenerate, attrs, discriminative_power)
    # vegetable_type's true/decoy gap (0.575) beats preparation_style's (0.02) --
    # vegetable_type is kept, preparation_style is the one actually dropped.
    assert kept == "vegetable_type"
    assert to_drop == ["preparation_style"]


def test_single_remaining_attribute_flagged_alone_is_kept_not_dropped():
    # len(degenerate) >= len(attrs) also covers the single-attribute case: rather than
    # drop the last attribute and crash on zero comparisons, keep it (even though its
    # own discriminative power is presumably weak -- shipping a known-weak model beats
    # crashing the round entirely).
    to_drop, kept = _choose_attributes_to_drop(
        ["only_attr"], _attrs("only_attr"),
        [{"attribute": "only_attr", "agreement_rate_true_pairs": 0.0, "agreement_rate_decoy_pairs": 0.0}],
    )
    assert kept == "only_attr"
    assert to_drop == []


def test_missing_discriminative_power_entry_defaults_to_worst_case():
    # An attribute absent from discriminative_power (e.g. untrained, never got a power
    # estimate at all) shouldn't crash the max() -- it's treated as the worst possible
    # candidate (-1.0), so any attribute WITH real data is preferred over it.
    degenerate = ["untrained_attr", "weak_but_measured"]
    attrs = _attrs("untrained_attr", "weak_but_measured")
    discriminative_power = [
        {"attribute": "weak_but_measured", "agreement_rate_true_pairs": 0.05, "agreement_rate_decoy_pairs": 0.04},
    ]
    to_drop, kept = _choose_attributes_to_drop(degenerate, attrs, discriminative_power)
    assert kept == "weak_but_measured"
    assert to_drop == ["untrained_attr"]

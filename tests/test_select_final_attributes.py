from agentic_matching.attributes.agent_loop import AttributeRound, select_final_attributes

ATTRS_A = [{"name": "bean_type", "kind": "categorical"}]
ATTRS_B = [{"name": "preparation_method", "kind": "categorical"}]


def _round(round_idx, attributes, n_flags):
    flags = [{"attribute_a": "x", "attribute_b": "y", "cramers_v": 0.9}] * n_flags
    return AttributeRound(round=round_idx, attributes=attributes, correlation_flags=flags, rationale="test")


def test_single_round_is_selected():
    r = _round(0, ATTRS_A, n_flags=0)
    assert select_final_attributes([r]) == ATTRS_A


def test_prefers_fewer_correlation_flags():
    r0 = _round(0, ATTRS_A, n_flags=2)
    r1 = _round(1, ATTRS_B, n_flags=0)
    assert select_final_attributes([r0, r1]) == ATTRS_B


def test_last_round_with_more_flags_than_an_earlier_round_is_not_selected():
    # The motivating case: loop hits max_rounds before returning to 0 flags -- the
    # LAST round isn't necessarily the best one produced.
    r0 = _round(0, ATTRS_A, n_flags=0)
    r1 = _round(1, ATTRS_B, n_flags=3)
    assert select_final_attributes([r0, r1]) == ATTRS_A


def test_ties_prefer_latest_round():
    r0 = _round(0, ATTRS_A, n_flags=1)
    r1 = _round(1, ATTRS_B, n_flags=1)
    assert select_final_attributes([r0, r1]) == ATTRS_B


def test_empty_attribute_set_disqualified_even_with_zero_flags():
    empty = _round(0, [], n_flags=0)
    real = _round(1, ATTRS_A, n_flags=2)
    assert select_final_attributes([empty, real]) == ATTRS_A

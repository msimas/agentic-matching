from agentic_matching.attributes.rules import filter_reintroduced_attributes

DROPPED = {
    "name": "beans_preparation_method",
    "kind": "boolean",
    "description": "prep",
    "fndds_keywords": ["refried"],
    "catalog_keywords": ["refried"],
}
KEPT_UNRELATED = {"name": "beans_contains_meat", "kind": "boolean", "description": "meat", "fndds_keywords": ["pork"], "catalog_keywords": ["pork"]}


def test_identical_reintroduction_is_filtered():
    # Same name, same content as what already failed -- a real repeat, filter it.
    new_attrs = [DROPPED, KEPT_UNRELATED]
    filtered, same_as_before, new_definition = filter_reintroduced_attributes(
        new_attrs, ["beans_preparation_method"], {"beans_preparation_method": DROPPED}
    )
    assert filtered == [KEPT_UNRELATED]
    assert same_as_before == ["beans_preparation_method"]
    assert new_definition == []


def test_reintroduction_with_different_definition_passes_through():
    # The exact regression this function fixes: same name, genuinely DIFFERENT
    # keywords -- must NOT be silently discarded by a name-only check.
    redefined = {**DROPPED, "fndds_keywords": ["cooked", "prepared"], "catalog_keywords": ["cooked", "prepared"]}
    new_attrs = [redefined, KEPT_UNRELATED]
    filtered, same_as_before, new_definition = filter_reintroduced_attributes(
        new_attrs, ["beans_preparation_method"], {"beans_preparation_method": DROPPED}
    )
    assert filtered == [redefined, KEPT_UNRELATED]  # let it through, judged fresh
    assert same_as_before == []
    assert new_definition == ["beans_preparation_method"]


def test_unrelated_attribute_never_flagged():
    filtered, same_as_before, new_definition = filter_reintroduced_attributes(
        [KEPT_UNRELATED], ["beans_preparation_method"], {"beans_preparation_method": DROPPED}
    )
    assert filtered == [KEPT_UNRELATED]
    assert same_as_before == []
    assert new_definition == []


def test_name_in_auto_dropped_but_no_captured_definition_passes_through():
    # auto_dropped can list a name whose definition was never captured (defensive --
    # shouldn't normally happen, but nothing to compare against means "let it through"
    # rather than guessing it's a repeat.
    new_attrs = [DROPPED]
    filtered, same_as_before, new_definition = filter_reintroduced_attributes(new_attrs, ["beans_preparation_method"], {})
    assert filtered == [DROPPED]
    assert same_as_before == []
    assert new_definition == ["beans_preparation_method"]


def test_empty_inputs_no_crash():
    filtered, same_as_before, new_definition = filter_reintroduced_attributes([], [], {})
    assert filtered == []
    assert same_as_before == []
    assert new_definition == []

from agentic_matching.attributes.rules import attribute_set_signature

A = {"name": "beans_bean_type", "kind": "categorical", "description": "type", "categories": {"kidney": {"fndds_keywords": ["kidney"], "off_keywords": ["kidney"]}}}
A_REDEFINED = {"name": "beans_bean_type", "kind": "categorical", "description": "type", "categories": {"kidney": {"fndds_keywords": ["kidney", "red kidney"], "off_keywords": ["kidney"]}}}
B = {"name": "beans_contains_meat", "kind": "boolean", "description": "meat", "fndds_keywords": ["pork"], "off_keywords": ["pork"]}


def test_identical_sets_have_equal_signature():
    assert attribute_set_signature([A, B]) == attribute_set_signature([A, B])


def test_order_independent():
    assert attribute_set_signature([A, B]) == attribute_set_signature([B, A])


def test_same_names_different_content_are_not_equal():
    # The exact bug this function fixes: a "redefine" that keeps the name but changes
    # keywords must NOT look identical to a name-only comparison.
    assert attribute_set_signature([A, B]) != attribute_set_signature([A_REDEFINED, B])


def test_dropped_attribute_changes_signature():
    assert attribute_set_signature([A, B]) != attribute_set_signature([A])


def test_added_attribute_changes_signature():
    assert attribute_set_signature([A]) != attribute_set_signature([A, B])


def test_empty_list_signature():
    assert attribute_set_signature([]) == frozenset()

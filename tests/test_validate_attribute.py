from agentic_matching.attributes.rules import filter_valid_attributes, validate_attribute

GOOD_BOOLEAN = {"name": "is_greek", "kind": "boolean", "fndds_keywords": ["greek"], "off_keywords": ["greek", "grec"]}
GOOD_CATEGORICAL = {
    "name": "bean_type",
    "kind": "categorical",
    "categories": {
        "kidney": {"fndds_keywords": ["kidney"], "off_keywords": ["kidney"]},
        "pinto": {"fndds_keywords": ["pinto"], "off_keywords": ["pinto"]},
    },
}


def test_well_formed_boolean_passes():
    assert validate_attribute(GOOD_BOOLEAN) is None


def test_well_formed_categorical_passes():
    assert validate_attribute(GOOD_CATEGORICAL) is None


def test_boolean_missing_fndds_keywords_field_entirely_flagged():
    # The real bug: an LLM revision response omitted fndds_keywords/off_keywords
    # entirely (not even an empty list) for an otherwise-familiar attribute.
    attr = {"name": "contains_meat", "kind": "boolean", "description": "..."}
    reason = validate_attribute(attr)
    assert reason is not None
    assert "contains_meat" in reason
    assert "fndds_keywords" in reason


def test_boolean_empty_off_keywords_list_flagged():
    attr = {"name": "contains_meat", "kind": "boolean", "fndds_keywords": ["pork"], "off_keywords": []}
    reason = validate_attribute(attr)
    assert reason is not None
    assert "off_keywords" in reason


def test_categorical_with_one_category_flagged():
    attr = {"name": "x", "kind": "categorical", "categories": {"only_one": {"fndds_keywords": ["a"], "off_keywords": ["a"]}}}
    reason = validate_attribute(attr)
    assert reason is not None
    assert "fewer than 2 categories" in reason


def test_categorical_no_category_has_fndds_keywords_flagged():
    attr = {
        "name": "x",
        "kind": "categorical",
        "categories": {
            "a": {"fndds_keywords": [], "off_keywords": ["a"]},
            "b": {"fndds_keywords": [], "off_keywords": ["b"]},
        },
    }
    reason = validate_attribute(attr)
    assert reason is not None
    assert "fndds_keywords" in reason


def test_filter_valid_attributes_drops_only_malformed_ones():
    broken = {"name": "contains_meat", "kind": "boolean", "description": "..."}
    kept = filter_valid_attributes([GOOD_BOOLEAN, broken, GOOD_CATEGORICAL])
    assert [a["name"] for a in kept] == ["is_greek", "bean_type"]


def test_filter_valid_attributes_empty_input_returns_empty():
    assert filter_valid_attributes([]) == []

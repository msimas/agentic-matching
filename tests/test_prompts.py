import json

from agentic_matching.llm.prompts import (
    _round_floats,
    _summarize_attributes,
    build_gap_identification_prompt,
    build_keep_drop_prompt,
)


# -- _round_floats ------------------------------------------------------------------


def test_round_floats_rounds_top_level_float():
    assert _round_floats(0.037940379403794036) == 0.0379


def test_round_floats_recurses_into_dicts_and_lists():
    obj = {"a": 0.123456789, "b": [0.111111, {"c": 0.999999}]}
    assert _round_floats(obj) == {"a": 0.1235, "b": [0.1111, {"c": 1.0}]}


def test_round_floats_leaves_ints_strings_none_untouched():
    obj = {"n": 5, "s": "hello", "x": None}
    assert _round_floats(obj) == obj


def test_round_floats_leaves_booleans_untouched_not_coerced_by_round():
    obj = {"flag": True, "other": False}
    result = _round_floats(obj)
    assert result == {"flag": True, "other": False}
    assert result["flag"] is True  # not e.g. 1 or 1.0


def test_round_floats_custom_ndigits():
    assert _round_floats(0.123456, ndigits=2) == 0.12


# -- _summarize_attributes -----------------------------------------------------------


BOOLEAN_ATTR = {
    "name": "beans_contains_meat",
    "kind": "boolean",
    "description": "contains meat",
    "fndds_keywords": ["pork", "bacon", "ham"],
    "off_keywords": ["pork", "bacon", "ham"],
}
CATEGORICAL_ATTR = {
    "name": "beans_bean_type",
    "kind": "categorical",
    "description": "bean type",
    "categories": {
        "kidney": {"fndds_keywords": ["kidney"], "off_keywords": ["kidney"]},
        "pinto": {"fndds_keywords": ["pinto"], "off_keywords": ["pinto"]},
    },
}


def test_summarize_boolean_attribute_drops_keywords():
    result = _summarize_attributes([BOOLEAN_ATTR])
    assert result == [{"name": "beans_contains_meat", "kind": "boolean", "description": "contains meat"}]


def test_summarize_categorical_attribute_keeps_category_names_only():
    result = _summarize_attributes([CATEGORICAL_ATTR])
    assert result == [{"name": "beans_bean_type", "kind": "categorical", "description": "bean type", "categories": ["kidney", "pinto"]}]


def test_summarize_empty_list():
    assert _summarize_attributes([]) == []


# -- keyword bulk actually excluded from the keep_drop / gap prompts ----------------


def test_keep_drop_prompt_excludes_attribute_keywords():
    _, user = build_keep_drop_prompt("beans", [BOOLEAN_ATTR, CATEGORICAL_ATTR], None, None, None)
    payload = json.loads(user)
    assert payload["existing_attributes"] == _summarize_attributes([BOOLEAN_ATTR, CATEGORICAL_ATTR])
    assert "bacon" not in user  # a keyword that only appears inside the trimmed field
    assert "fndds_keywords" not in user  # category names ("kidney"/"pinto") are kept, keyword lists are not


def test_gap_identification_prompt_excludes_attribute_keywords():
    error_examples = {"false_positives": [], "false_negatives": []}
    _, user = build_gap_identification_prompt("beans", error_examples, [BOOLEAN_ATTR, CATEGORICAL_ATTR], None)
    payload = json.loads(user)
    assert payload["existing_attributes"] == _summarize_attributes([BOOLEAN_ATTR, CATEGORICAL_ATTR])
    assert "bacon" not in user
    assert "fndds_keywords" not in user  # category names ("kidney"/"pinto") are kept, keyword lists are not

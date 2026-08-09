from agentic_matching.attributes.rules import apply_attribute, compute_attribute_values
from agentic_matching.attributes.seed_rules import SEED_ATTRIBUTES

IS_GREEK = next(a for a in SEED_ATTRIBUTES["yogurt"] if a["name"] == "is_greek")
FAT_LEVEL = next(a for a in SEED_ATTRIBUTES["yogurt"] if a["name"] == "fat_level")


def test_boolean_attribute_true_on_fndds():
    assert apply_attribute(IS_GREEK, "Yogurt, Greek, plain, whole milk", "fndds") is True


def test_boolean_attribute_false_when_keyword_absent():
    assert apply_attribute(IS_GREEK, "Yogurt, plain, whole milk", "fndds") is False


def test_boolean_attribute_matches_off_side_synonym():
    assert apply_attribute(IS_GREEK, "Yaourt grec nature", "off") is True


def test_boolean_attribute_handles_none_text():
    assert apply_attribute(IS_GREEK, None, "fndds") is False


def test_categorical_attribute_picks_matching_category():
    assert apply_attribute(FAT_LEVEL, "Yogurt, nonfat, plain", "fndds") == "fat_free"


def test_categorical_attribute_no_match_returns_none():
    result = apply_attribute(FAT_LEVEL, "Yogurt with no fat-related keywords at all", "fndds")
    assert result is None


def test_unknown_kind_raises():
    bad_attr = {"kind": "mystery", "name": "x"}
    try:
        apply_attribute(bad_attr, "text", "fndds")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown attribute kind")


def test_compute_attribute_values_vectorizes_over_texts():
    texts = ["Greek yogurt, plain", "Whole milk yogurt", None]
    result = compute_attribute_values([IS_GREEK], texts, side="fndds")
    assert result["is_greek"] == [True, False, False]

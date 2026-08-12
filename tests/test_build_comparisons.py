import pytest

from agentic_matching.linking.splink_model import build_comparisons

ATTR_A = {"name": "beans_bean_type", "kind": "categorical"}
ATTR_B = {"name": "beans_contains_meat", "kind": "boolean"}


def test_one_exact_match_comparison_per_attribute():
    comparisons = build_comparisons([ATTR_A, ATTR_B])
    assert len(comparisons) == 2


def test_no_description_comparison_appended():
    # The whole point of removing it (see this function's docstring) -- verify there's
    # nothing beyond one comparison per attribute, not an extra fixed one tacked on.
    comparisons = build_comparisons([ATTR_A])
    assert len(comparisons) == 1


def test_empty_attrs_raises_clear_error_not_silently_returning_empty_list():
    # Verified real crash (LLM_DEVICE=ollama, qwen3:4b-instruct-2507-q4_K_M): with
    # `description` removed, an empty attrs list used to silently produce an empty
    # comparisons list, which splink then failed on deep inside its own SQL generation
    # (a cryptic SplinkException/DuckDB parser error on a dangling-comma bayes-factor
    # expression) instead of a clear, immediate, actionable failure right here.
    with pytest.raises(ValueError, match="no matching attributes"):
        build_comparisons([])

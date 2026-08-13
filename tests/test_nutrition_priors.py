import copy

import pytest

from agentic_matching.linking.nutrition_priors import (
    CALORIE_DENSE_PRIOR,
    DIETARY_CLASSIFICATION_PRIOR,
    MAX_PRIOR_WEIGHT,
    NUTRIENT_DENSE_PRIOR,
    PRIOR_CONFIDENCE_PAIRS,
    apply_nutrition_priors,
    classify_nutrition_significance,
)

# -- classify_nutrition_significance ---------------------------------------------


def test_classifies_nutrient_dense_from_name():
    assert classify_nutrition_significance({"name": "beans_contains_meat", "description": ""}) == "nutrient_dense"


def test_classifies_nutrient_dense_from_description():
    attr = {"name": "beans_extra", "description": "Whether the product contains dairy."}
    assert classify_nutrition_significance(attr) == "nutrient_dense"


def test_classifies_calorie_dense():
    attr = {"name": "yogurt_has_honey", "description": "Whether honey is added."}
    assert classify_nutrition_significance(attr) == "calorie_dense"


def test_nutrient_dense_checked_before_calorie_dense_when_both_present():
    attr = {"name": "x", "description": "contains meat and sugar"}
    assert classify_nutrition_significance(attr) == "nutrient_dense"


def test_classifies_dietary_classification_vegan():
    attr = {"name": "beans_is_vegan", "description": "Whether the product is labeled vegan."}
    assert classify_nutrition_significance(attr) == "dietary_classification"


def test_classifies_dietary_classification_vegetarian():
    attr = {"name": "beans_is_vegetarian", "description": "Whether the product is vegetarian."}
    assert classify_nutrition_significance(attr) == "dietary_classification"


def test_classifies_dietary_classification_plant_based_hyphenated_and_spaced():
    assert classify_nutrition_significance({"name": "x", "description": "plant-based product"}) == (
        "dietary_classification"
    )
    assert classify_nutrition_significance({"name": "x", "description": "plant based product"}) == (
        "dietary_classification"
    )


def test_dietary_classification_checked_before_nutrient_dense():
    # "vegetarian" attributes often mention "meat" in their own description (e.g.
    # "contains no meat") -- must classify as dietary_classification, not
    # nutrient_dense, despite "meat" being a NUTRIENT_DENSE_TERMS member too.
    attr = {"name": "beans_is_vegetarian", "description": "Indicates the product is vegetarian (contains no meat)."}
    assert classify_nutrition_significance(attr) == "dietary_classification"


def test_unrelated_attribute_returns_none():
    attr = {"name": "beans_bean_type", "description": "Specific type of bean (kidney, pinto, black)."}
    assert classify_nutrition_significance(attr) is None


def test_missing_description_does_not_error():
    assert classify_nutrition_significance({"name": "beans_contains_meat"}) == "nutrient_dense"


# -- apply_nutrition_priors -------------------------------------------------------


def _settings_with_comparison(name: str, m: float, u: float) -> dict:
    return {
        "comparisons": [
            {
                "output_column_name": name,
                "comparison_levels": [
                    {"is_null_level": True},
                    {"sql_condition": "exact", "m_probability": m, "u_probability": u},
                    {"sql_condition": "else", "m_probability": 1 - m, "u_probability": 1 - u},
                ],
            }
        ]
    }


def test_non_nutrition_attribute_left_unchanged():
    settings = _settings_with_comparison("beans_bean_type", 0.5, 0.4)
    attrs = [{"name": "beans_bean_type", "description": "type of bean"}]
    result = apply_nutrition_priors(settings, attrs, [])
    assert result == settings


def test_original_settings_dict_not_mutated():
    settings = _settings_with_comparison("beans_contains_meat", 0.55, 0.48)
    attrs = [{"name": "beans_contains_meat", "description": "contains meat"}]
    original = copy.deepcopy(settings)
    apply_nutrition_priors(settings, attrs, [{"attribute": "beans_contains_meat", "n_true_pairs": 500}])
    assert settings == original


def test_nutrition_significant_attribute_blended_toward_prior():
    # m_data == u_data -> zero inherent separation (strength_weight=0), combined with
    # n_true_pairs=0 (sample_weight=0) -> weight=0 -> fully the prior. (A nonzero-gap
    # m_data/u_data pair at n_true_pairs=0 is covered separately by
    # test_already_strong_fit_overrides_the_prior_regardless_of_sample_size below --
    # since _blend became strength-aware, "zero sample count" alone no longer implies
    # "zero weight" when the fit's own separation is real.)
    m_data, u_data = 0.5, 0.5
    settings = _settings_with_comparison("beans_contains_meat", m_data, u_data)
    attrs = [{"name": "beans_contains_meat", "description": "contains meat"}]
    result = apply_nutrition_priors(settings, attrs, [{"attribute": "beans_contains_meat", "n_true_pairs": 0}])
    levels = result["comparisons"][0]["comparison_levels"]
    exact, other = levels[1], levels[2]
    assert exact["m_probability"] == pytest.approx(NUTRIENT_DENSE_PRIOR[0])
    assert exact["u_probability"] == pytest.approx(NUTRIENT_DENSE_PRIOR[1])
    assert other["m_probability"] == pytest.approx(1 - NUTRIENT_DENSE_PRIOR[0])
    assert other["u_probability"] == pytest.approx(1 - NUTRIENT_DENSE_PRIOR[1])


def test_already_strong_fit_overrides_the_prior_regardless_of_sample_size():
    # Real case this exists for (breaded_vegetables, LLM_DEVICE=databricks):
    # vegetable_type's real EM fit (m~0.9997, u~0.077) already separates far more
    # confidently than CALORIE_DENSE_PRIOR itself asserts, even with n_true_pairs=0 --
    # the prior should contribute ~nothing, not pull a strong fit down toward its own
    # weaker assertion.
    m_data, u_data = 0.9997, 0.077
    settings = _settings_with_comparison("breaded_vegetables_vegetable_type", m_data, u_data)
    attrs = [{"name": "breaded_vegetables_vegetable_type", "description": "The specific vegetable being fried or breaded."}]
    result = apply_nutrition_priors(
        settings, attrs, [{"attribute": "breaded_vegetables_vegetable_type", "n_true_pairs": 0}]
    )
    exact = result["comparisons"][0]["comparison_levels"][1]
    assert exact["m_probability"] == pytest.approx(m_data, abs=1e-9)
    assert exact["u_probability"] == pytest.approx(u_data, abs=1e-9)


def test_weak_fit_still_falls_back_to_sample_size_floor():
    # The real, previously-tuned beans_contains_meat case (see MAX_PRIOR_WEIGHT's
    # docstring): a weak/ambiguous EM-only fit, well-populated with true pairs, should
    # land on exactly the sample-size-only weight -- strength_weight contributes
    # ~nothing here since the fit barely separates match from non-match at all.
    m_data, u_data = 0.547, 0.479
    settings = _settings_with_comparison("beans_contains_meat", m_data, u_data)
    attrs = [{"name": "beans_contains_meat", "description": "contains meat"}]
    result = apply_nutrition_priors(settings, attrs, [{"attribute": "beans_contains_meat", "n_true_pairs": 500}])
    exact = result["comparisons"][0]["comparison_levels"][1]
    expected_m = MAX_PRIOR_WEIGHT * m_data + (1 - MAX_PRIOR_WEIGHT) * NUTRIENT_DENSE_PRIOR[0]
    assert exact["m_probability"] == pytest.approx(expected_m)


def test_weight_caps_below_one_even_with_abundant_data():
    m_data, u_data = 0.55, 0.48
    settings = _settings_with_comparison("beans_contains_meat", m_data, u_data)
    attrs = [{"name": "beans_contains_meat", "description": "contains meat"}]
    huge_n = PRIOR_CONFIDENCE_PAIRS * 100
    result = apply_nutrition_priors(settings, attrs, [{"attribute": "beans_contains_meat", "n_true_pairs": huge_n}])
    exact = result["comparisons"][0]["comparison_levels"][1]
    expected_m = MAX_PRIOR_WEIGHT * m_data + (1 - MAX_PRIOR_WEIGHT) * NUTRIENT_DENSE_PRIOR[0]
    assert exact["m_probability"] == pytest.approx(expected_m)
    # Never fully trusts EM regardless of sample size.
    assert exact["m_probability"] != pytest.approx(m_data)


def test_calorie_dense_uses_its_own_weaker_prior():
    settings = _settings_with_comparison("yogurt_has_honey", 0.5, 0.5)
    attrs = [{"name": "yogurt_has_honey", "description": "added honey"}]
    result = apply_nutrition_priors(settings, attrs, [{"attribute": "yogurt_has_honey", "n_true_pairs": 0}])
    exact = result["comparisons"][0]["comparison_levels"][1]
    assert exact["m_probability"] == pytest.approx(CALORIE_DENSE_PRIOR[0])
    assert exact["m_probability"] != pytest.approx(NUTRIENT_DENSE_PRIOR[0])


def test_dietary_classification_attribute_uses_its_own_prior():
    settings = _settings_with_comparison("beans_is_vegan", 0.5, 0.5)
    attrs = [{"name": "beans_is_vegan", "description": "Whether the product is vegan."}]
    result = apply_nutrition_priors(settings, attrs, [{"attribute": "beans_is_vegan", "n_true_pairs": 0}])
    exact = result["comparisons"][0]["comparison_levels"][1]
    assert exact["m_probability"] == pytest.approx(DIETARY_CLASSIFICATION_PRIOR[0])
    assert exact["u_probability"] == pytest.approx(DIETARY_CLASSIFICATION_PRIOR[1])


def test_missing_discriminative_power_entry_defaults_to_zero_true_pairs():
    # No entry for this attribute in discriminative_power -> n_true_pairs=0 -> prior dominates fully.
    settings = _settings_with_comparison("beans_contains_meat", 0.9, 0.9)
    attrs = [{"name": "beans_contains_meat", "description": "contains meat"}]
    result = apply_nutrition_priors(settings, attrs, [])
    exact = result["comparisons"][0]["comparison_levels"][1]
    assert exact["m_probability"] == pytest.approx(NUTRIENT_DENSE_PRIOR[0])


def test_non_two_level_comparison_skipped_without_error():
    settings = {
        "comparisons": [
            {
                "output_column_name": "beans_contains_meat",
                "comparison_levels": [
                    {"is_null_level": True},
                    {"sql_condition": "high", "m_probability": 0.7, "u_probability": 0.2},
                    {"sql_condition": "medium", "m_probability": 0.2, "u_probability": 0.3},
                    {"sql_condition": "low", "m_probability": 0.1, "u_probability": 0.5},
                ],
            }
        ]
    }
    attrs = [{"name": "beans_contains_meat", "description": "contains meat"}]
    result = apply_nutrition_priors(settings, attrs, [{"attribute": "beans_contains_meat", "n_true_pairs": 0}])
    assert result == settings


def test_description_comparison_never_touched_by_apply_nutrition_priors():
    settings = _settings_with_comparison("description", 0.9, 0.1)
    # Even if somehow classified (it wouldn't be, since it's not in `attrs`), only
    # comparisons whose column matches an attribute name are ever considered.
    attrs = [{"name": "beans_contains_meat", "description": "contains meat"}]
    result = apply_nutrition_priors(settings, attrs, [])
    assert result == settings


# dampen_description_weight and its tests were removed along with the `description`
# splink comparison itself -- see nutrition_priors.py's module docstring for why.


# -- untrained comparison (m_probability/u_probability missing) -- real verified crash --


def test_apply_nutrition_priors_untrained_attribute_does_not_crash():
    # Verified real crash (beans, LLM_DEVICE=databricks): EM doesn't always fully
    # train every comparison every round ("no m values are trained" is a real, valid
    # splink outcome) -- exact_level["m_probability"] can be missing, and a plain
    # dict[...] access raised KeyError, taking down the whole round.
    settings = {
        "comparisons": [
            {
                "output_column_name": "beans_contains_meat",
                "comparison_levels": [
                    {"is_null_level": True},
                    {"sql_condition": "exact"},  # no m_probability/u_probability keys at all
                    {"sql_condition": "else"},
                ],
            }
        ]
    }
    attrs = [{"name": "beans_contains_meat", "description": "contains meat"}]
    result = apply_nutrition_priors(settings, attrs, [{"attribute": "beans_contains_meat", "n_true_pairs": 500}])
    exact = result["comparisons"][0]["comparison_levels"][1]
    # No EM estimate to blend with -> trust the prior alone.
    assert exact["m_probability"] == pytest.approx(NUTRIENT_DENSE_PRIOR[0])
    assert exact["u_probability"] == pytest.approx(NUTRIENT_DENSE_PRIOR[1])


def test_apply_nutrition_priors_untrained_with_none_values_does_not_crash():
    settings = _settings_with_comparison("beans_contains_meat", 0.5, 0.5)
    settings["comparisons"][0]["comparison_levels"][1]["m_probability"] = None
    attrs = [{"name": "beans_contains_meat", "description": "contains meat"}]
    result = apply_nutrition_priors(settings, attrs, [{"attribute": "beans_contains_meat", "n_true_pairs": 500}])
    exact = result["comparisons"][0]["comparison_levels"][1]
    assert exact["m_probability"] == pytest.approx(NUTRIENT_DENSE_PRIOR[0])



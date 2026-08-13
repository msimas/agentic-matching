from agentic_matching.linking.evaluate import _category_level_score, _catalog_to_true_fndds


def test_catalog_to_true_fndds_reverses_the_mapping():
    true_labels = {"f1": "o1", "f2": "o2"}
    assert _catalog_to_true_fndds(true_labels) == {"o1": {"f1"}, "o2": {"f2"}}


def test_catalog_to_true_fndds_groups_multiple_fndds_ids_under_one_catalog_id():
    # Verified real data quirk on the beans holdout: one catalog_code labeled as the true
    # partner for several different fdc_ids.
    true_labels = {"f1": "o1", "f2": "o1", "f3": "o1"}
    assert _catalog_to_true_fndds(true_labels) == {"o1": {"f1", "f2", "f3"}}


def test_exact_match_counts_as_correct():
    predicted = {"o1": "f1"}
    true_partners = {"o1": {"f1"}}
    fndds_desc = {"f1": "Black Beans"}
    result = _category_level_score(predicted, true_partners, fndds_desc)
    assert result["n_category_correct"] == 1
    assert result["category_precision"] == 1.0
    assert result["category_recall"] == 1.0
    assert result["category_f1"] == 1.0


def test_one_of_several_true_partners_counts_as_correct():
    # catalog_id has multiple genuinely-true fdc_ids (see _catalog_to_true_fndds) -- landing on
    # ANY of them is correct, not just whichever was sampled first.
    predicted = {"o1": "f2"}
    true_partners = {"o1": {"f1", "f2", "f3"}}
    fndds_desc = {"f1": "a", "f2": "b", "f3": "c"}
    result = _category_level_score(predicted, true_partners, fndds_desc)
    assert result["n_category_correct"] == 1


def test_different_fndds_id_same_description_counts_as_correct():
    # The exact bug this function exists to route around: two different fdc_ids, same
    # product text -- exact-id scoring calls this wrong, category-level calls it right.
    predicted = {"o1": "f_wrong_id"}
    true_partners = {"o1": {"f_true_id"}}
    fndds_desc = {"f_wrong_id": "BLACK BEANS", "f_true_id": "black beans"}  # case differs too
    result = _category_level_score(predicted, true_partners, fndds_desc)
    assert result["n_category_correct"] == 1
    assert result["category_precision"] == 1.0


def test_genuinely_different_product_is_incorrect():
    predicted = {"o1": "f_pinto"}
    true_partners = {"o1": {"f_black"}}
    fndds_desc = {"f_pinto": "Pinto Beans", "f_black": "Black Beans"}
    result = _category_level_score(predicted, true_partners, fndds_desc)
    assert result["n_category_correct"] == 0
    assert result["category_precision"] == 0.0
    assert result["category_recall"] == 0.0
    assert result["category_f1"] == 0.0


def test_missing_description_neither_correct_nor_crashes():
    predicted = {"o1": "f_unknown"}
    true_partners = {"o1": {"f_true"}}
    fndds_desc = {"f_true": "Black Beans"}  # f_unknown has no description on file
    result = _category_level_score(predicted, true_partners, fndds_desc)
    assert result["n_category_correct"] == 0
    assert result["category_precision"] == 0.0  # still counted in the denominator (n_pred)


def test_predicted_catalog_id_with_no_true_partners_is_skipped():
    predicted = {"o1": "f1", "o_decoy": "f2"}
    true_partners = {"o1": {"f1"}}
    fndds_desc = {"f1": "Black Beans", "f2": "Black Beans"}
    result = _category_level_score(predicted, true_partners, fndds_desc)
    # o_decoy has no gold partner to compare against -- excluded, not a false anything.
    assert result["n_category_correct"] == 1
    assert result["category_precision"] == 0.5  # 1 correct / 2 predicted total


def test_empty_predicted_gives_zero_precision_but_no_crash():
    result = _category_level_score({}, {"o1": {"f1"}}, {"f1": "Black Beans"})
    assert result["category_precision"] == 0.0
    assert result["category_recall"] == 0.0
    assert result["category_f1"] == 0.0


def test_empty_true_partners_gives_zero_recall_but_no_crash():
    result = _category_level_score({"o1": "f1"}, {}, {"f1": "Black Beans"})
    assert result["category_recall"] == 0.0

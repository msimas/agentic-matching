from agentic_matching.linking.evaluate import _resolvable_ceiling


def test_no_impostors_all_resolvable():
    true_partners = {"o1": {"f1"}, "o2": {"f2"}}
    fndds_desc = {"f1": "Black Beans", "f2": "Pinto Beans"}
    result = _resolvable_ceiling(true_partners, fndds_desc)
    assert result["n_resolvable"] == 2
    assert result["n_ambiguous"] == 0
    assert result["max_achievable_f1"] == 1.0
    assert result["max_achievable_precision"] == 1.0
    assert result["max_achievable_recall"] == 1.0


def test_impostor_with_identical_text_makes_it_ambiguous():
    # f1 is o1's true partner; f_impostor shares f1's exact description text but is
    # NOT a true partner of o1 (and isn't anyone else's true partner either here) --
    # no text signal can tell f1 apart from f_impostor, so o1 is ambiguous.
    true_partners = {"o1": {"f1"}}
    fndds_desc = {"f1": "Black Beans", "f_impostor": "black beans"}  # case differs, still a collision
    result = _resolvable_ceiling(true_partners, fndds_desc)
    assert result["n_resolvable"] == 0
    assert result["n_ambiguous"] == 1
    assert result["max_achievable_f1"] == 0.0


def test_shared_text_among_a_true_items_own_partners_is_not_an_impostor():
    # catalog_id's true set already contains BOTH fdc_ids sharing that text -- landing on
    # either is correct (see _catalog_to_true_fndds), so this is still resolvable.
    true_partners = {"o1": {"f1", "f2"}}
    fndds_desc = {"f1": "Black Beans", "f2": "black beans"}
    result = _resolvable_ceiling(true_partners, fndds_desc)
    assert result["n_resolvable"] == 1
    assert result["n_ambiguous"] == 0


def test_mixed_resolvable_and_ambiguous():
    true_partners = {"o1": {"f1"}, "o2": {"f2"}}
    fndds_desc = {"f1": "Black Beans", "f2": "Pinto Beans", "f_impostor": "pinto beans"}
    result = _resolvable_ceiling(true_partners, fndds_desc)
    assert result["n_resolvable"] == 1  # o1 (f1's text is unique)
    assert result["n_ambiguous"] == 1  # o2 (f2 collides with f_impostor)
    assert result["max_achievable_f1"] == 0.5


def test_true_fndds_id_missing_its_own_description_is_not_falsely_flagged():
    # A true fdc_id with no description on file contributes no comparable text -- it
    # shouldn't spuriously collide with anything (empty text set -> no impostors found).
    true_partners = {"o1": {"f1"}}
    fndds_desc = {}
    result = _resolvable_ceiling(true_partners, fndds_desc)
    assert result["n_resolvable"] == 1
    assert result["n_ambiguous"] == 0


def test_empty_true_partners_gives_none_ceiling_not_crash():
    result = _resolvable_ceiling({}, {"f1": "Black Beans"})
    assert result["n_resolvable"] == 0
    assert result["n_ambiguous"] == 0
    assert result["max_achievable_f1"] is None

from agentic_matching.attributes.info_requests import MAX_REQUESTS, fulfill_requests


def test_term_frequency_counts_both_sides():
    fndds_texts = ["Black beans, canned", "Pinto beans"]
    catalog_texts = ["Black Beans", "Kidney Beans", "Chili with beans"]
    answers = fulfill_requests([{"kind": "term_frequency", "term": "black"}], fndds_texts, catalog_texts)
    assert answers[0]["kind"] == "term_frequency"
    assert answers[0]["fndds_count"] == 1
    assert answers[0]["catalog_count"] == 1
    assert answers[0]["fndds_fraction"] == 0.5


def test_term_frequency_case_insensitive():
    answers = fulfill_requests([{"kind": "term_frequency", "term": "BLACK"}], ["black beans"], [])
    assert answers[0]["fndds_count"] == 1


def test_sample_records_returns_matching_examples():
    fndds_texts = ["Black beans, canned", "Pinto beans", "Black beans, dried"]
    answers = fulfill_requests([{"kind": "sample_records", "term": "black", "side": "fndds"}], fndds_texts, [])
    assert answers[0]["kind"] == "sample_records"
    assert set(answers[0]["examples"]) == {"Black beans, canned", "Black beans, dried"}


def test_sample_records_defaults_to_fndds_side():
    answers = fulfill_requests([{"kind": "sample_records", "term": "x"}], ["x"], ["x"])
    assert answers[0]["side"] == "fndds"


def test_sample_records_capped_at_max():
    fndds_texts = [f"black beans {i}" for i in range(20)]
    answers = fulfill_requests([{"kind": "sample_records", "term": "black", "side": "fndds"}], fndds_texts, [])
    assert len(answers[0]["examples"]) == 5


def test_unrecognized_kind_returns_error_not_exception():
    answers = fulfill_requests([{"kind": "bogus", "term": "x"}], [], [])
    assert answers[0]["kind"] == "bogus"
    assert "error" in answers[0]


def test_missing_term_returns_error():
    answers = fulfill_requests([{"kind": "term_frequency"}], [], [])
    assert "error" in answers[0]


def test_requests_capped_at_max():
    requested = [{"kind": "term_frequency", "term": f"t{i}"} for i in range(MAX_REQUESTS + 5)]
    answers = fulfill_requests(requested, ["t"], ["t"])
    assert len(answers) == MAX_REQUESTS


def test_empty_requested_list():
    assert fulfill_requests([], [], []) == []


def test_none_requested():
    assert fulfill_requests(None, [], []) == []


def test_empty_text_lists_do_not_divide_by_zero():
    answers = fulfill_requests([{"kind": "term_frequency", "term": "x"}], [], [])
    assert answers[0]["fndds_fraction"] == 0.0
    assert answers[0]["catalog_fraction"] == 0.0

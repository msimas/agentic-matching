from agentic_matching.profiling import _rank_terms

# Synthetic frequency table -- these tests exercise the pure filter/sort/format logic
# only, so they don't require the full data pipeline (download/convert/build) to have
# been run first, unlike profiling.build()'s output.
FREQ = {"yogurt": 200, "greek": 80, "protein": 5000, "rice": 3000, "abc": 50, "milk": 900}
N = 10_000


def test_min_doc_count_excludes_below_floor():
    terms = _rank_terms(FREQ, N, k=40, min_len=4, min_doc_count=1000, max_terms=200)
    names = {t["term"] for t in terms}
    assert names == {"protein", "rice"}


def test_min_doc_count_includes_at_or_above_floor():
    terms = _rank_terms(FREQ, N, k=40, min_len=4, min_doc_count=3000, max_terms=200)
    names = {t["term"] for t in terms}
    assert names == {"protein", "rice"}  # rice is exactly at the floor


def test_top_k_mode_ignores_min_doc_count_when_unset():
    terms = _rank_terms(FREQ, N, k=2, min_len=4, min_doc_count=None, max_terms=200)
    assert [t["term"] for t in terms] == ["protein", "rice"]


def test_min_len_filters_short_tokens():
    terms = _rank_terms(FREQ, N, k=40, min_len=4, min_doc_count=None, max_terms=200)
    assert "abc" not in {t["term"] for t in terms}


def test_results_sorted_descending_by_doc_count():
    terms = _rank_terms(FREQ, N, k=40, min_len=4, min_doc_count=None, max_terms=200)
    counts = [t["doc_count"] for t in terms]
    assert counts == sorted(counts, reverse=True)


def test_catalog_fraction_computed_correctly():
    terms = _rank_terms(FREQ, N, k=40, min_len=4, min_doc_count=None, max_terms=200)
    by_term = {t["term"]: t for t in terms}
    assert by_term["protein"]["catalog_fraction"] == 0.5
    assert by_term["yogurt"]["catalog_fraction"] == 0.02


def test_max_terms_caps_result_length():
    terms = _rank_terms(FREQ, N, k=40, min_len=4, min_doc_count=1, max_terms=2)
    assert len(terms) == 2

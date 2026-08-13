import duckdb

from agentic_matching.calibration import _description_similarity_sql


def _similarity(text_a: str | None, text_b: str | None) -> float:
    con = duckdb.connect()
    sql = _description_similarity_sql("a", "b")
    row = con.execute(f"SELECT {sql} FROM (SELECT ? AS a, ? AS b)", [text_a, text_b]).fetchone()
    con.close()
    return row[0]


def test_identical_text_is_similarity_one():
    assert _similarity("Black Beans", "Black Beans") == 1.0


def test_case_insensitive():
    assert _similarity("BLACK BEANS", "black beans") == 1.0


def test_disjoint_text_is_similarity_zero():
    assert _similarity("Black Beans", "Vanilla Yogurt") == 0.0


def test_partial_overlap_is_between_zero_and_one():
    # {black, beans} vs {beans, low, sodium} -> intersection={beans}, union has 4 -> 0.25
    sim = _similarity("Black Beans", "Beans Low Sodium")
    assert 0.0 < sim < 1.0


def test_both_empty_is_zero_not_a_crash():
    # Real verified case: this is what a NULL/empty OFF product_name produces --
    # treated as zero similarity (no signal), not undefined/NaN.
    assert _similarity(None, None) == 0.0
    assert _similarity("", "") == 0.0


def test_one_side_empty_is_zero():
    assert _similarity("Black Beans", None) == 0.0
    assert _similarity(None, "Black Beans") == 0.0


def test_word_order_does_not_matter():
    # Jaccard is set-based -- word order is irrelevant, unlike e.g. Jaro-Winkler.
    assert _similarity("Beans Black Canned", "Canned Black Beans") == 1.0


def test_real_verified_case_cross_language_listing_scores_zero():
    # Real case found in this project's own gold_pairs data (English Branded
    # description vs. French OFF product name for the SAME physical product) -- a zero
    # score here is expected and correct for a naive bag-of-words check; this is
    # exactly why description_similarity is a review signal, not an auto-filter (see
    # _DESCRIPTION_SIMILARITY_SQL's docstring in calibration.py).
    sim = _similarity(
        "DARE, BRETON, WHITE BEAN CRACKERS WITH SALT & PEPPER",
        "Craquelin haricots blancs avec sel et poivre",
    )
    assert sim == 0.0

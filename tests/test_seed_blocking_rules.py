import duckdb

from agentic_matching.blocking.rules import fndds_predicate_sql, off_predicate_sql

YOGURT_RULE = {
    "fndds": {"keywords": ["yogurt", "yoghurt"], "exclude_keywords": ["frozen yogurt"]},
    "off": {"keywords": ["yogurt", "yoghurt", "yogourt"], "exclude_keywords": []},
}


def _member(rule: dict, side: str, text: str) -> bool:
    con = duckdb.connect()
    text_col = "description" if side == "fndds" else "search_text"
    predicate = (
        fndds_predicate_sql(rule, category_col=None) if side == "fndds" else off_predicate_sql(rule, category_col=None)
    )
    row = con.execute(f"SELECT {predicate} FROM (SELECT lower(?) AS {text_col})", [text]).fetchone()
    con.close()
    return bool(row[0])


def test_fndds_keyword_match():
    assert _member(YOGURT_RULE, "fndds", "Yogurt, plain, whole milk")


def test_fndds_no_match_unrelated():
    assert not _member(YOGURT_RULE, "fndds", "Cheese, cheddar")


def test_fndds_exclude_keyword_overrides_include():
    assert not _member(YOGURT_RULE, "fndds", "Frozen yogurt, chocolate")


def test_off_keyword_match_alt_spelling():
    assert _member(YOGURT_RULE, "off", "Yoghurt nature")


def test_off_no_match_unrelated():
    assert not _member(YOGURT_RULE, "off", "Whole wheat bread")


def test_empty_keywords_matches_nothing():
    empty_rule = {"fndds": {"keywords": [], "exclude_keywords": []}, "off": {"keywords": [], "exclude_keywords": []}}
    assert not _member(empty_rule, "fndds", "Yogurt, plain")


# -- structured category predicate ------------------------------------------------


def _category_member(rule: dict, side: str, category_value, text: str = "") -> bool:
    con = duckdb.connect()
    if side == "fndds":
        predicate = fndds_predicate_sql(rule, text_col="description", category_col="wweia_food_category_description")
        row = con.execute(
            f"SELECT {predicate} FROM (SELECT lower(?) AS description, ? AS wweia_food_category_description)",
            [text, category_value],
        ).fetchone()
    else:
        predicate = off_predicate_sql(rule, text_col="search_text", category_col="categories_tags")
        row = con.execute(
            f"SELECT {predicate} FROM (SELECT lower(?) AS search_text, ? AS categories_tags)",
            [text, category_value],
        ).fetchone()
    con.close()
    return bool(row[0])


def test_fndds_category_exact_match():
    rule = {"fndds": {"keywords": [], "categories": ["Yogurt, regular"]}}
    assert _category_member(rule, "fndds", "Yogurt, regular")


def test_fndds_category_no_match():
    rule = {"fndds": {"keywords": [], "categories": ["Yogurt, regular"]}}
    assert not _category_member(rule, "fndds", "Chicken, whole pieces")


def test_fndds_category_case_insensitive():
    rule = {"fndds": {"keywords": [], "categories": ["Yogurt, Regular".upper()]}}
    assert _category_member(rule, "fndds", "yogurt, regular")


def test_off_category_array_contains():
    rule = {"off": {"keywords": [], "categories": ["en:yogurts"]}}
    assert _category_member(rule, "off", ["en:dairies", "en:yogurts"])


def test_off_category_no_match():
    rule = {"off": {"keywords": [], "categories": ["en:yogurts"]}}
    assert not _category_member(rule, "off", ["en:snacks", "en:sweet-snacks"])


def test_category_or_keyword_either_matches():
    rule = {"fndds": {"keywords": ["yogurt"], "categories": ["Yogurt, regular"]}}
    # Matches via category even though the keyword wouldn't match this description.
    assert _category_member(rule, "fndds", "Yogurt, regular", text="Cultured dairy blend")
    # Matches via keyword even though the category wouldn't match.
    assert _category_member(rule, "fndds", "Other", text="Yogurt tube")


def test_exclude_keyword_overrides_category_match():
    rule = {"fndds": {"keywords": [], "categories": ["Ice cream and frozen dairy desserts"], "exclude_keywords": ["frozen yogurt"]}}
    assert not _category_member(rule, "fndds", "Ice cream and frozen dairy desserts", text="Frozen yogurt, chocolate")


def test_no_category_col_falls_back_to_false_when_no_keywords():
    rule = {"fndds": {"keywords": [], "categories": ["Yogurt, regular"]}}
    assert fndds_predicate_sql(rule, category_col=None) == "FALSE"


# -- hard stopword sanitization (backend-agnostic, applies to any proposed rule) ------


def test_never_useful_keyword_stripped_from_predicate():
    # Verified real case: LLM_DEVICE=ollama/qwen3:1.7b proposed "with" as an FNDDS
    # keyword for the yogurt block, matching "chicken with gravy", "pasta with sauce",
    # etc. -- this must never reach the generated SQL regardless of who proposed it.
    rule = {"fndds": {"keywords": ["yogurt", "with"]}}
    predicate = fndds_predicate_sql(rule, category_col=None)
    assert "with" not in predicate
    assert "yogurt" in predicate


def test_never_useful_keyword_alone_yields_false():
    rule = {"fndds": {"keywords": ["with", "the", "and"]}}
    assert fndds_predicate_sql(rule, category_col=None) == "FALSE"


def test_never_useful_keyword_does_not_falsely_match_unrelated_text():
    rule = {"fndds": {"keywords": ["yogurt", "with"]}}
    assert not _member(rule, "fndds", "Chicken with gravy")


def test_off_side_also_sanitizes_never_useful_keywords():
    rule = {"off": {"keywords": ["yogurt", "with"]}}
    predicate = off_predicate_sql(rule, category_col=None)
    assert "with" not in predicate

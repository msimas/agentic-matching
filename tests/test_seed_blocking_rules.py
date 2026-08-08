import duckdb

from agentic_matching.blocking.rules import fndds_predicate_sql, off_predicate_sql

YOGURT_RULE = {
    "fndds": {"keywords": ["yogurt", "yoghurt"], "exclude_keywords": ["frozen yogurt"]},
    "off": {"keywords": ["yogurt", "yoghurt", "yogourt"], "exclude_keywords": []},
}


def _member(rule: dict, side: str, text: str) -> bool:
    con = duckdb.connect()
    text_col = "fndds_search_text" if side == "fndds" else "search_text"
    predicate = fndds_predicate_sql(rule) if side == "fndds" else off_predicate_sql(rule)
    row = con.execute(
        f"SELECT {predicate} FROM (SELECT lower(?) AS {text_col})", [text]
    ).fetchone()
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

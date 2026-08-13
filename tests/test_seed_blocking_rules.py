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


def test_category_and_keyword_both_required():
    # POC change: keywords and categories are AND'd, not OR'd, when both are given
    # (see rules.py's module docstring for why) -- a record needs both to be in-block.
    rule = {"fndds": {"keywords": ["yogurt"], "categories": ["Yogurt, regular"]}}
    # Category matches but keyword doesn't -> no longer sufficient on its own.
    assert not _category_member(rule, "fndds", "Yogurt, regular", text="Cultured dairy blend")
    # Keyword matches but category doesn't -> no longer sufficient on its own.
    assert not _category_member(rule, "fndds", "Other", text="Yogurt tube")
    # Both match -> in-block.
    assert _category_member(rule, "fndds", "Yogurt, regular", text="Yogurt tube")


def test_missing_category_value_excludes_despite_keyword_match():
    # The concrete motivating case: a record with NO category value at all (NULL,
    # common on the OFF side -- verified ~60% of the OFF catalog overall, see
    # README's "Known constraints") can never satisfy the category clause, so once a
    # rule specifies categories, such a record is excluded even if its text is an
    # unambiguous keyword match.
    rule = {"off": {"keywords": ["bean"], "categories": ["en:canned-common-beans"]}}
    assert not _category_member(rule, "off", None, text="Cannellini beans")


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


# -- case-insensitivity is guaranteed by the predicate itself, not the caller --------
#
# Verified real bug: blocking/agent_loop.py's materialize_block (and metrics.py's
# block_sizes) both passed the RAW `description` column, never pre-lowered by the
# caller -- DuckDB's LIKE is case-sensitive by default, so a lowercase keyword like
# "fried mushrooms" against the real, Title-Case "Fried mushrooms" matched ZERO rows.
# This silently broke the `breaded_vegetables` block (short, keyword-first FNDDS
# descriptions -- the keyword needed to match starts capitalized) while going
# unnoticed on blocks like `beans`, where a generic keyword often happens to already
# appear lowercase mid-description ("Black beans"). These tests query genuinely
# UN-lowered text -- unlike `_member`/`_category_member` above, which always
# pre-lowered their fake data and so could never have caught this.


def _raw_member(rule: dict, side: str, text: str) -> bool:
    con = duckdb.connect()
    text_col = "description" if side == "fndds" else "search_text"
    predicate = (
        fndds_predicate_sql(rule, category_col=None) if side == "fndds" else off_predicate_sql(rule, category_col=None)
    )
    row = con.execute(f"SELECT {predicate} FROM (SELECT ? AS {text_col})", [text]).fetchone()  # NOT lower()'d
    con.close()
    return bool(row[0])


def test_fndds_keyword_matches_title_case_text_without_caller_lowering():
    rule = {"fndds": {"keywords": ["fried mushrooms"]}}
    assert _raw_member(rule, "fndds", "Fried mushrooms")


def test_fndds_keyword_matches_when_match_is_at_the_very_start():
    # The exact shape of the real breaded_vegetables failure: the keyword phrase is
    # also the first word(s) of the description, so it's capitalized by normal
    # sentence-case convention -- the case a mid-string match (like "bean" in "Black
    # beans") could accidentally dodge even under the old, broken behavior.
    rule = {"fndds": {"keywords": ["sweet potato fries"]}}
    assert _raw_member(rule, "fndds", "Sweet potato fries, NFS")


def test_off_keyword_matches_mixed_case_text_without_caller_lowering():
    rule = {"off": {"keywords": ["yogurt"]}}
    assert _raw_member(rule, "off", "Chobani Greek Yogurt, Plain")


def test_exclude_keyword_matches_mixed_case_text_without_caller_lowering():
    rule = {"fndds": {"keywords": ["yogurt"], "exclude_keywords": ["Frozen Yogurt"]}}
    assert not _raw_member(rule, "fndds", "Frozen Yogurt, Chocolate")

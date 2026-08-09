"""A blocking rule is a keyword-inclusion / keyword-exclusion predicate over a side's
text, optionally OR'd with a structured-category membership predicate (see
llm/prompts.py's blocking schema). This module turns a rule dict into SQL predicates
and applies it against FNDDS / OFF.

Free-text keyword matching alone is fragile: FNDDS's `additional_description` field is
full of boilerplate variant-annotations ("all flavors", "multigrain, whole grain, whole
wheat") shared across many unrelated food categories, and even a keyword mined from
clean description text (e.g. "fruit", "whole", "plain") can independently occur in many
other foods' own descriptions too (fruit salad, whole wheat muffins, plain pretzels).
Both FNDDS (WWEIA food category) and OFF (categories_tags) carry clean, human-curated
categorical labels that don't have this problem -- a categorical membership check
(exact match / tag containment) is far more precise than a keyword guess when a
relevant category exists, so a rule can specify "categories" as an alternative
membership signal, OR'd with the keyword predicate.
"""

from __future__ import annotations

from typing import Any, Literal

# Plain function/connector words -- these can never legitimately define ANY block on
# their own (there's no product category for which "with" or "the" is a meaningful
# signal), so they're stripped from every proposed rule's keywords unconditionally,
# regardless of who proposed the rule. This is a hard backend-agnostic safety net, not
# a suggestion: verified case (real LLM, LLM_DEVICE=ollama, qwen3:1.7b, yogurt block) --
# a revision added "with" as an FNDDS/OFF keyword (matching "chicken with gravy",
# "pasta with sauce", ...), growing the FNDDS block from 148 to 1,282 records while the
# calibration-proxy metric it's scored against happened to read as a meaningful
# improvement (recovering more Branded<->OFF gold pairs), giving the loop no signal to
# reject it -- unlike a genuinely-risky-but-sometimes-legitimate word (e.g. "plain",
# which IS a real qualifier for some products), a bare connector word is never worth
# even entertaining, so this doesn't have the whack-a-mole problem a broader stopword
# list would (see llm/mock.py's history of trying and abandoning that for keyword
# *mining* specifically -- this is deliberately a much smaller, uncontroversial set).
NEVER_USEFUL_KEYWORDS = {
    "the", "and", "or", "of", "with", "a", "an", "in", "to", "for", "as", "is", "on",
    "not", "no", "ns", "nfs", "type", "milk", "from", "added", "fat", "other", "than",
}


def _side_predicate_sql(
    side_rule: dict[str, Any],
    text_col: str,
    category_col: str | None = None,
    category_kind: Literal["exact", "array_contains"] = "exact",
) -> str:
    keywords = [
        k.lower().replace("'", "''")
        for k in side_rule.get("keywords", [])
        if k and k.strip().lower() not in NEVER_USEFUL_KEYWORDS
    ]
    excludes = [k.lower().replace("'", "''") for k in side_rule.get("exclude_keywords", []) if k]
    categories = [c.replace("'", "''") for c in side_rule.get("categories", []) if c]

    clauses = []
    if keywords:
        include = " OR ".join(f"{text_col} LIKE '%{kw}%'" for kw in keywords)
        clauses.append(f"({include})")
    if categories and category_col:
        if category_kind == "array_contains":
            cat_sql = " OR ".join(f"list_contains({category_col}, '{c}')" for c in categories)
        else:
            cat_sql = " OR ".join(f"lower({category_col}) = '{c.lower()}'" for c in categories)
        clauses.append(f"({cat_sql})")
    if not clauses:
        return "FALSE"

    sql = "(" + " OR ".join(clauses) + ")"
    if excludes:
        exclude = " OR ".join(f"{text_col} LIKE '%{kw}%'" for kw in excludes)
        sql = f"({sql} AND NOT ({exclude}))"
    return sql


def fndds_predicate_sql(
    rule: dict[str, Any],
    text_col: str = "description",
    category_col: str | None = "wweia_food_category_description",
) -> str:
    return _side_predicate_sql(rule.get("fndds", {}), text_col, category_col=category_col, category_kind="exact")


def off_predicate_sql(
    rule: dict[str, Any],
    text_col: str = "search_text",
    category_col: str | None = "categories_tags",
) -> str:
    return _side_predicate_sql(
        rule.get("off", {}), text_col, category_col=category_col, category_kind="array_contains"
    )

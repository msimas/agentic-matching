"""A blocking rule is a simple keyword-inclusion / keyword-exclusion predicate over a
side's search text (see llm/prompts.py's blocking schema). This module turns a rule
dict into SQL predicates and applies it against FNDDS / OFF.
"""

from __future__ import annotations

from typing import Any


def _side_predicate_sql(side_rule: dict[str, Any], text_col: str) -> str:
    keywords = [k.lower().replace("'", "''") for k in side_rule.get("keywords", []) if k]
    excludes = [k.lower().replace("'", "''") for k in side_rule.get("exclude_keywords", []) if k]
    if not keywords:
        return "FALSE"
    include = " OR ".join(f"{text_col} LIKE '%{kw}%'" for kw in keywords)
    sql = f"({include})"
    if excludes:
        exclude = " OR ".join(f"{text_col} LIKE '%{kw}%'" for kw in excludes)
        sql = f"({sql} AND NOT ({exclude}))"
    return sql


def fndds_predicate_sql(rule: dict[str, Any], text_col: str = "fndds_search_text") -> str:
    return _side_predicate_sql(rule.get("fndds", {}), text_col)


def off_predicate_sql(rule: dict[str, Any], text_col: str = "search_text") -> str:
    return _side_predicate_sql(rule.get("off", {}), text_col)

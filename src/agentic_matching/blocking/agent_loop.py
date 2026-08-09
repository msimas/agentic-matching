"""Bounded (N=3 by default) agentic loop for blocking-rule construction: the LLM
proposes a rule, it's scored against the calibration proxy (pair completeness,
reduction ratio) and block-size diagnostics, and the LLM revises it — stopping early if
metrics stabilize. Each round's rule + metrics is written to data/artifacts/ for SME
review; the final round also materializes the FNDDS/OFF block subsets used downstream.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import duckdb

from agentic_matching import profiling
from agentic_matching.blocking.metrics import evaluate_rule
from agentic_matching.blocking.rules import fndds_predicate_sql, off_predicate_sql
from agentic_matching.blocking.seed_rules import get_seed_rule
from agentic_matching.config import (
    ARTIFACTS_DIR,
    BLOCKS_DIR,
    FDC_DUCKDB_PATH,
    OFF_PARQUET,
    OFF_SEARCH_TEXT_PARQUET,
    agent_loop_settings,
)
from agentic_matching.llm.client import ChatClient, get_llm_client
from agentic_matching.llm.prompts import build_blocking_prompt

log = logging.getLogger(__name__)


@dataclass
class BlockingRound:
    round: int
    rule: dict[str, Any]
    metrics: dict[str, Any]
    rationale: str


def _sample_texts(con: duckdb.DuckDBPyConnection, block_name: str, n: int = 40) -> tuple[list[str], list[str]]:
    """Seed samples for round 0: records that already plausibly mention the block's
    canonical term, so the LLM has real in-block examples to mine keywords from."""
    from agentic_matching.blocking.metrics import term_predicate_sql

    # ORDER BY makes the sample (and therefore whatever keywords get mined from it,
    # whether by the real LLM or mock.py's frequency heuristic) reproducible across
    # runs -- LIMIT alone leaves row order, and hence the sample, up to DuckDB's query
    # plan, which was observed to change which keywords got proposed (and swing block
    # size by 2-3x) between two runs of otherwise-identical code.
    fndds_pred = term_predicate_sql("lower(description)", block_name)
    fndds_rows = con.execute(
        f"SELECT description FROM v_fndds WHERE {fndds_pred} ORDER BY fdc_id LIMIT {n}"
    ).fetchall()
    off_path = str(OFF_SEARCH_TEXT_PARQUET).replace("'", "''")
    off_pred = term_predicate_sql("lower(coalesce(product_name, ''))", block_name)
    off_rows = con.execute(
        f"""
        SELECT product_name FROM read_parquet('{off_path}')
        WHERE {off_pred} ORDER BY code LIMIT {n}
        """
    ).fetchall()
    return [r[0] for r in fndds_rows if r[0]], [r[0] for r in off_rows if r[0]]


def _category_options(
    con: duckdb.DuckDBPyConnection, block_name: str, k: int = 15, min_specificity: float = 0.2
) -> dict[str, list[dict[str, Any]]]:
    """Real category values seen among records already plausibly in this block (same
    seed-term-filtered population `_sample_texts` draws from), with their counts --
    lets the LLM (or mock.py) propose a *structured* category-membership predicate
    (see rules.py) instead of relying solely on free-text keywords, which is far more
    precise when a clean matching category exists (e.g. FNDDS's WWEIA "Yogurt, regular"
    / "Yogurt, Greek" categories, OFF's "en:yogurts" tag).

    Filters out categories below `min_specificity` -- the fraction of the category's
    *entire catalog-wide* membership that also plausibly belongs to this block (matches
    the seed term). A broad umbrella category (e.g. OFF's "en:dairies", which also
    covers cheese/milk/butter/cream) can have a large count *within* the term-filtered
    sample without being remotely block-specific, simply because it has enormous
    catalog-wide membership -- absolute/within-sample count alone can't tell a
    genuinely block-specific category (most of whose members are relevant) apart from a
    broad one (a small fraction of whose members happen to be relevant). Verified real
    case (LLM_DEVICE=ollama, yogurt block): "en:dairies" (175,806 catalog-wide members,
    specificity 0.097) inflated the OFF block to 194,722 records, 80% false positives,
    right alongside "en:yogurts" (36,121 members, specificity 0.449) and
    "en:greek-style-yogurts" (4,107 members, specificity 0.655) -- both of which stay
    well clear of the 0.2 cutoff, so this doesn't just reduce to "reject big
    categories."""
    from agentic_matching.blocking.metrics import term_predicate_sql

    fndds_term_pred = term_predicate_sql("lower(description)", block_name)
    fndds_rows = con.execute(
        f"""
        WITH term_matched AS (
            SELECT wweia_food_category_description AS cat, count(*) AS matched
            FROM v_fndds
            WHERE {fndds_term_pred} AND wweia_food_category_description IS NOT NULL
            GROUP BY 1
        ),
        catalog_wide AS (
            SELECT wweia_food_category_description AS cat, count(*) AS total
            FROM v_fndds
            WHERE wweia_food_category_description IS NOT NULL
            GROUP BY 1
        )
        SELECT tm.cat, tm.matched
        FROM term_matched tm JOIN catalog_wide cw ON cw.cat = tm.cat
        WHERE (tm.matched::DOUBLE / cw.total) >= {min_specificity}
        ORDER BY tm.matched DESC LIMIT {k}
        """
    ).fetchall()
    off_path = str(OFF_SEARCH_TEXT_PARQUET).replace("'", "''")
    off_raw_path = str(OFF_PARQUET).replace("'", "''")
    off_term_pred = term_predicate_sql("lower(coalesce(product_name, ''))", block_name)
    off_rows = con.execute(
        f"""
        WITH term_matched AS (
            SELECT tag, count(*) AS matched FROM (
                SELECT unnest(categories_tags) AS tag
                FROM read_parquet('{off_path}')
                WHERE {off_term_pred}
            )
            WHERE tag IS NOT NULL AND tag != 'en:null'
            GROUP BY 1
        ),
        catalog_wide AS (
            SELECT tag, count(*) AS total FROM (
                SELECT unnest(categories_tags) AS tag FROM read_parquet('{off_raw_path}')
            )
            WHERE tag IS NOT NULL AND tag != 'en:null'
            GROUP BY 1
        )
        SELECT tm.tag, tm.matched
        FROM term_matched tm JOIN catalog_wide cw ON cw.tag = tm.tag
        WHERE (tm.matched::DOUBLE / cw.total) >= {min_specificity}
        ORDER BY tm.matched DESC LIMIT {k}
        """
    ).fetchall()
    return {
        "fndds": [{"value": v, "count": c} for v, c in fndds_rows],
        "off": [{"value": v, "count": c} for v, c in off_rows],
    }


def _stabilized(prev: dict[str, Any] | None, cur: dict[str, Any]) -> bool:
    if prev is None:
        return False
    delta = agent_loop_settings.stabilization_delta
    return (
        abs(prev["pair_completeness"] - cur["pair_completeness"]) < delta
        and abs(prev["reduction_ratio"] - cur["reduction_ratio"]) < delta
    )


def _select_final_rule(rounds: list[BlockingRound]) -> dict[str, Any]:
    """Pick which round's rule to materialize as the block's final definition.

    If the loop stopped because the *last* round's metrics were negligibly different
    from the round before it (per `_stabilized`) -- rather than because it ran out of
    rounds -- prefer the earlier, more conservative rule. A revision that only moves
    the calibration-proxy metrics by a hair isn't worth whatever precision risk it
    introduces, and the proxy metric (recall against the Branded<->OFF calibration
    pairs) has no way to see that risk at all: verified case (real LLM, LLM_DEVICE=ollama,
    yogurt block) -- a revision added "plain" as an FNDDS keyword, moving
    pair_completeness by only +0.007 (well under the default 0.01 stabilization delta,
    so the loop stopped) while pulling in 69 new false positives (muffins, waffles,
    chicken wings, oatmeal, potato chips -- none of which are yogurt), because "plain"
    is also a common qualifier for countless unrelated foods. Taking the round *before*
    a change that small avoids adopting that kind of regression, at the cost of
    forgoing genuinely-small-but-real improvements too -- an acceptable trade since
    "small" here is explicitly the range the metric can't distinguish from noise.

    If the loop instead ran through every round without ever stabilizing, there's no
    "negligible change" signal to act on, so the last (most-recently-revised) round's
    rule is used, as before.
    """
    if len(rounds) >= 2 and _stabilized(rounds[-2].metrics, rounds[-1].metrics):
        return rounds[-2].rule
    return rounds[-1].rule


def materialize_block(con: duckdb.DuckDBPyConnection, block_name: str, rule: dict[str, Any]) -> dict[str, int]:
    """Write the FNDDS and OFF record subsets passing the final rule to data/blocks/."""
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    # Matched against the raw `description` column (qualified -- both joined tables
    # have one), not the fndds_search_text blob that also folds in WWEIA category /
    # additional_description text: that field is full of boilerplate variant-annotations
    # ("all flavors", "multigrain, whole grain, whole wheat") shared across many
    # unrelated food categories, which made keywords mined from real descriptions (e.g.
    # "fruit", "whole") match wildly unrelated records (chicken, pasta, cookies, ...)
    # when tested against the concatenated blob. See rules.py's module docstring.
    fndds_pred = fndds_predicate_sql(rule, text_col="f.description", category_col="f.wweia_food_category_description")
    off_pred = off_predicate_sql(rule, text_col="search_text")

    fndds_out = str(BLOCKS_DIR / f"{block_name}_fndds.parquet").replace("'", "''")
    off_out = str(BLOCKS_DIR / f"{block_name}_off.parquet").replace("'", "''")

    con.execute(
        f"""
        COPY (
            SELECT f.*, s.fndds_search_text
            FROM v_fndds f
            JOIN fndds_search s ON s.fdc_id = f.fdc_id
            WHERE {fndds_pred}
        ) TO '{fndds_out}' (FORMAT PARQUET)
        """
    )
    con.execute(f"COPY (SELECT * FROM off_search WHERE {off_pred}) TO '{off_out}' (FORMAT PARQUET)")

    n_fndds = con.execute(f"SELECT count(*) FROM read_parquet('{fndds_out}')").fetchone()[0]
    n_off = con.execute(f"SELECT count(*) FROM read_parquet('{off_out}')").fetchone()[0]
    log.info("Materialized block '%s': %d FNDDS records, %d OFF records", block_name, n_fndds, n_off)
    return {"n_fndds": n_fndds, "n_off": n_off}


def run_blocking_agent(block_name: str, client: ChatClient | None = None) -> list[BlockingRound]:
    client = client or get_llm_client()
    profiling.build(force=False)  # no-op if already built (see scripts/03_build_fdc_db.py)
    con = duckdb.connect(str(FDC_DUCKDB_PATH))
    off_path = str(OFF_SEARCH_TEXT_PARQUET).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW off_search AS SELECT * FROM read_parquet('{off_path}')")
    con.execute(
        """
        CREATE OR REPLACE VIEW fndds_search AS
        SELECT fdc_id, description, wweia_food_category_description, additional_description,
               lower(coalesce(description, '') || ' ' ||
                     coalesce(wweia_food_category_description, '') || ' ' ||
                     coalesce(additional_description, '')) AS fndds_search_text
        FROM v_fndds
        """
    )

    fndds_samples, off_samples = _sample_texts(con, block_name)
    category_options = _category_options(con, block_name)
    corpus_stats = {
        # FNDDS is only ~5.4K rows total -- informational top-k, not a hard genericness
        # bar (see profiling.OFF_GENERIC_TERM_MIN_DOC_COUNT's docstring for why).
        "fndds": {"n_records": profiling.n_records("fndds"), "catalog_wide_common_terms": profiling.high_frequency_terms("fndds")},
        # OFF is ~4.66M rows -- an absolute doc-count floor, not a fixed top-k, so a
        # term like "green"/"black" (broad, but not quite top-40 by rank) isn't missed.
        "off": {
            "n_records": profiling.n_records("off"),
            "catalog_wide_common_terms": profiling.high_frequency_terms(
                "off", min_doc_count=profiling.OFF_GENERIC_TERM_MIN_DOC_COUNT
            ),
        },
    }

    rounds: list[BlockingRound] = []
    # None for most blocks (fully LLM-proposed, as originally designed); a hand-curated
    # starting point for a block with an entry in seed_rules.py (e.g. exclude_keywords
    # already known to be needed) -- see that module's docstring.
    prev_rule: dict[str, Any] | None = get_seed_rule(block_name)
    prev_metrics: dict[str, Any] | None = None

    for round_idx in range(agent_loop_settings.max_rounds):
        sys_p, user_p = build_blocking_prompt(
            block_name,
            fndds_samples,
            off_samples,
            previous_rule=prev_rule,
            metrics=prev_metrics,
            corpus_stats=corpus_stats,
            category_options=category_options,
        )
        try:
            response = client.complete_json(sys_p, user_p)
            rule = {"fndds": response["fndds"], "off": response["off"]}
        except Exception:
            # A real LLM backend can fail a round outright (e.g. a reasoning model
            # exhausting its token budget mid-thought and never emitting valid JSON --
            # observed against Ollama/qwen3 on this project's own blocking prompt).
            # Rounds already completed (and already materialized as artifacts on disk)
            # are still useful; don't let one bad round throw all of that away. Only a
            # genuine problem if round 0 itself fails, since there's nothing to fall
            # back to yet -- that still propagates.
            if not rounds:
                raise
            log.exception(
                "Round %d failed for block '%s'; stopping here and using the best "
                "round completed so far instead of retrying indefinitely.",
                round_idx,
                block_name,
            )
            break
        rationale = response.get("rationale", "")

        metrics = evaluate_rule(block_name, rule)
        rounds.append(BlockingRound(round=round_idx, rule=rule, metrics=metrics, rationale=rationale))
        _write_artifact(block_name, rounds[-1])

        if _stabilized(prev_metrics, metrics):
            log.info("Blocking loop for '%s' stabilized after round %d", block_name, round_idx)
            break
        prev_rule, prev_metrics = rule, metrics

    final_rule = _select_final_rule(rounds)
    if final_rule is not rounds[-1].rule:
        log.info(
            "Round %d's change was within the stabilization delta; keeping round %d's "
            "rule instead (see _select_final_rule's docstring)",
            rounds[-1].round,
            rounds[-2].round,
        )
    materialize_block(con, block_name, final_rule)
    con.close()
    return rounds


def _write_artifact(block_name: str, r: BlockingRound) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"blocking_{block_name}_round{r.round}.json"
    path.write_text(json.dumps(asdict(r), indent=2))
    log.info("Wrote %s", path)

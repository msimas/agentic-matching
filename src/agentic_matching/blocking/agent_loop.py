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

from agentic_matching.blocking.metrics import evaluate_rule
from agentic_matching.blocking.rules import fndds_predicate_sql, off_predicate_sql
from agentic_matching.config import (
    ARTIFACTS_DIR,
    BLOCKS_DIR,
    FDC_DUCKDB_PATH,
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
    from agentic_matching.blocking.metrics import CANONICAL_BLOCK_TERMS

    # ORDER BY makes the sample (and therefore whatever keywords get mined from it,
    # whether by the real LLM or mock.py's frequency heuristic) reproducible across
    # runs -- LIMIT alone leaves row order, and hence the sample, up to DuckDB's query
    # plan, which was observed to change which keywords got proposed (and swing block
    # size by 2-3x) between two runs of otherwise-identical code.
    term = CANONICAL_BLOCK_TERMS[block_name].replace("'", "''")
    fndds_rows = con.execute(
        f"SELECT description FROM v_fndds WHERE lower(description) LIKE '%{term}%' ORDER BY fdc_id LIMIT {n}"
    ).fetchall()
    off_path = str(OFF_SEARCH_TEXT_PARQUET).replace("'", "''")
    off_rows = con.execute(
        f"""
        SELECT product_name FROM read_parquet('{off_path}')
        WHERE lower(coalesce(product_name, '')) LIKE '%{term}%' ORDER BY code LIMIT {n}
        """
    ).fetchall()
    return [r[0] for r in fndds_rows if r[0]], [r[0] for r in off_rows if r[0]]


def _stabilized(prev: dict[str, Any] | None, cur: dict[str, Any]) -> bool:
    if prev is None:
        return False
    delta = agent_loop_settings.stabilization_delta
    return (
        abs(prev["pair_completeness"] - cur["pair_completeness"]) < delta
        and abs(prev["reduction_ratio"] - cur["reduction_ratio"]) < delta
    )


def materialize_block(con: duckdb.DuckDBPyConnection, block_name: str, rule: dict[str, Any]) -> dict[str, int]:
    """Write the FNDDS and OFF record subsets passing the final rule to data/blocks/."""
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    fndds_pred = fndds_predicate_sql(rule, text_col="fndds_search_text")
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

    rounds: list[BlockingRound] = []
    prev_rule: dict[str, Any] | None = None
    prev_metrics: dict[str, Any] | None = None

    for round_idx in range(agent_loop_settings.max_rounds):
        sys_p, user_p = build_blocking_prompt(
            block_name, fndds_samples, off_samples, previous_rule=prev_rule, metrics=prev_metrics
        )
        response = client.complete_json(sys_p, user_p)
        rule = {"fndds": response["fndds"], "off": response["off"]}
        rationale = response.get("rationale", "")

        metrics = evaluate_rule(block_name, rule)
        rounds.append(BlockingRound(round=round_idx, rule=rule, metrics=metrics, rationale=rationale))
        _write_artifact(block_name, rounds[-1])

        if _stabilized(prev_metrics, metrics):
            log.info("Blocking loop for '%s' stabilized after round %d", block_name, round_idx)
            break
        prev_rule, prev_metrics = rule, metrics

    materialize_block(con, block_name, rounds[-1].rule)
    con.close()
    return rounds


def _write_artifact(block_name: str, r: BlockingRound) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"blocking_{block_name}_round{r.round}.json"
    path.write_text(json.dumps(asdict(r), indent=2))
    log.info("Wrote %s", path)

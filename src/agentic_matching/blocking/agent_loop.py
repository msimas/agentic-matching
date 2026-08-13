"""Bounded (N=3 by default) agentic loop for blocking-rule construction: the LLM
proposes a rule, it's scored against the calibration proxy (pair completeness,
reduction ratio) and block-size diagnostics, and the LLM revises it — stopping early if
metrics stabilize. Each round's rule + metrics is written to data/artifacts/ for SME
review; the final round also materializes the FNDDS/OFF block subsets used downstream.

`run_blocking_agent`'s optional `linking_feedback` parameter is how outer_loop.py closes
the loop from a *later* stage back to this one: a finding from a previous full run
through attribute-generation and linking (see outer_loop.diagnose_blocking_problem) that
specifically implicated the blocking rule, not just the attributes, echoed to the LLM on
every round the same way `notes` is. A plain call (no outer loop involved) leaves it
None, unchanged from before.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from agentic_matching import profiling
from agentic_matching.blocking.metrics import evaluate_rule
from agentic_matching.blocking.rules import fndds_predicate_sql, off_predicate_sql
from agentic_matching.blocking.seed_rules import get_seed_notes, get_seed_rule
from agentic_matching.config import (
    BLOCKS_DIR,
    FDC_DUCKDB_PATH,
    OFF_PARQUET,
    OFF_SEARCH_TEXT_PARQUET,
    agent_loop_settings,
    new_run_id,
    round_temperature,
    run_artifacts_dir,
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
    con: duckdb.DuckDBPyConnection, block_name: str, k: int = 25, min_specificity: float = 0.2
) -> dict[str, list[dict[str, Any]]]:
    """Real category values seen among records already plausibly in this block (same
    seed-term-filtered population `_sample_texts` draws from), with their counts --
    lets the LLM (or mock.py) propose a *structured* category-membership predicate
    (see rules.py) instead of relying solely on free-text keywords, which is far more
    precise when a clean matching category exists (e.g. FNDDS's WWEIA "Yogurt, regular"
    / "Yogurt, Greek" categories, OFF's "en:yogurts" tag).

    `k` was 15; bumped to 25 after keywords/categories became AND'd rather than OR'd
    (see rules.py's module docstring) -- a low-volume-but-specific category can matter
    more now (missing it drops every record tagged with only that category, not just
    narrows an alternative match path), and a real case was cut off purely by the old
    limit: verified against 'beans', `en:cannellini-beans` (specificity 0.7, well
    above min_specificity) ranked 16th by within-block count and was invisible to the
    LLM under k=15.

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


# Floors for select_best_blocking_round, below. Grounded in this project's own real
# blocking metrics (see blocking_<block>_round0.json artifacts): reduction_ratio
# clusters at 0.99996-0.9999999 across every real block observed (FNDDS ~5.4K x OFF
# ~4.66M means even "0.99" still allows up to ~253M candidate pairs -- deliberately
# generous headroom above real values, and below this project's own test suite's
# accepted-as-fine synthetic value of 0.998 (see test_blocking_round_selection.py) --
# only meant to catch a genuine order-of-magnitude blowup, not fine-grained
# differences between otherwise-healthy rounds. pair_completeness ranged 0.0-0.85
# across real/test blocks -- 0.05 catches "recovered essentially none of the
# calibration pairs" without penalizing a modest-but-real score on a small/sparse
# block.
MIN_REDUCTION_RATIO = 0.99
MIN_PAIR_COMPLETENESS = 0.05


def select_best_blocking_round(rounds: list[BlockingRound]) -> BlockingRound:
    """Pick the best-scoring round across the WHOLE history -- not just a comparison
    between the last two rounds (that's `_stabilized`/`_select_final_rule`'s narrower,
    different job: catching a small metric move that hides a real regression). Mirrors
    linking.agent_loop.select_best_round's structure and rationale: a later round can
    be a genuine regression on the actual metrics without ever being "negligibly
    different" from the round right before it -- a big, non-negligible change can
    still be a bad one -- and nothing previously protected against that; the loop
    would just ship whatever `rounds[-1]` happened to be.

    Scored as (usable pair_completeness, usable reduction_ratio, balanced score) in
    that priority order:
      - A rule that recovers next to none of the calibration pairs is unusable
        regardless of how clean reduction_ratio looks (mirrors linking's "zero
        confident real-world matches" disqualifier in its own select_best_round) --
        floored at MIN_PAIR_COMPLETENESS.
      - A rule so broad it barely reduces the candidate pool overwhelms downstream
        splink training regardless of recall -- floored at MIN_REDUCTION_RATIO.
      - Among rounds clearing both floors, maximize the harmonic mean of the two.
        pair_completeness and reduction_ratio pull in different directions (recall vs.
        selectivity), the same way linking's own precision/recall do, so neither
        should be optimized alone -- f1-style balance, not a simple average, punishes
        a round that's great on one and poor on the other rather than letting one
        strong number hide a weak one.
    """

    def _score(r: BlockingRound) -> tuple[bool, bool, float]:
        pc = r.metrics.get("pair_completeness") or 0.0
        rr = r.metrics.get("reduction_ratio") or 0.0
        balance = (2 * pc * rr / (pc + rr)) if (pc + rr) > 0 else 0.0
        return (pc >= MIN_PAIR_COMPLETENESS, rr >= MIN_REDUCTION_RATIO, balance)

    return max(rounds, key=_score)


def _select_final_rule(rounds: list[BlockingRound]) -> dict[str, Any]:
    """Pick which round's rule to materialize as the block's final definition.

    If the loop stopped because the *last* round's metrics were negligibly different
    from the round before it (per `_stabilized`) -- rather than because it ran out of
    rounds -- that last round isn't trusted on its own merits: a revision that only
    moves the calibration-proxy metrics by a hair isn't worth whatever precision risk
    it introduces, and the proxy metric (recall against the Branded<->OFF calibration
    pairs) has no way to see that risk at all. Verified case (real LLM,
    LLM_DEVICE=ollama, yogurt block): a revision added "plain" as an FNDDS keyword,
    moving pair_completeness by only +0.007 (well under the default 0.01 stabilization
    delta, so the loop stopped) while pulling in 69 new false positives (muffins,
    waffles, chicken wings, oatmeal, potato chips -- none of which are yogurt), because
    "plain" is also a common qualifier for countless unrelated foods. The last round is
    excluded from consideration entirely in this case -- not blindly replaced with
    "the round right before it": an earlier round further back could still be the
    genuinely best one on the merits, and select_best_blocking_round (not a fixed
    index) is what finds it.

    Otherwise (the loop ran through every round without ever stabilizing, or stopped
    for some other reason), every round is a fair candidate and
    select_best_blocking_round picks among all of them -- this is what actually
    protects against a later round being a real, non-negligible regression, which
    nothing used to check at all (the old version of this function just took
    `rounds[-1]` unconditionally in this case).
    """
    candidates = rounds
    if len(rounds) >= 2 and _stabilized(rounds[-2].metrics, rounds[-1].metrics):
        candidates = rounds[:-1]
    return select_best_blocking_round(candidates).rule


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


def run_blocking_agent(
    block_name: str,
    client: ChatClient | None = None,
    linking_feedback: str | None = None,
    run_dir: Path | None = None,
) -> list[BlockingRound]:
    """`run_dir` -- where this run's artifacts (blocking_round<N>.json) are written;
    data/artifacts/<block_name>/<run_id>/. Defaults to a fresh one (own new_run_id())
    for a standalone call (e.g. scripts/05_run_blocking_agent.py); outer_loop.py passes
    the SAME run_dir it's using for attributes/linking too, so one outer-loop
    invocation's artifacts all land in one run folder together."""
    run_dir = run_dir or run_artifacts_dir(block_name, new_run_id())
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
    # starting point for a block with an entry in seed_rules.json (e.g. exclude_keywords
    # already known to be needed) -- see seed_rules.py's docstring. Only "fndds"/"off"
    # seed round 0 -- "notes" (below) is echoed every round instead, since it's
    # persistent domain guidance, not rule state the LLM revises round over round. Only
    # the recognized rule keys are kept -- seed_rules.json may also carry a documentation
    # -only "exclude_keywords_rationale" per side that shouldn't be echoed into the
    # prompt as if it were part of the rule schema.
    seed_rule = get_seed_rule(block_name)
    prev_rule: dict[str, Any] | None = (
        {
            side: {k: seed_rule[side][k] for k in ("keywords", "categories", "exclude_keywords") if k in seed_rule[side]}
            for side in ("fndds", "off")
        }
        if seed_rule
        else None
    )
    notes = get_seed_notes(block_name)
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
            notes=notes,
            linking_feedback=linking_feedback,
        )
        temperature = round_temperature(round_idx, has_prior_state=prev_rule is not None)
        log.info(
            "Generating blocking attributes for '%s' round %d: asking the LLM to %s the blocking rule (temperature=%.2f)",
            block_name,
            round_idx,
            "propose" if round_idx == 0 else "revise",
            temperature,
        )
        try:
            response = client.complete_json(sys_p, user_p, temperature=temperature)
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
        # Concise, always-INFO summary of what this round actually selected -- distinct
        # from the full prompt/response dump at DEBUG (see llm/client.py) and from
        # _write_artifact's full-fidelity JSON below: this is the one line to scan when
        # skimming a run's log to see how the rule evolved round over round without
        # opening every artifact file.
        log.info(
            "block '%s' round %d selected blocking rule: "
            "fndds keywords=%s categories=%s exclude_keywords=%s | "
            "off keywords=%s categories=%s exclude_keywords=%s",
            block_name,
            round_idx,
            rule["fndds"].get("keywords"),
            rule["fndds"].get("categories"),
            rule["fndds"].get("exclude_keywords"),
            rule["off"].get("keywords"),
            rule["off"].get("categories"),
            rule["off"].get("exclude_keywords"),
        )
        _write_artifact(run_dir, rounds[-1])

        if _stabilized(prev_metrics, metrics):
            log.info("Blocking loop for '%s' stabilized after round %d", block_name, round_idx)
            break
        prev_rule, prev_metrics = rule, metrics

    final_rule = _select_final_rule(rounds)
    if final_rule is not rounds[-1].rule:
        final_round = next(r for r in rounds if r.rule is final_rule)
        log.info(
            "block '%s': round %d wasn't the best round produced -- using round %d's "
            "rule instead (see _select_final_rule/select_best_blocking_round's "
            "docstrings)",
            block_name,
            rounds[-1].round,
            final_round.round,
        )
    materialize_block(con, block_name, final_rule)
    con.close()
    client.log_usage_summary(label=f"{block_name} blocking, cumulative")
    return rounds


def _write_artifact(run_dir: Path, r: BlockingRound) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"blocking_round{r.round}.json"
    path.write_text(json.dumps(asdict(r), indent=2))
    log.info("Wrote %s", path)

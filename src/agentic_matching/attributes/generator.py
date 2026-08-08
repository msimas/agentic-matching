"""Bounded (N=3 by default) agentic loop for matching-attribute construction: the LLM
proposes an attribute set (seeded from library.SEED_ATTRIBUTES when available, e.g.
yogurt; from scratch otherwise, e.g. beans), it's checked for pairwise correlation on
the block's own population, and the LLM revises based on the correlation flags —
stopping early once no attribute pair is flagged (or the set stops changing). Each
round is logged to data/artifacts/ for SME review, and the final attribute set is
persisted as a versioned JSON artifact under attributes/generated/<block>/ — this is
the "versioned, testable artifact" the extraction logic is defined by (attributes are
keyword-rule based, computed uniformly by attributes/library.py's apply_attribute, so a
declarative versioned JSON plays the role generated Python code would for a richer
rule language).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from agentic_matching.attributes.correlation_check import check_correlations
from agentic_matching.attributes.library import compute_attribute_values, get_seed_attributes
from agentic_matching.config import ARTIFACTS_DIR, BLOCKS_DIR, agent_loop_settings
from agentic_matching.llm.client import ChatClient, get_llm_client
from agentic_matching.llm.prompts import build_attribute_prompt

log = logging.getLogger(__name__)

GENERATED_DIR = Path(__file__).resolve().parent / "generated"


@dataclass
class AttributeRound:
    round: int
    attributes: list[dict[str, Any]]
    correlation_flags: list[dict[str, Any]]
    rationale: str


def _load_block(block_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fndds_path = BLOCKS_DIR / f"{block_name}_fndds.parquet"
    off_path = BLOCKS_DIR / f"{block_name}_off.parquet"
    if not fndds_path.exists() or not off_path.exists():
        raise FileNotFoundError(
            f"Block subsets not found for '{block_name}'. Run the blocking agent "
            "(scripts/05_run_blocking_agent.py) first."
        )
    return pd.read_parquet(fndds_path), pd.read_parquet(off_path)


def _sample_pairs(fndds_df: pd.DataFrame, off_df: pd.DataFrame, n: int = 30) -> list[dict[str, Any]]:
    rng = random.Random(42)
    fndds_sample = fndds_df["description"].dropna().tolist()
    off_sample = off_df["product_name"].dropna().tolist()
    rng.shuffle(fndds_sample)
    rng.shuffle(off_sample)
    k = min(n, len(fndds_sample), len(off_sample)) or 0
    return [{"fndds_description": f, "off_product_name": o} for f, o in zip(fndds_sample[:k], off_sample[:k])]


def _pooled_values(attrs: list[dict[str, Any]], fndds_df: pd.DataFrame, off_df: pd.DataFrame) -> pd.DataFrame:
    fndds_vals = compute_attribute_values(attrs, fndds_df["fndds_search_text"].tolist(), side="fndds")
    off_vals = compute_attribute_values(attrs, off_df["search_text"].tolist(), side="off")
    fndds_frame = pd.DataFrame(fndds_vals)
    off_frame = pd.DataFrame(off_vals)
    return pd.concat([fndds_frame, off_frame], ignore_index=True)


def run_attribute_agent(block_name: str, client: ChatClient | None = None) -> list[AttributeRound]:
    client = client or get_llm_client()
    fndds_df, off_df = _load_block(block_name)
    sample_pairs = _sample_pairs(fndds_df, off_df)

    rounds: list[AttributeRound] = []
    existing = get_seed_attributes(block_name)  # None -> LLM proposes from scratch
    correlation_flags: list[dict[str, Any]] = []
    evaluation = None

    for round_idx in range(agent_loop_settings.max_rounds):
        sys_p, user_p = build_attribute_prompt(
            block_name,
            sample_pairs,
            existing_attributes=existing,
            correlation_flags=correlation_flags or None,
            evaluation=evaluation,
        )
        response = client.complete_json(sys_p, user_p)
        attrs = response["attributes"]
        rationale = response.get("rationale", "")

        values_df = _pooled_values(attrs, fndds_df, off_df)
        correlation_flags = check_correlations(values_df)

        rounds.append(
            AttributeRound(round=round_idx, attributes=attrs, correlation_flags=correlation_flags, rationale=rationale)
        )
        _write_artifact(block_name, rounds[-1])

        names = {a["name"] for a in attrs}
        prev_names = {a["name"] for a in existing} if existing else None
        if not correlation_flags and prev_names == names:
            log.info("Attribute loop for '%s' stabilized after round %d", block_name, round_idx)
            break
        existing = attrs

    _persist_generated(block_name, rounds[-1].attributes, version=len(rounds))
    return rounds


def _write_artifact(block_name: str, r: AttributeRound) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"attributes_{block_name}_round{r.round}.json"
    path.write_text(json.dumps(asdict(r), indent=2))
    log.info(
        "Wrote %s (%d attributes, %d correlation flags)",
        path,
        len(r.attributes),
        len(r.correlation_flags),
    )


def _persist_generated(block_name: str, attrs: list[dict[str, Any]], version: int) -> None:
    out_dir = GENERATED_DIR / block_name
    out_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = out_dir / f"v{version}.json"
    latest_path = out_dir / "latest.json"
    payload = json.dumps(attrs, indent=2)
    versioned_path.write_text(payload)
    latest_path.write_text(payload)
    log.info("Persisted final attribute set for '%s' -> %s", block_name, versioned_path)


def load_latest_attributes(block_name: str) -> list[dict[str, Any]]:
    path = GENERATED_DIR / block_name / "latest.json"
    if not path.exists():
        raise FileNotFoundError(f"No generated attributes for '{block_name}'; run the attribute agent first.")
    return json.loads(path.read_text())

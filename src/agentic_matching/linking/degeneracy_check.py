"""Sanity-check a trained splink model's fitted m/u probabilities for collapsed or
label-switched solutions before its predictions are trusted.

Heuristics (a real SME reviews the full match-weight chart; these catch the clearest
failure modes automatically):
  - label-switching: a "more agreement" comparison level should have a higher
    m_probability (P(that level | match)) than a "less agreement" level below it.
  - collapse: m_probability ~= u_probability for the exact-match level means the
    comparison carries no discriminating power.
  - degenerate global prior: probability_two_random_records_match pinned near 0 or 1.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from splink import Linker

log = logging.getLogger(__name__)

COLLAPSE_THRESHOLD = 0.02


def export_trained_settings(linker: Linker) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "model.json")
        linker.misc.save_model_to_json(path, overwrite=True)
        import json

        return json.loads(Path(path).read_text())


def check_degeneracy(settings: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    p2r = settings.get("probability_two_random_records_match")
    if p2r is not None and (p2r < 1e-4 or p2r > 0.999):
        flags.append(
            {
                "kind": "degenerate_prior",
                "detail": f"probability_two_random_records_match={p2r:.6f} is pinned near 0 or 1",
            }
        )

    for comparison in settings.get("comparisons", []):
        col = comparison["output_column_name"]
        levels = [
            lvl
            for lvl in comparison["comparison_levels"]
            if lvl.get("m_probability") is not None and lvl.get("u_probability") is not None
        ]
        if not levels:
            flags.append({"kind": "untrained", "column": col, "detail": "no trained levels"})
            continue

        # label-switching: levels are listed most-agreement-first; m_probability should
        # be non-increasing as we move down the list (less agreement).
        m_seq = [lvl["m_probability"] for lvl in levels]
        if any(m_seq[i] < m_seq[i + 1] - 1e-9 for i in range(len(m_seq) - 1)):
            flags.append(
                {
                    "kind": "label_switching",
                    "column": col,
                    "detail": f"m_probability not monotonically decreasing across levels: {m_seq}",
                }
            )

        # collapse: top level (most agreement) should clearly separate m from u.
        top = levels[0]
        if abs(top["m_probability"] - top["u_probability"]) < COLLAPSE_THRESHOLD:
            flags.append(
                {
                    "kind": "collapsed",
                    "column": col,
                    "detail": (
                        f"top level m={top['m_probability']:.4f} u={top['u_probability']:.4f} "
                        "-- comparison has little discriminating power"
                    ),
                }
            )

    return flags

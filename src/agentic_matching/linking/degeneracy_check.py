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


# Degeneracy-flag kinds that, when they land on a specific ATTRIBUTE's comparison (as
# opposed to `description`'s), all mean the same practical thing: that attribute isn't
# contributing any usable matching signal in this block. See
# degenerate_attribute_columns's docstring for why each one qualifies, and why the
# same reasoning does NOT extend to `description`.
ATTRIBUTE_DEGENERACY_KINDS = frozenset({"collapsed", "label_switching", "untrained"})


def degenerate_attribute_columns(
    flags: list[dict[str, Any]],
    attr_names: set[str],
    kinds: frozenset[str] = ATTRIBUTE_DEGENERACY_KINDS,
) -> list[str]:
    """Which degeneracy flags are on a specific *attribute*'s comparison, as opposed to
    `description`'s -- see splink_model.build_comparisons: every comparison column is
    named either an attribute's own `name` or the literal "description", so this is an
    exact-membership check, not a heuristic.

    This distinction matters because an attribute-column flag and a `description`-
    column flag call for opposite fixes: a degenerate `description` comparison means
    the candidate pairs from *blocking* had too little real variation for EM to
    estimate string-similarity parameters from -- no attribute change can fix that,
    only a broader/narrower blocking rule can (see outer_loop.diagnose_blocking_
    problem). A degenerate *attribute* comparison means that one derived attribute
    isn't pulling its weight in this block, and the fix is to drop or redefine it, not
    to touch blocking at all.

    All three flag `kinds` mean this for an ATTRIBUTE comparison specifically, because
    every attribute comparison in this codebase is built via cl.ExactMatch (see
    build_comparisons), which always produces exactly two levels ("Exact match" / "All
    other comparisons") regardless of how many categories the attribute has:

      - "collapsed": m(exact) ~= u(exact) -- agreeing on this attribute is no more
        likely among true matches than among random pairs. No signal.
      - "label_switching": with only two levels, this can only mean m(exact) <
        m(not-exact) -- true matches are LESS likely to agree on this attribute than
        to disagree. Actively anti-correlated with matching, not merely uninformative
        -- arguably worse than "collapsed", and structurally guaranteed to mean that
        for a 2-level comparison (unlike description's 3-level JaroWinklerAtThresholds,
        where an inversion between adjacent similarity thresholds is a different,
        blocking-shaped question -- exactly why this never applies outside
        `attr_names`). Verified real case (beans, `beans_preparation_method`,
        LLM_DEVICE=ollama): agreement_rate_true_pairs=0.0 vs
        agreement_rate_decoy_pairs=0.002 across 4 real rounds (see
        attribute_discriminative_power) -- true pairs never agreed on this attribute,
        so EM correctly learned exact-match as evidence AGAINST a match, and kept
        flagging label_switching every single round because attribute revision never
        touched this attribute.
      - "untrained": EM never got a usable estimate at all (no observed comparison
        levels) -- can't even attempt to say whether the comparison is useful, so it's
        treated the same as "no usable signal" rather than given the benefit of the
        doubt.

    `kinds` is overridable (e.g. outer_loop.diagnose_blocking_problem passes just
    `{"collapsed"}`, matching its own narrower, already-documented scope) but defaults
    to all three for the proactive-drop use case in linking/agent_loop.py, where the
    goal is catching every attribute-shaped degeneracy, not just one flavor of it.
    """
    return [f["column"] for f in flags if f.get("kind") in kinds and f.get("column") in attr_names]

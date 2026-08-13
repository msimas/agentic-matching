"""Splink's built-in Altair charts, for visually evaluating a block's trained linkage
model -- saved as standalone HTML files under the block's most recent run directory
(data/artifacts/<block>/<run_id>/, see config.latest_run_dir) so they can be opened
directly in a browser (no notebook needed), sitting alongside that run's own
blocking/attributes/linking artifacts rather than in a separate flat location.

Retrains the model fresh each call (same as every other consumer of a trained linker in
this codebase -- see linking/evaluate.py, linking/degeneracy_check.py -- nothing
persists a trained splink model to disk), except with
`retain_intermediate_calculation_columns=True`, which the waterfall chart specifically
requires and which linking/splink_model.py's production path otherwise turns off (see
its docstring) to bound memory. This is safe here: it was the *combination* of that flag
with the unbounded EM blocking pairs that exhausted memory, and that's fixed
independently in `splink_model.train()`; retaining the columns alone, at this project's
bounded block sizes, costs a few hundred MB to a couple GB, not tens of GB (verified:
~3.3GB peak for yogurt's ~660K candidate pairs).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from splink import Linker

from agentic_matching.attributes.agent_loop import load_latest_attributes
from agentic_matching.config import ARTIFACTS_DIR, latest_run_dir
from agentic_matching.linking import splink_model
from agentic_matching.linking.degeneracy_check import check_degeneracy, export_trained_settings

log = logging.getLogger(__name__)

SelectionMode = Literal["stratified", "top", "bottom", "borderline"]


def _default_chart_path(block_name: str, chart_kind: str) -> Path:
    """Where a chart lands when the caller doesn't pass an explicit `out_path` -- the
    block's most recent run directory (see config.latest_run_dir), not a new one:
    generating a chart isn't itself a pipeline run, it's a follow-up visualization of
    a run that already happened, so it belongs alongside that run's own artifacts, not
    off starting a fresh timestamped directory of its own.

    Raises FileNotFoundError, not a silent fallback to some flat/default location, if
    the block has no run directory yet -- run the pipeline (e.g.
    scripts/09_run_outer_loop.py) at least once first; a chart with nowhere real to
    live is a sign of that, not something to paper over with a made-up path."""
    run_dir = latest_run_dir(block_name)
    if run_dir is None:
        raise FileNotFoundError(
            f"No run directory exists yet for block '{block_name}' under "
            f"{ARTIFACTS_DIR / block_name} -- run the pipeline (e.g. "
            "scripts/09_run_outer_loop.py or scripts/07_run_splink_and_evaluate.py) "
            "at least once before generating charts, or pass an explicit out_path."
        )
    return run_dir / f"chart_{chart_kind}.html"


class ChartGenerationError(RuntimeError):
    """Splink's own chart-rendering code isn't resilient to a degenerate/tiny-block
    trained model (see where this is raised, below, for the two concrete bugs this was
    written for) -- raised in place of the raw upstream traceback so the failure is
    diagnosable (which flag, which block) instead of a bare ZeroDivisionError/TypeError
    several stack frames deep in splink/duckdb internals."""


# Splink's own chart-rendering code takes 1/probability_two_random_records_match and
# log()s it directly (see _run_splink_chart_call's docstring) -- a trained value of
# exactly 0.0 (a real possibility on a tiny/sparse block, verified against this
# project's "breaded_vegetables" block) crashes those charts outright. This floor is a
# chart-rendering-only workaround, applied *after* computing degeneracy_flags below (so
# the real degenerate value -- not this nudged one -- is what gets reported/flagged in
# linking_<block>_round*.json); it doesn't change the actual trained model or any
# non-chart evaluation, only lets the charts render instead of crashing.
_MIN_PROBABILITY_TWO_RANDOM_RECORDS_MATCH = 1e-8

# Splink's histogram chart bins match weights via DuckDB SQL; on a small/sparse block
# (few distinct values to bin) DuckDB can return those bin edges as its DECIMAL type
# rather than DOUBLE, which pandas then carries through as Python Decimal objects --
# and Decimal isn't JSON-serializable, so Altair's own `.save()` crashes trying to
# inline the chart's data into the HTML. Verified independent of (not downstream of) the
# probability_two_random_records_match=0.0 bug above: reproduced with a fully-trained,
# non-degenerate model (p2r=0.083) on this project's "breaded_vegetables" block once its
# attribute set grew a categorical attribute (vegetable_type) with several small
# categories. Altair's own `.save()` accepts a `json_kwds` passthrough to the stdlib
# `json.dumps` call that actually raises, so converting Decimal -> float there is a
# targeted fix for exactly this, not a broad monkeypatch of json serialization.
def _json_default(o: Any) -> Any:
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


_SAVE_KWARGS: dict[str, Any] = {"json_kwds": {"default": _json_default}}


def _build_trained_linker(block_name: str) -> tuple[Linker, list[dict[str, Any]]]:
    attrs = load_latest_attributes(block_name)
    linker, _fndds_df, _catalog_df, attrs = splink_model.build_linker(
        block_name, attrs, retain_intermediate_calculation_columns=True
    )
    splink_model.train(linker, attrs)
    degeneracy_flags = check_degeneracy(export_trained_settings(linker))

    p2r = linker._settings_obj._probability_two_random_records_match
    if p2r is not None and p2r <= 0:
        log.warning(
            "block '%s': trained probability_two_random_records_match was exactly %s "
            "-- nudging it to %.0e for chart rendering only (splink's chart code "
            "can't handle the literal degenerate value); the real value is still what "
            "degeneracy_flags/linking_round*.json (under the block's run directory) "
            "report.",
            block_name,
            p2r,
            _MIN_PROBABILITY_TWO_RANDOM_RECORDS_MATCH,
        )
        linker._settings_obj._probability_two_random_records_match = _MIN_PROBABILITY_TWO_RANDOM_RECORDS_MATCH

    return linker, degeneracy_flags


def _run_splink_chart_call(fn, block_name: str, chart_kind: str, degeneracy_flags: list[dict[str, Any]]):
    """Call a splink chart-generation function, converting any failure into a
    ChartGenerationError that names the block/chart and surfaces this round's
    degeneracy flags (the most common real cause -- e.g. a `probability_two_random_
    records_match` that trained to exactly 0.0 makes splink's own match_weights_chart
    divide by zero internally -- both this and the histogram's Decimal-serialization
    bug, once common on this project's own "breaded_vegetables" block, are worked
    around above/via _SAVE_KWARGS now, but this wrapper stays as a safety net for
    whatever's left/any other splink chart bug this project hasn't hit yet). Not every
    failure will be degeneracy-caused, so the flags are informational context, not a
    diagnosis."""
    try:
        return fn()
    except Exception as e:
        flags_desc = (
            "; this round's degeneracy flags: " + "; ".join(f"{f['kind']} ({f.get('column', '?')})" for f in degeneracy_flags)
            if degeneracy_flags
            else "; this round had no degeneracy flags, so this may be an unrelated failure"
        )
        raise ChartGenerationError(
            f"splink's {chart_kind} chart failed to generate for block '{block_name}' "
            f"({type(e).__name__}: {e}){flags_desc}. This is a failure inside splink's "
            "own chart-rendering code (commonly triggered by a very small/degenerate "
            "block), not something agentic_matching's own code can fix directly -- see "
            f"linking_round*.json under the most recent data/artifacts/{block_name}/<run_id>/ "
            "for the full evaluation."
        ) from e


def select_records(predictions: pd.DataFrame, n: int, mode: SelectionMode = "stratified") -> list[dict[str, Any]]:
    """Pick `n` predicted pairs to show in a waterfall chart. "top"/"bottom" show the
    most/least confident matches; "borderline" shows pairs nearest match_probability
    0.5 (the most informative for tuning); "stratified" (default) spreads the
    selection evenly across the whole score range so a chart shows a representative mix
    of clear matches, clear non-matches, and everything in between, rather than n
    near-identical high-confidence pairs."""
    if predictions.empty or n <= 0:
        return []
    df = predictions.sort_values("match_probability", ascending=False).reset_index(drop=True)
    if mode == "top":
        chosen = df.head(n)
    elif mode == "bottom":
        chosen = df.tail(n)
    elif mode == "borderline":
        chosen = df.iloc[(df["match_probability"] - 0.5).abs().sort_values().index[:n]]
    elif mode == "stratified":
        idx = [round(i * (len(df) - 1) / max(n - 1, 1)) for i in range(min(n, len(df)))]
        chosen = df.iloc[sorted(set(idx))]
    else:
        raise ValueError(f"Unknown selection mode: {mode!r}")
    return chosen.to_dict(orient="records")


def waterfall_chart(
    block_name: str,
    n: int = 10,
    mode: SelectionMode = "stratified",
    threshold: float = 0.0,
    out_path: Path | None = None,
) -> Path:
    """Save the match-weight waterfall chart (how each comparison contributed to the
    final match score) for `n` selected pairs from the block."""
    linker, degeneracy_flags = _build_trained_linker(block_name)
    predictions = linker.inference.predict(threshold_match_probability=threshold).as_pandas_dataframe()
    records = select_records(predictions, n=n, mode=mode)
    if not records:
        raise ValueError(f"No predicted pairs for block '{block_name}' at threshold={threshold}.")
    out_path = out_path or _default_chart_path(block_name, "waterfall")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_and_save() -> None:
        # Altair charts serialize their data lazily -- .save() is where a bad value
        # (e.g. the Decimal-type bug below) actually raises, not chart construction, so
        # both steps need to be inside the same wrapped call to be caught as one.
        linker.visualisations.waterfall_chart(records, filter_nulls=False).save(str(out_path), **_SAVE_KWARGS)

    _run_splink_chart_call(_create_and_save, block_name, "waterfall", degeneracy_flags)
    log.info("Wrote %s (%d records, mode=%s)", out_path, len(records), mode)
    return out_path


def match_weights_chart(block_name: str, out_path: Path | None = None) -> Path:
    """Save the per-comparison-level match weights chart (the model's learned
    'strength of evidence' for each comparison level) -- doesn't need predictions,
    just the trained model."""
    linker, degeneracy_flags = _build_trained_linker(block_name)
    out_path = out_path or _default_chart_path(block_name, "match_weights")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_and_save() -> None:
        linker.visualisations.match_weights_chart().save(str(out_path), **_SAVE_KWARGS)

    _run_splink_chart_call(_create_and_save, block_name, "match_weights", degeneracy_flags)
    log.info("Wrote %s", out_path)
    return out_path


def match_weights_histogram(block_name: str, threshold: float = 0.0, out_path: Path | None = None) -> Path:
    """Save a histogram of match weights across every predicted pair in the block."""
    linker, degeneracy_flags = _build_trained_linker(block_name)
    df_predict = linker.inference.predict(threshold_match_probability=threshold)
    out_path = out_path or _default_chart_path(block_name, "histogram")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_and_save() -> None:
        linker.visualisations.match_weights_histogram(df_predict).save(str(out_path), **_SAVE_KWARGS)

    _run_splink_chart_call(_create_and_save, block_name, "histogram", degeneracy_flags)
    log.info("Wrote %s", out_path)
    return out_path


def comparison_viewer_dashboard(
    block_name: str, threshold: float = 0.0, num_example_rows: int = 3, out_path: Path | None = None
) -> Path:
    """Save splink's interactive comparison-viewer dashboard (browse predicted pairs
    grouped by comparison-vector pattern) -- the most thorough of these for manual SME
    review, at the cost of a larger HTML file."""
    linker, degeneracy_flags = _build_trained_linker(block_name)
    df_predict = linker.inference.predict(threshold_match_probability=threshold)
    out_path = out_path or _default_chart_path(block_name, "dashboard")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_splink_chart_call(
        lambda: linker.visualisations.comparison_viewer_dashboard(
            df_predict, str(out_path), overwrite=True, num_example_rows=num_example_rows
        ),
        block_name,
        "dashboard",
        degeneracy_flags,
    )
    log.info("Wrote %s", out_path)
    return out_path

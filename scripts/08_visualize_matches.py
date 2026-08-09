#!/usr/bin/env python3
"""Generate splink's built-in Altair/HTML charts for a block's trained linkage model,
for visual evaluation of the matches (see linking/charts.py)."""

from __future__ import annotations

import logging
from typing import Callable

import typer

from agentic_matching.config import BLOCKS
from agentic_matching.linking import charts

app = typer.Typer(add_completion=False, help=__doc__)


def _check_block(block: str) -> None:
    if block not in BLOCKS:
        raise typer.BadParameter(f"--block must be one of {BLOCKS}, got {block!r}")


def _run(fn: Callable[[], object]) -> None:
    # charts.ChartGenerationError already carries a clear, block/chart-specific message
    # (see its docstring) -- print that instead of the raw upstream splink/duckdb
    # traceback, and exit non-zero rather than crashing typer's own handler.
    try:
        out = fn()
    except charts.ChartGenerationError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"wrote {out}")


@app.command()
def waterfall(
    block: str = typer.Option(..., "--block", help=f"Block to visualize: one of {BLOCKS}"),
    n: int = typer.Option(10, help="Number of pairs to include in the chart."),
    mode: str = typer.Option(
        "stratified", help="Which pairs to pick: stratified | top | bottom | borderline"
    ),
    threshold: float = typer.Option(0.0, help="Minimum match_probability to consider."),
) -> None:
    """Waterfall chart: how each comparison contributed to the final match score, for
    a selection of pairs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _check_block(block)
    _run(lambda: charts.waterfall_chart(block, n=n, mode=mode, threshold=threshold))  # type: ignore[arg-type]


@app.command()
def weights(block: str = typer.Option(..., "--block", help=f"Block to visualize: one of {BLOCKS}")) -> None:
    """Match weights chart: the model's learned strength of evidence per comparison level."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _check_block(block)
    _run(lambda: charts.match_weights_chart(block))


@app.command()
def histogram(
    block: str = typer.Option(..., "--block", help=f"Block to visualize: one of {BLOCKS}"),
    threshold: float = typer.Option(0.0, help="Minimum match_probability to consider."),
) -> None:
    """Histogram of match weights across every predicted pair in the block."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _check_block(block)
    _run(lambda: charts.match_weights_histogram(block, threshold=threshold))


@app.command()
def dashboard(
    block: str = typer.Option(..., "--block", help=f"Block to visualize: one of {BLOCKS}"),
    threshold: float = typer.Option(0.0, help="Minimum match_probability to consider."),
    num_example_rows: int = typer.Option(3, help="Example rows per comparison-vector pattern."),
) -> None:
    """Interactive comparison-viewer dashboard for browsing predicted pairs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _check_block(block)
    _run(lambda: charts.comparison_viewer_dashboard(block, threshold=threshold, num_example_rows=num_example_rows))


if __name__ == "__main__":
    app()

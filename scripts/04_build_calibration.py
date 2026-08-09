#!/usr/bin/env python3
"""Build the Branded<->OFF calibration (gold-match) dataset: gold pairs, stratified
SME spot-check sample, and train/holdout split."""

from __future__ import annotations

import typer

from agentic_matching.calibration import build
from agentic_matching.config import CALIBRATION_DIR, configure_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(
    holdout_frac: float = typer.Option(0.3, help="Fraction of gold GTIN groups held out for evaluation."),
    n_per_stratum: int = typer.Option(25, help="Max SME spot-check rows per branded_food_category."),
) -> None:
    configure_logging()
    build(holdout_frac=holdout_frac, n_per_stratum=n_per_stratum)
    typer.echo(f"Calibration artifacts written to {CALIBRATION_DIR}")


if __name__ == "__main__":
    app()

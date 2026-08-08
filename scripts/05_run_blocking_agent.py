#!/usr/bin/env python3
"""Run the bounded blocking-rule agent loop for a block, materializing the FNDDS/OFF
candidate subsets under data/blocks/ and logging each round to data/artifacts/."""

from __future__ import annotations

import logging

import typer

from agentic_matching.blocking.agent_loop import run_blocking_agent
from agentic_matching.config import BLOCKS

app = typer.Typer(add_completion=False)


@app.command()
def main(
    block: str = typer.Option(..., "--block", help=f"Block to run: one of {BLOCKS}"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if block not in BLOCKS:
        raise typer.BadParameter(f"--block must be one of {BLOCKS}, got {block!r}")
    rounds = run_blocking_agent(block)
    for r in rounds:
        typer.echo(
            f"round {r.round}: pair_completeness={r.metrics['pair_completeness']:.3f} "
            f"reduction_ratio={r.metrics['reduction_ratio']:.4f} "
            f"fndds={r.metrics['n_fndds_block']} off={r.metrics['n_off_block']}"
        )


if __name__ == "__main__":
    app()

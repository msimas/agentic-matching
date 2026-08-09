#!/usr/bin/env python3
"""Run the bounded outer loop for a block: blocking -> attributes -> linking, then
(if linking's result looks like a blocking problem, not an attribute problem) feed that
finding back into another round of blocking and repeat. See outer_loop.py's module
docstring for when/why this re-blocks."""

from __future__ import annotations

import typer

from agentic_matching.config import BLOCKS, configure_logging
from agentic_matching.outer_loop import run_outer_loop

app = typer.Typer(add_completion=False)


@app.command()
def main(
    block: str = typer.Option(..., "--block", help=f"Block to run: one of {BLOCKS}"),
) -> None:
    configure_logging()
    if block not in BLOCKS:
        raise typer.BadParameter(f"--block must be one of {BLOCKS}, got {block!r}")
    rounds = run_outer_loop(block)
    for r in rounds:
        typer.echo(
            f"outer round {r.round}: linking_rounds={r.linking_rounds_completed} "
            f"n_candidate_pairs={r.final_n_candidate_pairs} holdout_f1={r.final_holdout_f1} "
            f"trigger={r.trigger!r}"
        )
    if rounds and rounds[-1].trigger is None:
        typer.echo("No blocking problem found in the final outer round.")
    elif rounds:
        typer.echo("Outer loop ended with a blocking problem still flagged -- see the finding above.")


if __name__ == "__main__":
    app()

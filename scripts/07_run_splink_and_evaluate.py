#!/usr/bin/env python3
"""Run the bounded linking agent loop for a block: train splink -> check degeneracy ->
score against the Branded<->OFF holdout proxy -> LLM revises attributes -> retrain."""

from __future__ import annotations

import typer

from agentic_matching.config import BLOCKS, configure_logging
from agentic_matching.linking.agent_loop import run_linking_agent

app = typer.Typer(add_completion=False)


@app.command()
def main(
    block: str = typer.Option(..., "--block", help=f"Block to run: one of {BLOCKS}"),
) -> None:
    configure_logging()
    if block not in BLOCKS:
        raise typer.BadParameter(f"--block must be one of {BLOCKS}, got {block!r}")
    rounds = run_linking_agent(block)
    for r in rounds:
        ev = r.holdout_evaluation
        typer.echo(
            f"round {r.round}: degeneracy_flags={len(r.degeneracy_flags)} "
            f"precision={ev.get('precision')} recall={ev.get('recall')} f1={ev.get('f1')} "
            f"(category: precision={ev.get('category_precision')} recall={ev.get('category_recall')} "
            f"f1={ev.get('category_f1')}) "
            f"(ceiling: max_achievable_f1={ev.get('max_achievable_f1')} "
            f"resolvable={ev.get('n_resolvable')} ambiguous={ev.get('n_ambiguous')}) "
            f"n_pairs={r.plausibility.get('n_pairs')} -> {r.matches_csv}"
        )
    if rounds:
        last = rounds[-1]
        typer.echo(
            f"final deliverable: {last.n_final_matches} OFF records with a best FNDDS "
            f"match attached -> {last.final_matches_csv}"
        )


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""Run the bounded outer loop for a block: blocking -> attributes -> linking, then
(if linking's result looks like a blocking problem, not an attribute problem) feed that
finding back into another round of blocking and repeat. See outer_loop.py's module
docstring for when/why this re-blocks.

Use --steps to run only a subset of stages against a block already partway through the
pipeline -- e.g. --steps attributes,linking to redo attribute selection (and relink)
without touching the existing blocking rule, or --steps linking to just relink against
whatever attributes are already persisted."""

from __future__ import annotations

import typer

from agentic_matching.config import BLOCKS, configure_logging
from agentic_matching.llm.client import get_llm_client
from agentic_matching.outer_loop import ALL_STEPS, run_outer_loop

app = typer.Typer(add_completion=False)


@app.command()
def main(
    block: str = typer.Option(..., "--block", help=f"Block to run: one of {BLOCKS}"),
    steps: str = typer.Option(
        ",".join(ALL_STEPS),
        "--steps",
        help=f"Comma-separated subset of {ALL_STEPS} to run (pipeline order regardless "
        "of how you list them). A skipped stage isn't touched -- it isn't rerun with "
        "cached results, its prior output (e.g. attributes/generated/<block>/latest.json) "
        "is used as-is by whichever later stages you did include.",
    ),
) -> None:
    configure_logging()
    if block not in BLOCKS:
        raise typer.BadParameter(f"--block must be one of {BLOCKS}, got {block!r}")
    parsed_steps = [s.strip() for s in steps.split(",") if s.strip()]
    invalid = set(parsed_steps) - set(ALL_STEPS)
    if invalid:
        raise typer.BadParameter(f"--steps must be a subset of {ALL_STEPS}, got unknown step(s) {sorted(invalid)}")
    # Built here (not left for run_outer_loop to create internally) so this script can
    # echo the real usage total below -- an internally-created client would go out of
    # scope with no way to read its usage_totals back out.
    client = get_llm_client()
    rounds = run_outer_loop(block, client=client, steps=parsed_steps)
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
    u = client.usage_totals
    typer.echo(
        f"LLM usage (full outer loop -- steps={sorted(set(parsed_steps))}, block={block}): "
        f"{u['calls']} call(s), {u['prompt_tokens']} prompt + {u['completion_tokens']} completion "
        f"= {u['total_tokens']} total tokens"
    )


if __name__ == "__main__":
    app()

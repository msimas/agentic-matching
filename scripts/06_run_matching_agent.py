#!/usr/bin/env python3
"""Run the bounded matching-attribute agent loop for a block, persisting the final
attribute set under src/agentic_matching/attributes/generated/<block>/."""

from __future__ import annotations

import logging

import typer

from agentic_matching.attributes.generator import run_attribute_agent
from agentic_matching.config import BLOCKS

app = typer.Typer(add_completion=False)


@app.command()
def main(
    block: str = typer.Option(..., "--block", help=f"Block to run: one of {BLOCKS}"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if block not in BLOCKS:
        raise typer.BadParameter(f"--block must be one of {BLOCKS}, got {block!r}")
    rounds = run_attribute_agent(block)
    if not rounds:
        typer.echo(f"'{block}': round 0 failed with no seed to fall back to further, or no rounds ran.")
    for r in rounds:
        names = [a["name"] for a in r.attributes]
        typer.echo(f"round {r.round}: {len(names)} attributes ({names}), {len(r.correlation_flags)} correlation flags")


if __name__ == "__main__":
    app()

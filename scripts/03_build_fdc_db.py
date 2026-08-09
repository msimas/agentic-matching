#!/usr/bin/env python3
"""Build data/fdc.duckdb: raw per-dataset tables + unified per-dataset-type views
(v_fndds, v_sr_legacy, v_foundation, v_branded), all joined on fdc_id. Also builds the
corpus-wide profiling artifacts (token/category frequency over the full datasets) that
the blocking and matching-attribute agent loops use to ground LLM proposals in real
catalog statistics -- see profiling.py."""

from __future__ import annotations

import duckdb
import typer

from agentic_matching import profiling
from agentic_matching.build_fdc_db import build
from agentic_matching.config import FDC_DUCKDB_PATH, configure_logging

app = typer.Typer(add_completion=False)


@app.command()
def main(force: bool = typer.Option(True, help="Rebuild the DB from scratch.")) -> None:
    configure_logging()
    build(force=force)
    con = duckdb.connect(str(FDC_DUCKDB_PATH), read_only=True)
    for view in ("v_fndds", "v_sr_legacy", "v_foundation", "v_branded"):
        n = con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
        typer.echo(f"{view}: {n} rows")
    con.close()
    profiling.build(force=force)
    typer.echo(f"profiling: n_fndds={profiling.n_records('fndds')} n_off={profiling.n_records('off')}")


if __name__ == "__main__":
    app()

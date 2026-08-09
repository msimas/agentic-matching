#!/usr/bin/env python3
"""Convert every extracted FDC CSV to Parquet, and flatten OFF's struct/array text
fields into a single search-text Parquet file (used by every downstream stage)."""

from __future__ import annotations

import typer

from agentic_matching import off_text
from agentic_matching.config import PARQUET_FDC_DIR, configure_logging
from agentic_matching.csv_to_parquet import convert_all

app = typer.Typer(add_completion=False)


@app.command()
def main(force: bool = typer.Option(False, help="Re-convert even if Parquet outputs already exist.")) -> None:
    configure_logging()
    out = convert_all(force=force)
    for slug, paths in out.items():
        typer.echo(f"{slug}: {len(paths)} tables -> {PARQUET_FDC_DIR / slug}")
    off_text.build(force=force)
    typer.echo(f"off_search_text -> {off_text.OFF_SEARCH_TEXT_PARQUET}")


if __name__ == "__main__":
    app()

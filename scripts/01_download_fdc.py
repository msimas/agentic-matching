#!/usr/bin/env python3
"""Download + unzip FNDDS, SR Legacy, Foundation Foods, and Branded Foods CSVs."""

from __future__ import annotations

import typer

from agentic_matching.config import configure_logging
from agentic_matching.download_fdc import download_and_extract_all

app = typer.Typer(add_completion=False)


@app.command()
def main(force: bool = typer.Option(False, help="Re-download and re-extract even if already present.")) -> None:
    configure_logging()
    dirs = download_and_extract_all(force=force)
    for slug, d in dirs.items():
        n_csv = len(list(d.rglob("*.csv")))
        typer.echo(f"{slug}: {d} ({n_csv} CSV files)")


if __name__ == "__main__":
    app()

"""Download the current CSV releases of FNDDS, SR Legacy, Foundation Foods, and Branded
Foods from USDA FoodData Central, and unzip them.

The download page (https://fdc.nal.usda.gov/download-datasets) lists both a "Latest
Downloads" section and a "Historical Downloads" archive going back to 2019; filenames
embed a release date that changes every few months, so we scrape the page for the
current links rather than hardcoding them.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from agentic_matching.config import FDC_DATASETS, RAW_FDC_DIR, ZIPS_DIR, configure_logging

log = logging.getLogger(__name__)

DOWNLOAD_PAGE_URL = "https://fdc.nal.usda.gov/download-datasets"
FDC_DATASETS_HOST = "https://fdc.nal.usda.gov"

# Section markers used to scope the scrape to "current" releases only (skip the
# Historical Downloads archive further down the same page).
_LATEST_SECTION_START = 'id="bkmk-2"'
_LATEST_SECTION_END = "Historical Downloads"

_ZIP_HREF_RE = re.compile(r'href="([^"]+\.zip)"')


@dataclass(frozen=True)
class DatasetRelease:
    dataset: str  # key into FDC_DATASETS, e.g. "branded_food"
    url: str
    filename: str


def _fetch_download_page(client: httpx.Client) -> str:
    resp = client.get(DOWNLOAD_PAGE_URL, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    return resp.text


def _latest_section(html: str) -> str:
    start = html.find(_LATEST_SECTION_START)
    end = html.find(_LATEST_SECTION_END, start)
    if start == -1 or end == -1:
        raise RuntimeError(
            "Could not locate the 'Latest Downloads' section on the FDC download "
            "page — the page layout may have changed."
        )
    return html[start:end]


def discover_latest_csv_releases(html: str | None = None, client: httpx.Client | None = None) -> list[DatasetRelease]:
    """Scrape the FDC download page and return the current CSV zip for each of the
    four in-scope datasets."""
    if html is None:
        owns_client = client is None
        client = client or httpx.Client()
        try:
            html = _fetch_download_page(client)
        finally:
            if owns_client:
                client.close()

    section = _latest_section(html)
    hrefs = _ZIP_HREF_RE.findall(section)

    releases: list[DatasetRelease] = []
    for dataset_key in FDC_DATASETS:
        # e.g. "FoodData_Central_branded_food_csv_2026-04-30.zip"
        pattern = re.compile(rf"FoodData_Central_{dataset_key}_csv_[^/\"]+\.zip$")
        matches = [h for h in hrefs if pattern.search(h)]
        if not matches:
            raise RuntimeError(
                f"No current CSV release found for dataset '{dataset_key}' on the FDC "
                "download page. Check https://fdc.nal.usda.gov/download-datasets manually."
            )
        href = matches[0]
        url = href if href.startswith("http") else f"{FDC_DATASETS_HOST}{href}"
        filename = Path(href).name
        releases.append(DatasetRelease(dataset=dataset_key, url=url, filename=filename))
    return releases


def _download(client: httpx.Client, release: DatasetRelease, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest / release.filename
    if zip_path.exists() and zip_path.stat().st_size > 0:
        log.info("Already downloaded: %s", zip_path)
        return zip_path

    log.info("Downloading %s -> %s", release.url, zip_path)
    tmp_path = zip_path.with_suffix(".zip.part")
    with client.stream("GET", release.url, follow_redirects=True, timeout=300.0) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
    tmp_path.rename(zip_path)
    return zip_path


def _extract(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s -> %s", zip_path, dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def download_and_extract_all(force: bool = False) -> dict[str, Path]:
    """Download + unzip all four in-scope datasets. Returns dataset slug -> extracted dir."""
    with httpx.Client() as client:
        releases = discover_latest_csv_releases(client=client)
        result: dict[str, Path] = {}
        for release in releases:
            slug = FDC_DATASETS[release.dataset]
            extract_dir = RAW_FDC_DIR / slug
            if force and extract_dir.exists():
                for p in extract_dir.glob("*"):
                    p.unlink()
            zip_path = _download(client, release, ZIPS_DIR)
            if force or not any(extract_dir.rglob("*.csv")):
                _extract(zip_path, extract_dir)
            else:
                log.info("Already extracted: %s", extract_dir)
            result[slug] = extract_dir
        return result


if __name__ == "__main__":
    configure_logging()
    dirs = download_and_extract_all()
    for slug, d in dirs.items():
        n_csv = len(list(d.rglob("*.csv")))
        print(f"{slug}: {d} ({n_csv} CSV files)")

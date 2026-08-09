"""Central configuration: filesystem paths + LLM backend settings.

LLM backend settings (model, host/port, timeouts, ...) are controlled by environment
variables here (see .env.example) so no other module needs to change when they do.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

OFF_PARQUET = DATA_DIR / "food.parquet"

RAW_FDC_DIR = DATA_DIR / "raw" / "fdc"
ZIPS_DIR = RAW_FDC_DIR / "_zips"
PARQUET_FDC_DIR = DATA_DIR / "parquet" / "fdc"
FDC_DUCKDB_PATH = DATA_DIR / "fdc.duckdb"

CALIBRATION_DIR = DATA_DIR / "calibration"
BLOCKS_DIR = DATA_DIR / "blocks"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

# Corpus-wide profiling (token document frequency, categorical field distributions)
# computed once over the *full* datasets -- shared by both the blocking and
# matching-attribute prompts (and by llm/mock.py) so keyword/attribute proposals can be
# grounded in real catalog statistics instead of guessed from a handful of samples.
PROFILING_DIR = DATA_DIR / "profiling"

# Precomputed flat text view over OFF (struct-array fields extracted to plain strings)
# so blocking/matching stages don't repeatedly pay the struct-extraction cost over
# OFF's ~4.66M rows.
OFF_SEARCH_TEXT_PARQUET = DATA_DIR / "parquet" / "off_search_text.parquet"

# The four USDA datasets in scope for this POC (name -> directory slug).
FDC_DATASETS = {
    "foundation_food": "foundation",
    "sr_legacy_food": "sr_legacy",
    "survey_food": "fndds",
    "branded_food": "branded",
}

# Blocks in scope for the FNDDS<->OFF splink linkage.
BLOCKS = ["yogurt", "beans", "breaded_vegetables"]

for d in (
    RAW_FDC_DIR,
    ZIPS_DIR,
    PARQUET_FDC_DIR,
    CALIBRATION_DIR,
    BLOCKS_DIR,
    ARTIFACTS_DIR,
    PROFILING_DIR,
):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM backend settings
# ---------------------------------------------------------------------------


class LLMSettings(BaseSettings):
    """Settings for the locally-hosted Ollama LLM server, exposed as an
    OpenAI-compatible endpoint (see llm/client.py). `LLM_DEVICE=mock` selects the
    offline stand-in instead (see llm/mock.py) -- no server settings below apply then.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    device: str = "ollama"  # ollama | mock
    model: str = "qwen2.5:1.5b"
    base_url: str | None = None  # if set, connect to an already-running server instead
    host: str = "127.0.0.1"
    port: int = 11434
    api_key: str = "not-needed"
    request_timeout_s: float = 600.0
    max_tokens: int = 8192
    temperature: float = 0.33
    startup_timeout_s: float = 900.0  # self-hosted LLM server startup can take a while

    @property
    def effective_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        return f"http://{self.host}:{self.port}/v1"


class AgentLoopSettings(BaseSettings):
    """Bounds for the autonomous propose -> evaluate -> revise loops."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    max_rounds: int = 3
    # Stop early if the tracked metric changes by less than this between rounds.
    stabilization_delta: float = 0.01
    # Bound for outer_loop.run_outer_loop's blocking<->linking feedback cycle -- each
    # unit is a full blocking+attributes+linking cycle (many real-LLM calls, which can
    # add up to a long wall-clock time depending on the LLM backend/hardware in use),
    # so this is deliberately much smaller than max_rounds: 2 means "give re-blocking
    # one chance to fix a problem attribute revision structurally can't," not an
    # open-ended search.
    max_outer_rounds: int = 2


llm_settings = LLMSettings()
agent_loop_settings = AgentLoopSettings()


def configure_logging() -> None:
    """Call once, at the top of a script's `if __name__ == "__main__"` block, to set
    up console logging -- replaces the `logging.basicConfig(level=logging.INFO, ...)`
    line every script/entry point previously duplicated.

    Respects LOG_LEVEL (default INFO; not under LLMSettings/AgentLoopSettings' `LLM_`/
    `AGENT_` prefixes since it's not specific to either -- it's the standard,
    unprefixed name most CLI tools use for this). Set LOG_LEVEL=DEBUG to also see
    llm/client.py's and llm/mock.py's full LLM query/response logging, which is
    deliberately gated at DEBUG rather than INFO: it's the actual reasoning input/output
    for every agent-loop call, invaluable when diagnosing a bad or surprising proposal,
    but too verbose (full prompts, every round) to want on by default.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

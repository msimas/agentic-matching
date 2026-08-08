"""Central configuration: filesystem paths + device-agnostic LLM backend settings.

Everything that varies between "run on this CPU-only box" and "run on an NVIDIA/ROCm
GPU box later" is controlled by environment variables here (see .env.example) so no
other module needs to change when the hardware changes.
"""

from __future__ import annotations

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
BLOCKS = ["yogurt", "beans"]

for d in (
    RAW_FDC_DIR,
    ZIPS_DIR,
    PARQUET_FDC_DIR,
    CALIBRATION_DIR,
    BLOCKS_DIR,
    ARTIFACTS_DIR,
):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM backend settings
# ---------------------------------------------------------------------------


class LLMSettings(BaseSettings):
    """Settings for the locally-hosted vLLM OpenAI-compatible server.

    Switching hardware later is a one-variable change:
      - CPU (default, this box):       LLM_DEVICE=cpu
      - NVIDIA GPU:                    LLM_DEVICE=cuda LLM_DTYPE=bfloat16
      - AMD ROCm GPU:                  LLM_DEVICE=rocm LLM_DTYPE=float16
    No application code changes are required for the switch; only server.py's
    launch-flag construction branches on `device`.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    device: str = "cpu"  # cpu | cuda | rocm
    model: str = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    dtype: str = "bfloat16"
    base_url: str | None = None  # if set, connect to an already-running server instead
    host: str = "127.0.0.1"
    port: int = 8001
    api_key: str = "not-needed"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    request_timeout_s: float = 600.0
    max_tokens: int = 1024
    temperature: float = 0.2
    startup_timeout_s: float = 900.0  # CPU cold-start of an 8B model can be slow

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


llm_settings = LLMSettings()
agent_loop_settings = AgentLoopSettings()

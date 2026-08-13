"""Central configuration: filesystem paths + LLM backend settings.

LLM backend settings (model, host/port, timeouts, ...) are controlled by environment
variables here (see .env.example) so no other module needs to change when they do.
"""

from __future__ import annotations

import datetime
import logging
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
# Base directory for agent-loop run artifacts -- see run_artifacts_dir/new_run_id/
# latest_run_dir below for the actual data/artifacts/<block>/<run_id>/ layout each run
# writes under. This constant alone is not where any individual run's files live
# anymore (that's ARTIFACTS_DIR / block_name / run_id) -- kept as the shared root both
# helpers below build on, and for anything that genuinely wants the whole tree (e.g. a
# future "list all blocks with any runs" tool).
ARTIFACTS_DIR = DATA_DIR / "artifacts"


def new_run_id() -> str:
    """A sortable, filesystem-safe identifier for one top-level agent-loop invocation
    (a single run_blocking_agent/run_attribute_agent/run_linking_agent/run_outer_loop
    call) -- e.g. "20260812_194703". Lexicographic sort order matches chronological
    order (UTC, zero-padded, no separators ambiguous across filesystems) precisely so
    `latest_run_dir` below can find "the most recent run" with a plain max() over
    directory names, no separate pointer file or symlink to keep in sync."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_artifacts_dir(block_name: str, run_id: str) -> Path:
    """Where one run's artifacts (blocking/attributes/linking round records, matches
    CSVs, outer_loop summaries) are written -- data/artifacts/<block_name>/<run_id>/.
    Does NOT create the directory -- callers create it (or not) depending on whether
    they're about to write into it or just checking whether it exists."""
    return ARTIFACTS_DIR / block_name / run_id


def latest_run_dir(block_name: str, before: str | None = None) -> Path | None:
    """The most recently created run directory for `block_name`, or None if it has no
    runs yet. Run directories sort lexicographically by their `new_run_id` timestamp,
    so this is `max()` over what's actually on disk -- not a separate "latest" pointer
    that could drift out of sync with reality.

    `before`, if given (a run_id, not a full path), excludes that run from
    consideration -- for a run already in progress checking "what's the most recent
    OTHER run to copy artifacts forward from" (see outer_loop.py), which must not find
    its own not-yet-complete directory."""
    block_dir = ARTIFACTS_DIR / block_name
    if not block_dir.is_dir():
        return None
    run_dirs = sorted(p for p in block_dir.iterdir() if p.is_dir() and p.name != before)
    return run_dirs[-1] if run_dirs else None

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


class DatabricksSettings(BaseSettings):
    """Databricks Model Serving credentials, read from the same `DATABRICKS_HOST`/
    `DATABRICKS_TOKEN` env var names the Databricks CLI/SDK themselves use -- deliberately
    not duplicated under an `LLM_`-prefixed name, so a workspace already configured for
    other Databricks tooling doesn't need a second copy of the token. Only consulted
    when `LLM_DEVICE=databricks` (see LLMSettings.effective_base_url/effective_model/
    effective_api_key)."""

    model_config = SettingsConfigDict(env_prefix="DATABRICKS_", env_file=".env", extra="ignore")

    host: str | None = None
    token: str | None = None
    # Either a bare serving-endpoint name, or a full ".../serving-endpoints/<name>/
    # invocations?o=..." URL -- exactly what the Databricks UI's "Query endpoint" page
    # gives you to copy/paste (see _model_name_from_databricks_endpoint, below).
    llm_endpoint: str | None = None


databricks_settings = DatabricksSettings()


def _model_name_from_databricks_endpoint(value: str) -> str:
    """Accepts either a bare serving-endpoint name or a full Databricks invocations URL
    and returns just the endpoint name -- the `model` value Databricks' OpenAI-compatible
    `/serving-endpoints` gateway expects in the request body. Databricks' own foundation
    model endpoints also accept an OpenAI-style chat-completions body posted directly to
    the literal ".../invocations" URL, but the `openai` SDK always appends
    "/chat/completions" to whatever `base_url` it's given, so this project instead points
    `base_url` at the fixed "<host>/serving-endpoints" gateway path and lets Databricks
    route by the `model` field in the request body -- Databricks' own documented
    OpenAI-SDK integration pattern -- which is why the endpoint name has to be pulled
    back out of a full URL if that's what was pasted into DATABRICKS_LLM_ENDPOINT."""
    if "/serving-endpoints/" not in value:
        return value
    tail = value.split("/serving-endpoints/", 1)[1]
    return tail.split("/", 1)[0]


class LLMSettings(BaseSettings):
    """Settings for the LLM backend, exposed as an OpenAI-compatible endpoint (see
    llm/client.py) -- `LLM_DEVICE=ollama` (default) for a local/remote Ollama server,
    `LLM_DEVICE=databricks` for a Databricks Model Serving pay-per-token endpoint (see
    DatabricksSettings above), `LLM_DEVICE=mock` for the offline stand-in (see
    llm/mock.py, no server settings below apply then).
    """

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    device: str = "ollama"  # ollama | mock | databricks
    model: str = "qwen2.5:1.5b"
    base_url: str | None = None  # if set, connect to an already-running server instead
    host: str = "127.0.0.1"
    port: int = 11434
    api_key: str = "not-needed"
    request_timeout_s: float = 600.0
    max_tokens: int = 8192
    # Used for revision rounds (an existing rule/attribute set to refine, evidence to
    # stay faithful to -- see llm/prompts.py's "existing_attributes"/"previous_rule"
    # branch) and as the fallback for any call that doesn't distinguish rounds (e.g.
    # linking/agent_loop.py, which never has a "propose from scratch" round -- it
    # always starts from an already-persisted attribute set, so every one of its
    # rounds is a revision). Kept low on purpose: a revision round should mostly
    # behave deterministically -- read the concrete evaluation feedback (false
    # positives/negatives, discriminative power, correlation flags) and make a
    # targeted, well-grounded edit, or explicitly leave things unchanged if they
    # already look optimal -- not sample across stylistically different revisions.
    # Was 0.33; lowered after a real Databricks/Llama-3.3-70B run where a revision
    # round made the attribute set measurably worse (degeneracy_flags 4 -> 6) despite
    # having the same evaluation feedback as the round before it -- see
    # select_best_round's regression guard, which caught and reverted it, but the
    # goal is fewer unforced regressions like that in the first place, not just
    # catching them after the fact.
    temperature: float = 0.15
    # Used only for a genuine "propose from scratch" round (blocking/attributes' round
    # 0 with no seed/previous state) -- higher than `temperature` on purpose: nothing
    # exists yet to be faithful to, so more sampling diversity is pure upside there,
    # unlike a revision round where it risks drifting from the evidence just shown.
    exploration_temperature: float = 0.6
    startup_timeout_s: float = 900.0  # self-hosted LLM server startup can take a while

    @property
    def effective_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        if self.device == "databricks":
            if not databricks_settings.host:
                raise ValueError(
                    "LLM_DEVICE=databricks requires DATABRICKS_HOST to be set (or set "
                    "LLM_BASE_URL directly)."
                )
            return f"{databricks_settings.host.rstrip('/')}/serving-endpoints"
        return f"http://{self.host}:{self.port}/v1"

    @property
    def effective_invocation_url(self) -> str:
        """The literal Databricks ".../serving-endpoints/<name>/invocations" URL to
        POST directly -- NOT the same as `effective_base_url` (which points at the
        shared "<host>/serving-endpoints" gateway path Databricks documents for their
        curated pay-per-token Foundation Model APIs, routing by a "model" field in the
        request body -- the pattern `llm/client.py`'s openai-SDK path would normally
        use). Verified real case: that shared gateway returned an HTML login-page
        redirect for a real named serving endpoint on a real workspace, while POSTing
        directly to this literal per-endpoint URL returned a real, well-formed API
        response (a 403 with a clear scope error, once auth was actually reached) --
        the shared gateway's routing pattern doesn't reliably apply to every endpoint,
        but a named endpoint's own literal invocations URL always does, so
        `llm/client.py` uses this (via a direct HTTP POST, bypassing the openai SDK's
        automatic "/chat/completions" URL suffixing) for every Databricks call.
        """
        if not databricks_settings.llm_endpoint:
            raise ValueError("LLM_DEVICE=databricks requires DATABRICKS_LLM_ENDPOINT to be set.")
        value = databricks_settings.llm_endpoint
        if "/serving-endpoints/" in value:
            return value
        if not databricks_settings.host:
            raise ValueError(
                "LLM_DEVICE=databricks requires DATABRICKS_HOST to be set when "
                "DATABRICKS_LLM_ENDPOINT is a bare endpoint name rather than a full URL."
            )
        return f"{databricks_settings.host.rstrip('/')}/serving-endpoints/{value}/invocations"

    @property
    def effective_model(self) -> str:
        """The human-readable model/endpoint name -- sent as the `model` field in the
        request body for Ollama/generic OpenAI-compatible backends; for Databricks
        it's log/display-only (see effective_invocation_url's docstring for why that
        one, not this one, determines which endpoint a Databricks call actually hits --
        the literal URL already identifies the endpoint, no `model` field needed)."""
        # "model" not in model_fields_set -- only fall back to the Databricks-derived
        # value when the user hasn't explicitly set LLM_MODEL themselves (same
        # explicit-always-wins pattern this project used for its old Ollama-default
        # fields, before Ollama became the only local backend and that got simplified
        # away -- still the right call here since there IS more than one backend again).
        if self.device == "databricks" and "model" not in self.model_fields_set:
            if not databricks_settings.llm_endpoint:
                raise ValueError(
                    "LLM_DEVICE=databricks requires DATABRICKS_LLM_ENDPOINT (or LLM_MODEL) to be set."
                )
            return _model_name_from_databricks_endpoint(databricks_settings.llm_endpoint)
        return self.model

    @property
    def effective_api_key(self) -> str:
        if self.device == "databricks" and "api_key" not in self.model_fields_set:
            if not databricks_settings.token:
                raise ValueError(
                    "LLM_DEVICE=databricks requires DATABRICKS_TOKEN (or LLM_API_KEY) to be set."
                )
            return databricks_settings.token
        return self.api_key


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


class LoggingSettings(BaseSettings):
    """`LOG_LEVEL` (default INFO; not under LLMSettings/AgentLoopSettings' `LLM_`/
    `AGENT_` prefixes since it's not specific to either -- it's the standard,
    unprefixed name most CLI tools use for this). A pydantic BaseSettings, like every
    other settings class here, specifically so it reads `.env` the same way they do --
    verified real bug otherwise: a plain `os.environ.get("LOG_LEVEL")` (this class's
    predecessor) only sees a variable actually exported in the shell, silently ignoring
    `.env`'s LOG_LEVEL=DEBUG whenever a script is launched without that also being
    exported (e.g. via `nohup uv run ...` from a shell that never `export`ed it)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"


llm_settings = LLMSettings()
agent_loop_settings = AgentLoopSettings()
logging_settings = LoggingSettings()


def round_temperature(round_idx: int, has_prior_state: bool) -> float:
    """Shared temperature-selection logic for all three agent loops (blocking,
    attributes, linking): `LLMSettings.exploration_temperature` only for a genuine
    round-0 "propose from scratch" call (round_idx==0 AND nothing to build on -- no
    seed rule/attribute set), `LLMSettings.temperature` for every other call. A round-0
    call that already has a seed to refine is a revision, not a from-scratch proposal,
    same as a round-0 call in linking's loop (which never lacks prior state -- it
    always starts from an already-persisted attribute set)."""
    if round_idx == 0 and not has_prior_state:
        return llm_settings.exploration_temperature
    return llm_settings.temperature


def configure_logging() -> None:
    """Call once, at the top of a script's `if __name__ == "__main__"` block, to set
    up console logging -- replaces the `logging.basicConfig(level=logging.INFO, ...)`
    line every script/entry point previously duplicated.

    Respects LOG_LEVEL (see LoggingSettings above). Set LOG_LEVEL=DEBUG to also see
    llm/client.py's and llm/mock.py's full LLM query/response logging, which is
    deliberately gated at DEBUG rather than INFO: it's the actual reasoning input/output
    for every agent-loop call, invaluable when diagnosing a bad or surprising proposal,
    but too verbose (full prompts, every round) to want on by default.
    """
    level = getattr(logging, logging_settings.log_level.upper(), None)
    if not isinstance(level, int):
        level = logging.INFO
    # force=True -- logging.basicConfig() is a no-op if the root logger already has
    # handlers (e.g. something else configured logging first, or this is called more
    # than once), which would otherwise make a second call to this function silently
    # not change anything.
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)

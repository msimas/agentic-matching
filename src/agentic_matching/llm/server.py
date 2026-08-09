"""Start/stop a local LLM server (vLLM or Ollama) as a subprocess, both exposed as an
OpenAI-compatible endpoint.

The launch flags are the only thing that branch on hardware/backend (`LLM_DEVICE`);
every other module talks to the server exclusively through the OpenAI-compatible HTTP
API (see llm/client.py), so switching backend or moving from this CPU box to an
NVIDIA/ROCm GPU box later is a config change, not a code change.

Neither vLLM nor Ollama is a hard dependency of this project (see README) -- vLLM's CPU
build in particular is not published as a regular PyPI wheel (the default `vllm` wheel
bundles CUDA runtime libraries and expects an NVIDIA GPU) and can be finicky to build;
Ollama is generally the easier path to a working CPU backend (a single installer, no
manual build step) if vLLM's CPU build gives trouble. This module shells out to the
`vllm`/`ollama` CLIs rather than importing either as a library, so the rest of the
codebase has no import-time dependency on either.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time

import httpx

from agentic_matching.config import LLMSettings, llm_settings

log = logging.getLogger(__name__)


class VLLMNotInstalledError(RuntimeError):
    pass


class VLLMStartupError(RuntimeError):
    pass


class OllamaNotInstalledError(RuntimeError):
    pass


class OllamaStartupError(RuntimeError):
    pass


def _build_launch_args(settings: LLMSettings) -> list[str]:
    args = [
        "vllm",
        "serve",
        settings.model,
        "--host",
        settings.host,
        "--port",
        str(settings.port),
        "--dtype",
        settings.dtype,
        "--max-model-len",
        str(settings.max_model_len),
    ]
    if settings.device == "cpu":
        # vLLM's CPU backend infers device from how it was installed (the CPU build);
        # no --device flag is needed/supported on recent versions, but tensor-parallel
        # and gpu-memory-utilization are meaningless here.
        pass
    elif settings.device in ("cuda", "rocm"):
        args += [
            "--gpu-memory-utilization",
            str(settings.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(settings.tensor_parallel_size),
        ]
    else:
        raise ValueError(f"Unknown LLM_DEVICE for vLLM: {settings.device!r}")
    return args


class VLLMServerManager:
    """Context manager around a `vllm serve` subprocess. No-op if `LLM_BASE_URL` is
    already set (assume an externally-managed / remote server)."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or llm_settings
        self._proc: subprocess.Popen | None = None

    @property
    def externally_managed(self) -> bool:
        return bool(self.settings.base_url)

    def start(self) -> None:
        if self.externally_managed:
            log.info("LLM_BASE_URL is set (%s) — not launching a local server.", self.settings.base_url)
            self._wait_ready()
            return
        if self._proc is not None:
            return
        if shutil.which("vllm") is None:
            raise VLLMNotInstalledError(
                "The `vllm` CLI was not found on PATH. Install it for your device "
                "(see README.md's 'LLM backend setup' section) or set LLM_BASE_URL to "
                "point at an already-running OpenAI-compatible server. If vLLM's CPU "
                "build is giving you trouble, consider LLM_DEVICE=ollama instead."
            )
        args = _build_launch_args(self.settings)
        log.info("Launching vLLM server: %s", " ".join(args))
        self._proc = subprocess.Popen(args)
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.settings.startup_timeout_s
        url = f"{self.settings.effective_base_url}/models"
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise VLLMStartupError(
                    f"vLLM server process exited early with code {self._proc.returncode}."
                )
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    log.info("vLLM server ready at %s", self.settings.effective_base_url)
                    return
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(2.0)
        raise VLLMStartupError(
            f"vLLM server did not become ready within {self.settings.startup_timeout_s}s "
            f"(last error: {last_err})"
        )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            log.info("Stopping vLLM server (pid %d)", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __enter__(self) -> "VLLMServerManager":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class OllamaServerManager:
    """Context manager around Ollama, which behaves differently from vLLM in two ways
    this class has to account for:

    1. Ollama is commonly already running as a persistent background service (e.g. the
       official installer sets up a systemd service on Linux that starts on login) --
       unlike vLLM, `ollama serve` isn't normally something you launch per-run. `start()`
       checks whether something is already listening at the configured host/port and
       reuses it rather than erroring or double-launching, only spawning its own
       `ollama serve` subprocess if nothing answers yet. `stop()` mirrors this: it only
       terminates the process if *this* manager was the one that started it, so it
       doesn't kill a persistent service other things may depend on.
    2. Ollama doesn't take the model as a launch argument the way `vllm serve <model>`
       does -- a model has to be pulled (downloaded) into Ollama's local store before it
       can be served. `start()` runs `ollama pull <model>` (a no-op, fast, if already
       present) after the server is confirmed reachable.

    No-op (beyond waiting ready + pulling the model) if `LLM_BASE_URL` is already set
    (assume an externally-managed / remote server that already has the model).

    Requires a reasonably recent Ollama version for its OpenAI-compatible `/v1` API
    (including `/v1/models` for the readiness check, and `response_format` support in
    `/v1/chat/completions`, which `llm/client.py`'s JSON-mode calls rely on) -- this
    hasn't been exercised against a real Ollama install in this repo's own testing
    (Ollama is not installed in this dev environment either), so treat it as a starting
    point and check your installed version's docs if something doesn't line up.
    """

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or llm_settings
        self._proc: subprocess.Popen | None = None
        self._we_started_it = False

    @property
    def externally_managed(self) -> bool:
        return bool(self.settings.base_url)

    def _is_reachable(self) -> bool:
        try:
            resp = httpx.get(f"{self.settings.effective_base_url}/models", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def start(self) -> None:
        if self.externally_managed:
            log.info("LLM_BASE_URL is set (%s) — not launching a local server.", self.settings.base_url)
            self._wait_ready()
        elif self._is_reachable():
            log.info("Ollama server already running at %s — reusing it.", self.settings.effective_base_url)
        else:
            if shutil.which("ollama") is None:
                raise OllamaNotInstalledError(
                    "The `ollama` CLI was not found on PATH. Install it from "
                    "https://ollama.com/download, or set LLM_BASE_URL to point at an "
                    "already-running OpenAI-compatible server."
                )
            log.info("Launching `ollama serve`...")
            self._proc = subprocess.Popen(["ollama", "serve"])
            self._we_started_it = True
            self._wait_ready()
        self._pull_model()

    def _pull_model(self) -> None:
        log.info("Ensuring model '%s' is pulled (no-op if already present)...", self.settings.model)
        result = subprocess.run(["ollama", "pull", self.settings.model])
        if result.returncode != 0:
            raise OllamaStartupError(
                f"`ollama pull {self.settings.model}` failed (exit code {result.returncode})."
            )

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.settings.startup_timeout_s
        url = f"{self.settings.effective_base_url}/models"
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise OllamaStartupError(
                    f"Ollama server process exited early with code {self._proc.returncode}."
                )
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code == 200:
                    log.info("Ollama server ready at %s", self.settings.effective_base_url)
                    return
            except httpx.HTTPError as e:
                last_err = e
            time.sleep(2.0)
        raise OllamaStartupError(
            f"Ollama server did not become ready within {self.settings.startup_timeout_s}s "
            f"(last error: {last_err})"
        )

    def stop(self) -> None:
        if self._we_started_it and self._proc is not None and self._proc.poll() is None:
            log.info("Stopping Ollama server (pid %d)", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __enter__(self) -> "OllamaServerManager":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


ServerManager = VLLMServerManager | OllamaServerManager


def get_server_manager(settings: LLMSettings | None = None) -> ServerManager:
    """Selects the right manager for `settings.device` (default: module-level
    `llm_settings`, i.e. whatever `LLM_DEVICE` is set to)."""
    settings = settings or llm_settings
    if settings.device == "ollama":
        return OllamaServerManager(settings)
    return VLLMServerManager(settings)


if __name__ == "__main__":
    # Standalone entry point: `uv run python -m agentic_matching.llm.server` starts the
    # server (per LLM_DEVICE/LLM_MODEL/... in .env) and blocks in the foreground until
    # interrupted -- run this in its own terminal, then run the pipeline scripts
    # (LLM_DEVICE=cpu is the default, so no extra flags needed there) against it. Not
    # invoked by any agent-loop script itself: they only ever talk to a server that's
    # already listening (llm/client.py connects to LLM_BASE_URL/LLM_HOST:LLM_PORT), so
    # if `LLM_BASE_URL` is unset and no server is running, they'll fail with a
    # connection error the first time an LLM call is made.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manager = get_server_manager()
    try:
        manager.start()
        if manager.externally_managed:
            log.info("LLM_BASE_URL is externally managed; nothing to hold open here. Exiting.")
        elif isinstance(manager, OllamaServerManager) and not manager._we_started_it:
            log.info("Ollama was already running independently; nothing to hold open here. Exiting.")
        else:
            log.info(
                "Server ready at %s (model=%s) -- Ctrl+C to stop",
                manager.settings.effective_base_url,
                manager.settings.model,
            )
            assert manager._proc is not None
            manager._proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()

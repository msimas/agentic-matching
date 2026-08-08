"""Start/stop a local vLLM OpenAI-compatible server as a subprocess.

The launch flags are the only thing that branch on hardware (`LLM_DEVICE`); every other
module talks to the server exclusively through the OpenAI-compatible HTTP API, so moving
from this CPU box to an NVIDIA/ROCm GPU box later is a config change, not a code change.

vLLM itself is *not* a hard dependency of this project (see README) — its CPU build is
not published as a regular PyPI wheel (the default `vllm` wheel bundles CUDA runtime
libraries and expects an NVIDIA GPU) and must be installed separately following vLLM's
device-specific instructions. This module shells out to the `vllm` CLI rather than
importing vllm as a library, so the rest of the codebase has no import-time dependency
on it.
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
        raise ValueError(f"Unknown LLM_DEVICE: {settings.device!r}")
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
                "point at an already-running OpenAI-compatible server."
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

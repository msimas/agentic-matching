"""Thin client for talking to an OpenAI-compatible LLM server -- a local/remote Ollama
server, or a Databricks Model Serving pay-per-token endpoint (see
config.DatabricksSettings), selected via `LLM_DEVICE`.

Every agent-loop module (blocking, attributes, linking) calls `LLMClient.complete_json`
and never touches the OpenAI SDK, httpx, or raw HTTP directly, so the hosting details
(which of the above, or a mock for offline development) are fully swappable via
`get_llm_client()` / `LLM_DEVICE`.

Databricks is handled as a direct HTTP POST (httpx), not through the `openai` SDK, even
though the rest of this module is otherwise SDK-based -- verified real case: the openai
SDK always appends "/chat/completions" to whatever `base_url` it's given, which matches
Databricks' documented shared gateway path for their curated pay-per-token Foundation
Model APIs ("<host>/serving-endpoints", routing by a "model" field in the request body),
but that shared gateway returned an HTML login-page redirect for a real named serving
endpoint on a real workspace -- POSTing directly to that endpoint's own literal
".../serving-endpoints/<name>/invocations" URL (LLMSettings.effective_invocation_url)
instead returned a real, well-formed API response. Not every Databricks endpoint
apparently supports the shared-gateway routing pattern, but every endpoint's own literal
invocations URL does, so that's what this module always uses.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from openai import OpenAI

from agentic_matching.config import LLMSettings, llm_settings

log = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(content: str) -> str:
    """Strip a wrapping ```json ... ``` (or bare ``` ... ```) markdown code fence.

    Verified real case: Databricks-hosted Llama-3.3-70B wraps every JSON response in a
    markdown fence despite being told to reply with ONLY a JSON object -- unlike the
    Ollama-served qwen3 models, which don't (and which additionally get
    response_format={"type": "json_object"} enforcement that Databricks doesn't, see
    LLMClient._send's docstring). Without this, every Databricks call fails
    json.loads and burns all of complete_json's retries. A no-op if there's no fence.
    """
    stripped = content.strip()
    match = _CODE_FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


class ChatClient(ABC):
    """Common interface implemented by both the real Ollama-backed client and the
    offline mock used for development/testing without a running LLM server."""

    @abstractmethod
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion and parse the response as JSON. Raises ValueError if
        the response is not valid JSON after retries."""
        raise NotImplementedError


class LLMClient(ChatClient):
    """Real client, talking to an Ollama, Databricks, or any OpenAI-compatible server."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or llm_settings
        self._is_databricks = self.settings.device == "databricks"
        if self._is_databricks:
            self._httpx_client = httpx.Client(timeout=self.settings.request_timeout_s)
        else:
            self._client = OpenAI(
                base_url=self.settings.effective_base_url,
                api_key=self.settings.effective_api_key,
                timeout=self.settings.request_timeout_s,
            )

    def _send(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        """Returns the raw response content string. See this module's docstring for why
        Databricks bypasses the openai SDK entirely here."""
        if self._is_databricks:
            resp = self._httpx_client.post(
                self.settings.effective_invocation_url,
                headers={"Authorization": f"Bearer {self.settings.effective_api_key}"},
                json={"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""
        resp = self._client.chat.completions.create(
            model=self.settings.effective_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            # Not requested for Databricks -- unverified whether a given
            # Databricks-served model honors OpenAI's strict response_format contract
            # (see this module's docstring for the same "don't assume, verify" lesson
            # that applied to the URL shape); the system prompts already instruct
            # "reply with ONLY a JSON object", and complete_json's retry-on-parse-
            # failure loop below is the fallback if that alone isn't reliable enough.
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        effective_max_tokens = max_tokens or self.settings.max_tokens
        effective_temperature = temperature if temperature is not None else self.settings.temperature
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            # Full query/response logged at DEBUG, not INFO -- this is the actual
            # reasoning input/output for every agent-loop call, invaluable when
            # diagnosing a bad or surprising proposal, but full prompts on every round
            # are too verbose to want on by default. Set LOG_LEVEL=DEBUG (see
            # config.configure_logging) to see it.
            log.debug(
                "LLM request (model=%s, attempt %d/%d):\n----- system -----\n%s\n----- user -----\n%s",
                self.settings.effective_model,
                attempt + 1,
                max_retries + 1,
                system,
                user,
            )
            # At INFO (not DEBUG) -- unlike the full prompt dump above, this is the one
            # message every caller needs to see by default: an LLM call is the only
            # slow step in these agent loops (everything else -- splink training, SQL
            # queries -- finishes in seconds), so without this the program can look
            # hung for a minute or more with no indication of what it's doing.
            log.info(
                "Waiting for LLM response (model=%s%s)...",
                self.settings.effective_model,
                f", retry {attempt}/{max_retries}" if attempt else "",
            )
            content = self._send(messages, effective_max_tokens, effective_temperature)
            log.debug("LLM response (attempt %d):\n%s", attempt + 1, content)
            try:
                return json.loads(_strip_code_fence(content))
            except json.JSONDecodeError as e:
                last_err = e
                log.warning("LLM response was not valid JSON (attempt %d): %s", attempt, content[:500])
        raise ValueError(f"LLM did not return valid JSON after {max_retries + 1} attempts: {last_err}")


def get_llm_client(settings: LLMSettings | None = None) -> ChatClient:
    """Factory: LLM_DEVICE=mock returns the offline stand-in (see llm/mock.py); any
    other value returns the real Ollama-backed client."""
    settings = settings or llm_settings
    if settings.device == "mock":
        from agentic_matching.llm.mock import MockChatClient

        return MockChatClient()
    return LLMClient(settings)

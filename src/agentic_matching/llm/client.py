"""Thin client for talking to the local (or remote) vLLM OpenAI-compatible server.

Every agent-loop module (blocking, attributes, linking) calls `LLMClient.complete_json`
and never touches the OpenAI SDK or HTTP directly, so the hosting details (CPU today,
CUDA/ROCm later, or a mock for offline development) are fully swappable via
`get_llm_client()` / `LLM_DEVICE`.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from agentic_matching.config import LLMSettings, llm_settings

log = logging.getLogger(__name__)


class ChatClient(ABC):
    """Common interface implemented by both the real vLLM-backed client and the
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
    """Real client, talking to a vLLM (or any OpenAI-compatible) server."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or llm_settings
        self._client = OpenAI(
            base_url=self.settings.effective_base_url,
            api_key=self.settings.api_key,
            timeout=self.settings.request_timeout_s,
        )

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            resp = self._client.chat.completions.create(
                model=self.settings.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens or self.settings.max_tokens,
                temperature=temperature if temperature is not None else self.settings.temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                last_err = e
                log.warning("LLM response was not valid JSON (attempt %d): %s", attempt, content[:500])
        raise ValueError(f"LLM did not return valid JSON after {max_retries + 1} attempts: {last_err}")


def get_llm_client(settings: LLMSettings | None = None) -> ChatClient:
    """Factory: LLM_DEVICE=mock returns the offline stand-in (see llm/mock.py); any
    other value returns the real vLLM-backed client."""
    settings = settings or llm_settings
    if settings.device == "mock":
        from agentic_matching.llm.mock import MockChatClient

        return MockChatClient()
    return LLMClient(settings)

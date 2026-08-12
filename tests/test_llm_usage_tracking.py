import logging

from agentic_matching.config import LLMSettings
from agentic_matching.llm.client import LLMClient
from agentic_matching.llm.mock import MockChatClient


def _settings(**kwargs) -> LLMSettings:
    # See test_llm_server.py's _settings for why _env_file=None: isolates from a real
    # dev's .env (e.g. LLM_DEVICE=databricks) so these tests build a plain ollama-style
    # client -- constructing it never makes a network call, only complete_json does.
    return LLMSettings(_env_file=None, **kwargs)


def _client() -> LLMClient:
    return LLMClient(_settings())


def test_usage_totals_start_at_zero():
    client = _client()
    assert client.usage_totals == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_record_usage_accumulates_across_calls():
    client = _client()
    client._record_usage({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    client._record_usage({"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60})
    assert client.usage_totals == {"calls": 2, "prompt_tokens": 150, "completion_tokens": 30, "total_tokens": 180}


def test_record_usage_counts_the_call_even_when_usage_is_none():
    # A backend that doesn't report usage still made a real call -- the call count
    # must reflect that even though the token counts can't.
    client = _client()
    client._record_usage(None)
    assert client.usage_totals == {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_record_usage_missing_keys_default_to_zero_not_crash():
    client = _client()
    client._record_usage({"prompt_tokens": 10})  # no completion_tokens/total_tokens key
    assert client.usage_totals == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 0}


def test_log_usage_summary_does_not_crash_with_or_without_label(caplog):
    caplog.set_level(logging.INFO)
    client = _client()
    client._record_usage({"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10})
    client.log_usage_summary()
    client.log_usage_summary(label="beans linking, cumulative")
    assert any("LLM usage" in r.message for r in caplog.records)


def test_mock_client_has_usage_totals_and_stays_zero():
    # MockChatClient makes no real API calls -- reporting nonzero usage would be
    # fabricated, not estimated, so it must inherit ChatClient's zeroed accumulator
    # and never touch it.
    client = MockChatClient()
    assert client.usage_totals == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    client.complete_json("sys", '{"existing_attributes": [], "decisions": []}')
    assert client.usage_totals == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

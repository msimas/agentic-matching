import pytest

import agentic_matching.config as config_module
from agentic_matching.config import DatabricksSettings, LLMSettings, _model_name_from_databricks_endpoint


def _llm_settings(**kwargs) -> LLMSettings:
    """See test_llm_server.py's identical helper for why `_env_file=None` -- isolates
    these tests from whatever the real project .env currently has set."""
    return LLMSettings(_env_file=None, **kwargs)


def _databricks_settings(**kwargs) -> DatabricksSettings:
    return DatabricksSettings(_env_file=None, **kwargs)


@pytest.fixture
def databricks(monkeypatch):
    """Swap the module-level `databricks_settings` singleton (what LLMSettings'
    effective_* properties actually read) for an isolated instance, so these tests
    don't depend on -- or clobber -- the real project .env's Databricks credentials."""

    def _set(**kwargs):
        instance = _databricks_settings(**kwargs)
        monkeypatch.setattr(config_module, "databricks_settings", instance)
        return instance

    return _set


# -- _model_name_from_databricks_endpoint ---------------------------------------------


def test_bare_endpoint_name_returned_unchanged():
    assert _model_name_from_databricks_endpoint("databricks-meta-llama-3-3-70b-instruct") == (
        "databricks-meta-llama-3-3-70b-instruct"
    )


def test_full_invocations_url_extracts_endpoint_name():
    url = "https://weslytics.databricks.com/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations?o=523071927315659"
    assert _model_name_from_databricks_endpoint(url) == "databricks-meta-llama-3-3-70b-instruct"


def test_full_url_without_query_string_also_works():
    url = "https://host.databricks.com/serving-endpoints/my-endpoint/invocations"
    assert _model_name_from_databricks_endpoint(url) == "my-endpoint"


# -- LLMSettings.effective_base_url / effective_model / effective_api_key -------------


def test_effective_base_url_derived_from_databricks_host(databricks):
    databricks(host="https://weslytics.databricks.com")
    s = _llm_settings(device="databricks")
    assert s.effective_base_url == "https://weslytics.databricks.com/serving-endpoints"


def test_effective_base_url_strips_trailing_slash(databricks):
    databricks(host="https://weslytics.databricks.com/")
    s = _llm_settings(device="databricks")
    assert s.effective_base_url == "https://weslytics.databricks.com/serving-endpoints"


def test_explicit_llm_base_url_wins_over_databricks_host(databricks):
    databricks(host="https://weslytics.databricks.com")
    s = _llm_settings(device="databricks", base_url="https://override.example.com/v1")
    assert s.effective_base_url == "https://override.example.com/v1"


def test_effective_base_url_raises_without_databricks_host(databricks):
    databricks(host=None)
    s = _llm_settings(device="databricks")
    with pytest.raises(ValueError, match="DATABRICKS_HOST"):
        _ = s.effective_base_url


def test_effective_invocation_url_passes_through_full_url_unchanged(databricks):
    url = "https://weslytics.databricks.com/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations?o=523071927315659"
    databricks(llm_endpoint=url)
    s = _llm_settings(device="databricks")
    assert s.effective_invocation_url == url


def test_effective_invocation_url_constructed_from_bare_name_and_host(databricks):
    databricks(host="https://weslytics.databricks.com", llm_endpoint="my-endpoint")
    s = _llm_settings(device="databricks")
    assert s.effective_invocation_url == "https://weslytics.databricks.com/serving-endpoints/my-endpoint/invocations"


def test_effective_invocation_url_raises_without_endpoint(databricks):
    databricks(llm_endpoint=None)
    s = _llm_settings(device="databricks")
    with pytest.raises(ValueError, match="DATABRICKS_LLM_ENDPOINT"):
        _ = s.effective_invocation_url


def test_effective_invocation_url_raises_for_bare_name_without_host(databricks):
    databricks(host=None, llm_endpoint="my-endpoint")
    s = _llm_settings(device="databricks")
    with pytest.raises(ValueError, match="DATABRICKS_HOST"):
        _ = s.effective_invocation_url


def test_effective_model_derived_from_databricks_endpoint(databricks):
    databricks(llm_endpoint="databricks-meta-llama-3-3-70b-instruct")
    s = _llm_settings(device="databricks")
    assert s.effective_model == "databricks-meta-llama-3-3-70b-instruct"


def test_effective_model_parses_full_url(databricks):
    databricks(llm_endpoint="https://host/serving-endpoints/my-endpoint/invocations?o=1")
    s = _llm_settings(device="databricks")
    assert s.effective_model == "my-endpoint"


def test_explicit_llm_model_wins_over_databricks_endpoint(databricks):
    databricks(llm_endpoint="databricks-meta-llama-3-3-70b-instruct")
    s = _llm_settings(device="databricks", model="some-other-endpoint")
    assert s.effective_model == "some-other-endpoint"


def test_effective_model_raises_without_databricks_endpoint(databricks):
    databricks(llm_endpoint=None)
    s = _llm_settings(device="databricks")
    with pytest.raises(ValueError, match="DATABRICKS_LLM_ENDPOINT"):
        _ = s.effective_model


def test_effective_api_key_derived_from_databricks_token(databricks):
    databricks(token="dapi-some-token")
    s = _llm_settings(device="databricks")
    assert s.effective_api_key == "dapi-some-token"


def test_explicit_llm_api_key_wins_over_databricks_token(databricks):
    databricks(token="dapi-some-token")
    s = _llm_settings(device="databricks", api_key="override-key")
    assert s.effective_api_key == "override-key"


def test_effective_api_key_raises_without_databricks_token(databricks):
    databricks(token=None)
    s = _llm_settings(device="databricks")
    with pytest.raises(ValueError, match="DATABRICKS_TOKEN"):
        _ = s.effective_api_key


def test_ollama_device_ignores_databricks_settings_entirely(databricks):
    databricks(host=None, token=None, llm_endpoint=None)
    s = _llm_settings(device="ollama")
    assert s.effective_base_url == "http://127.0.0.1:11434/v1"
    assert s.effective_model == "qwen2.5:1.5b"
    assert s.effective_api_key == "not-needed"

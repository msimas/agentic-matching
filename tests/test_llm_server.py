import pytest

import agentic_matching.llm.server as server_module
from agentic_matching.config import LLMSettings
from agentic_matching.llm.server import OllamaServerManager, get_server_manager


def _settings(**kwargs) -> LLMSettings:
    """LLMSettings reads the real project .env by default (env_file=".env"), so a
    dev's actual local settings (e.g. a customized LLM_MODEL) would otherwise leak into
    these tests, which want to check pure class defaults + explicit overrides in
    isolation. `_env_file=None` disables that file lookup for just this instance
    without touching real process env vars."""
    return LLMSettings(_env_file=None, **kwargs)


def test_get_server_manager_returns_ollama_manager():
    assert isinstance(get_server_manager(_settings()), OllamaServerManager)


def test_get_server_manager_uses_module_default_when_no_settings_given(monkeypatch):
    # Shouldn't raise -- falls back to the module-level llm_settings. The real
    # project .env's LLM_DEVICE (e.g. "databricks") shouldn't affect this: swap the
    # module-level singleton for an isolated ollama-device instance for this check.
    monkeypatch.setattr(server_module, "llm_settings", _settings(device="ollama"))
    manager = get_server_manager()
    assert isinstance(manager, OllamaServerManager)


def test_get_server_manager_rejects_mock_device():
    with pytest.raises(ValueError):
        get_server_manager(_settings(device="mock"))


def test_get_server_manager_rejects_databricks_device():
    # Databricks Model Serving is always an already-running cloud service -- nothing
    # for get_server_manager to start/stop, same rationale as mock.
    with pytest.raises(ValueError):
        get_server_manager(_settings(device="databricks"))


def test_default_settings_use_ollama_port_and_model():
    s = _settings()
    assert s.device == "ollama"
    assert s.port == 11434
    assert s.model == "qwen2.5:1.5b"
    assert s.effective_base_url == "http://127.0.0.1:11434/v1"


def test_settings_respect_explicit_overrides():
    s = _settings(port=9999, model="llama3.2:1b")
    assert s.port == 9999
    assert s.model == "llama3.2:1b"


def test_externally_managed_true_when_base_url_set():
    s = _settings(base_url="http://remote:11434/v1")
    assert OllamaServerManager(s).externally_managed
    assert s.effective_base_url == "http://remote:11434/v1"


def test_externally_managed_false_by_default():
    assert not OllamaServerManager(_settings()).externally_managed


def test_ollama_stop_is_noop_if_it_did_not_start_the_process():
    manager = OllamaServerManager(_settings())
    manager._we_started_it = False
    manager.stop()  # should not raise even though _proc is None
    assert manager._proc is None

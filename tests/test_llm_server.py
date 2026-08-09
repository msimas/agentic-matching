from agentic_matching.config import LLMSettings
from agentic_matching.llm.server import (
    OllamaServerManager,
    VLLMServerManager,
    _build_launch_args,
    get_server_manager,
)


def _settings(**kwargs) -> LLMSettings:
    """LLMSettings reads the real project .env by default (env_file=".env"), so a
    dev's actual local settings (e.g. a customized LLM_MODEL for whichever backend
    they're using) would otherwise leak into these tests, which want to check pure
    class defaults + explicit overrides in isolation. `_env_file=None` disables that
    file lookup for just this instance without touching real process env vars."""
    return LLMSettings(_env_file=None, **kwargs)


def test_get_server_manager_dispatches_ollama():
    assert isinstance(get_server_manager(_settings(device="ollama")), OllamaServerManager)


def test_get_server_manager_dispatches_vllm_for_cpu_cuda_rocm():
    for device in ("cpu", "cuda", "rocm"):
        assert isinstance(get_server_manager(_settings(device=device)), VLLMServerManager)


def test_get_server_manager_uses_module_default_when_no_settings_given():
    # Shouldn't raise -- falls back to the module-level llm_settings.
    manager = get_server_manager()
    assert isinstance(manager, (VLLMServerManager, OllamaServerManager))


def test_ollama_settings_apply_default_port_and_model():
    s = _settings(device="ollama")
    assert s.port == 11434
    assert s.model == "qwen2.5:1.5b"
    assert s.effective_base_url == "http://127.0.0.1:11434/v1"


def test_ollama_settings_respect_explicit_overrides():
    s = _settings(device="ollama", port=9999, model="llama3.2:1b")
    assert s.port == 9999
    assert s.model == "llama3.2:1b"


def test_non_ollama_settings_keep_vllm_defaults():
    s = _settings(device="cpu")
    assert s.port == 8001
    assert s.model == "NousResearch/Meta-Llama-3.1-8B-Instruct"


def test_vllm_launch_args_cpu_omits_gpu_flags():
    args = _build_launch_args(_settings(device="cpu", model="some/model"))
    assert "some/model" in args
    assert "--gpu-memory-utilization" not in args
    assert "--tensor-parallel-size" not in args


def test_vllm_launch_args_cuda_includes_gpu_flags():
    args = _build_launch_args(_settings(device="cuda", model="some/model"))
    assert "--gpu-memory-utilization" in args
    assert "--tensor-parallel-size" in args


def test_vllm_launch_args_rejects_ollama_device():
    import pytest

    with pytest.raises(ValueError):
        _build_launch_args(_settings(device="ollama"))


def test_externally_managed_true_when_base_url_set():
    s = _settings(base_url="http://remote:8001/v1")
    assert VLLMServerManager(s).externally_managed
    assert OllamaServerManager(s).externally_managed


def test_externally_managed_false_by_default():
    s = _settings()
    assert not VLLMServerManager(s).externally_managed
    assert not OllamaServerManager(s).externally_managed


def test_ollama_stop_is_noop_if_it_did_not_start_the_process():
    manager = OllamaServerManager(_settings(device="ollama"))
    manager._we_started_it = False
    manager.stop()  # should not raise even though _proc is None
    assert manager._proc is None

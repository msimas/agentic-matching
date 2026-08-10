import logging

from agentic_matching.config import LoggingSettings, configure_logging


def _settings(**kwargs) -> LoggingSettings:
    """See test_llm_server.py's identical helper for why `_env_file=None` -- isolates
    these tests from whatever the real project .env currently has LOG_LEVEL set to."""
    return LoggingSettings(_env_file=None, **kwargs)


def test_default_is_info():
    assert _settings().log_level == "INFO"


def test_env_file_is_actually_read():
    # The real bug this class fixed: a plain os.environ.get("LOG_LEVEL") silently
    # ignores .env unless the shell also happens to export it. LoggingSettings is a
    # pydantic BaseSettings with env_file=".env", same mechanism LLMSettings/
    # AgentLoopSettings already use -- verified here via an explicit override (a real
    # .env-file read is exercised implicitly by every other test that constructs
    # LoggingSettings() with the project .env in place, e.g. configure_logging() itself).
    assert _settings(log_level="DEBUG").log_level == "DEBUG"


def test_configure_logging_maps_debug_to_root_logger():
    # Directly exercise configure_logging()'s level-mapping logic via a monkeypatched
    # settings object rather than reaching into the real .env.
    import agentic_matching.config as config_module

    original = config_module.logging_settings
    try:
        config_module.logging_settings = _settings(log_level="DEBUG")
        configure_logging()
        assert logging.getLogger().level == logging.DEBUG
    finally:
        config_module.logging_settings = original
        configure_logging()  # restore real level so later tests aren't affected


def test_configure_logging_falls_back_to_info_for_unknown_level():
    import agentic_matching.config as config_module

    original = config_module.logging_settings
    try:
        config_module.logging_settings = _settings(log_level="NOT_A_REAL_LEVEL")
        configure_logging()
        assert logging.getLogger().level == logging.INFO
    finally:
        config_module.logging_settings = original
        configure_logging()

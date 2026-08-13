import pytest

import agentic_matching.config as config
from agentic_matching.linking.charts import _default_chart_path


def test_raises_when_block_has_no_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="beans"):
        _default_chart_path("beans", "waterfall")


def test_uses_the_most_recent_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    (tmp_path / "beans" / "20200101_000000").mkdir(parents=True)
    (tmp_path / "beans" / "20260101_000000").mkdir(parents=True)  # more recent
    result = _default_chart_path("beans", "waterfall")
    assert result == tmp_path / "beans" / "20260101_000000" / "chart_waterfall.html"


def test_filename_reflects_chart_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    (tmp_path / "beans" / "20260101_000000").mkdir(parents=True)
    assert _default_chart_path("beans", "dashboard").name == "chart_dashboard.html"

from pathlib import Path

import pytest

from tdp_core.paths import data_dir, models_dir, results_dir


def test_defaults_are_relative_to_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("TDP_DATA_DIR", "TDP_MODELS_DIR", "TDP_RESULTS_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    assert data_dir() == tmp_path / "data"
    assert models_dir() == tmp_path / "models"
    assert results_dir() == tmp_path / "results"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TDP_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("TDP_MODELS_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("TDP_RESULTS_DIR", str(tmp_path / "r"))
    assert (data_dir(), models_dir(), results_dir()) == (tmp_path / "d", tmp_path / "m", tmp_path / "r")

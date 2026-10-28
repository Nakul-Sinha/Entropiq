"""The production pointer and the per-process bundle cache."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tdp_core.model.bundle import VERSION_PATTERN, Bundle
from tdp_worker import bundle_loader
from tdp_worker.bundle_loader import (
    VERSION_RE,
    BundleUnavailableError,
    ProductionPointer,
    bundle_path,
    clear_cache,
    get_bundle,
    get_production_bundle,
    production_version,
    read_production_pointer,
)


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_cache()


def _pointer(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "production.json").write_text(text, encoding="utf-8")
    return folder


def test_version_pattern_matches_the_bundle_module() -> None:
    assert VERSION_RE.pattern == VERSION_PATTERN.pattern


def test_read_production_pointer(tmp_path: Path) -> None:
    folder = _pointer(tmp_path, json.dumps({"version": "ranker-2026.10.1", "uri": "u", "sha256": "abc"}))
    assert read_production_pointer(folder) == ProductionPointer("ranker-2026.10.1", "u", "abc")
    _pointer(tmp_path, json.dumps({"version": "dev-1"}))
    assert read_production_pointer(folder) == ProductionPointer("dev-1", None, None)
    assert production_version(folder) == "dev-1"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (None, "No production model is configured."),
        ('{"version": null, "uri": null, "sha256": null}', "No production model is configured."),
        ("{}", "No production model is configured."),
        ("{not json", "The production model pointer is invalid."),
        ('["dev-1"]', "The production model pointer is invalid."),
        ('{"version": "../evil"}', "The production model pointer is invalid."),
        ('{"version": 7}', "The production model pointer is invalid."),
        ('{"version": "dev-1", "uri": 3}', "The production model pointer is invalid."),
        ('{"version": "dev-1", "sha256": ["x"]}', "The production model pointer is invalid."),
    ],
)
def test_bad_pointers(tmp_path: Path, text: str | None, message: str) -> None:
    if text is not None:
        _pointer(tmp_path, text)
    with pytest.raises(BundleUnavailableError) as info:
        read_production_pointer(tmp_path)
    assert str(info.value) == message
    assert str(tmp_path) not in str(info.value)


def test_undecodable_pointer(tmp_path: Path) -> None:
    (tmp_path / "production.json").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(BundleUnavailableError, match="invalid"):
        read_production_pointer(tmp_path)


def test_bundle_path_rejects_traversal(tmp_path: Path) -> None:
    assert bundle_path(tmp_path, "dev-1") == tmp_path / "dev-1"
    for bad in ("../evil", "a/b", "", ".hidden", "x" * 200):
        with pytest.raises(BundleUnavailableError) as info:
            bundle_path(tmp_path, bad)
        assert str(info.value) == "The model version is not valid."


def test_get_bundle_caches_per_folder_and_version(models_dir: Path) -> None:
    first = get_bundle("dev-worker", model_dir=models_dir)
    assert isinstance(first, Bundle)
    assert first.version == "dev-worker"
    assert get_bundle("dev-worker", model_dir=models_dir) is first
    clear_cache()
    assert get_bundle("dev-worker", model_dir=models_dir) is not first


def test_missing_version(models_dir: Path) -> None:
    with pytest.raises(BundleUnavailableError) as info:
        get_bundle("dev-missing", model_dir=models_dir)
    assert str(info.value) == "Model version dev-missing is not installed on this worker."


def test_folder_holding_another_version(models_dir: Path, tmp_path: Path) -> None:
    shutil.copytree(models_dir / "dev-worker", tmp_path / "dev-other")
    with pytest.raises(BundleUnavailableError) as info:
        get_bundle("dev-other", model_dir=tmp_path)
    assert str(info.value) == "Model version dev-other is not installed on this worker."


def test_broken_bundle(models_dir: Path, tmp_path: Path) -> None:
    shutil.copytree(models_dir / "dev-worker", tmp_path / "dev-worker")
    (tmp_path / "dev-worker" / "calibrator.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BundleUnavailableError) as info:
        get_bundle("dev-worker", model_dir=tmp_path)
    assert str(info.value) == "Model version dev-worker could not be loaded."


def test_get_production_bundle(models_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    bundle = get_production_bundle(model_dir=models_dir)
    assert bundle.version == "dev-worker"
    assert "synthetic development bundle" in caplog.text


def test_defaults_come_from_the_settings(worker_env: dict[str, Path]) -> None:
    assert production_version() == "dev-worker"
    assert get_production_bundle() is get_bundle("dev-worker")
    assert bundle_loader.get_bundle("dev-worker", model_dir=worker_env["models"]) is get_bundle("dev-worker")

"""Tests of the MLflow helpers and the ``tdp-ml mlflow-relocate`` command."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

import tdp_ml.tracking as trk
from tdp_ml.commands.train import register

_ENV = ("MLFLOW_ALLOW_FILE_STORE", "MLFLOW_DISABLE_TELEMETRY", "MLFLOW_DISABLE_AGENT_HINT")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the MLflow switches for the test and restore them afterwards."""
    for name in _ENV:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _cli() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        """Test app."""

    register(app)
    return app


def test_configure_mlflow_sets_env_defaults(tmp_path: Path, clean_env: None) -> None:
    uri = (tmp_path / "mlruns").as_uri()
    assert trk.configure_mlflow(uri) == uri
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert os.environ["MLFLOW_DISABLE_AGENT_HINT"] == "1"
    import mlflow

    assert mlflow.get_tracking_uri() == uri


def test_configure_mlflow_keeps_existing_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "false")
    trk.configure_mlflow((tmp_path / "m").as_uri())
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "false"


def test_tracking_uri(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "  file:///x/mlruns ")
    assert trk.tracking_uri() == "file:///x/mlruns"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    monkeypatch.chdir(tmp_path)
    assert trk.tracking_uri() == (tmp_path / "mlruns").absolute().as_uri()


def test_file_store_root(tmp_path: Path) -> None:
    assert trk.file_store_root(tmp_path.as_uri()) == tmp_path
    assert trk.file_store_root(str(tmp_path)) == tmp_path
    assert trk.file_store_root("C:/x") == Path("C:/x")
    assert trk.file_store_root("mlruns") == Path("mlruns")
    assert trk.file_store_root("s3://bucket/mlruns") is None
    assert trk.file_store_root("http://localhost:5000") is None


def test_relocate_a_moved_store(tmp_path: Path, clean_env: None) -> None:
    import mlflow

    source = tmp_path / "a"
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")
    with trk.mlflow_run("relocate-test", "run-1", uri=source.as_uri(), tags={"kind": "test"}) as run:
        run.log_params({"flag": True, "items": [3, 1], "spec": {"b": 1, "a": 2}, "name": "x"})
        run.log_metrics({"score": 0.5, "bad": float("nan")}, step=1)
        run.log_artifact(note, "notes")
        run.set_tags({"extra": "1"})
    assert run.experiment == "relocate-test" and run.tracking_uri == source.as_uri()
    client = mlflow.tracking.MlflowClient()
    stored = client.get_run(run.run_id)
    assert stored.info.status == "FINISHED"
    assert stored.data.params == {"flag": "true", "items": "[3,1]", "spec": '{"a":2,"b":1}', "name": "x"}
    assert stored.data.metrics == {"score": 0.5}
    assert stored.data.tags["kind"] == "test" and stored.data.tags["extra"] == "1"

    target = tmp_path / "b"
    shutil.move(str(source), str(target))
    trk.configure_mlflow(target.as_uri())
    with pytest.raises(Exception):  # noqa: B017  the old artifact folder is gone
        mlflow.artifacts.download_artifacts(
            run_id=run.run_id, artifact_path="notes/note.txt", dst_path=str(tmp_path / "before")
        )
    changed = trk.relocate_file_store(target)
    exp_meta = target / run.experiment_id / "meta.yaml"
    run_meta = target / run.experiment_id / run.run_id / "meta.yaml"
    assert exp_meta in changed and run_meta in changed
    assert changed == sorted(changed) and all(p.name == "meta.yaml" for p in changed)
    local = mlflow.artifacts.download_artifacts(
        run_id=run.run_id, artifact_path="notes/note.txt", dst_path=str(tmp_path / "after")
    )
    assert Path(local).read_text(encoding="utf-8") == "hello"
    assert trk.relocate_file_store(target) == []


def test_failed_block_marks_the_run_failed(tmp_path: Path, clean_env: None) -> None:
    import mlflow

    with (
        pytest.raises(RuntimeError, match="boom"),
        trk.mlflow_run("fail-test", "r", uri=(tmp_path / "m").as_uri()),
    ):
        raise RuntimeError("boom")
    runs = mlflow.search_runs(experiment_names=["fail-test"], output_format="list")
    assert [r.info.status for r in runs] == ["FAILED"]


def test_log_params_rejects_long_values(tmp_path: Path, clean_env: None) -> None:
    with trk.mlflow_run("long-test", "r", uri=(tmp_path / "m").as_uri()) as run, pytest.raises(ValueError):
        run.log_params({"big": "x" * 6001})


def _fake_store(root: Path, artifact_location: str, artifact_uri: str) -> tuple[Path, Path]:
    run_dir = root / "7" / "abc"
    run_dir.mkdir(parents=True)
    (root / ".trash").mkdir()
    exp_meta = root / "7" / "meta.yaml"
    run_meta = run_dir / "meta.yaml"
    exp_meta.write_text(
        yaml.safe_dump({"artifact_location": artifact_location, "name": "e"}), encoding="utf-8"
    )
    run_meta.write_text(yaml.safe_dump({"artifact_uri": artifact_uri, "run_id": "abc"}), encoding="utf-8")
    return exp_meta, run_meta


def test_relocate_leaves_remote_locations(tmp_path: Path) -> None:
    exp_meta, run_meta = _fake_store(tmp_path / "s", "s3://bucket/7", "s3://bucket/7/abc/artifacts")
    before = exp_meta.read_text(encoding="utf-8"), run_meta.read_text(encoding="utf-8")
    assert trk.relocate_file_store(tmp_path / "s") == []
    assert (exp_meta.read_text(encoding="utf-8"), run_meta.read_text(encoding="utf-8")) == before
    with pytest.raises(FileNotFoundError):
        trk.relocate_file_store(tmp_path / "missing")


def test_relocate_rewrites_box_paths(tmp_path: Path) -> None:
    root = tmp_path / "s"
    exp_meta, run_meta = _fake_store(root, "file:///work/mlruns/7", "/work/mlruns/7/abc/artifacts")
    assert trk.relocate_file_store(root) == sorted([exp_meta, run_meta])
    exp = yaml.safe_load(exp_meta.read_text(encoding="utf-8"))
    run = yaml.safe_load(run_meta.read_text(encoding="utf-8"))
    assert exp == {"artifact_location": (root / "7").absolute().as_uri(), "name": "e"}
    assert run["artifact_uri"] == (root / "7" / "abc" / "artifacts").absolute().as_uri()
    assert list(exp) == ["artifact_location", "name"]


def test_relocate_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "s"
    _fake_store(root, "file:///work/mlruns/7", "file:///work/mlruns/7/abc/artifacts")
    runner = CliRunner()
    result = runner.invoke(_cli(), ["mlflow-relocate", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert f"relocated 2 files under {root}" in result.output
    monkeypatch.setenv("MLFLOW_TRACKING_URI", root.as_uri())
    assert "relocated 0 files" in runner.invoke(_cli(), ["mlflow-relocate"]).output
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "s3://bucket/mlruns")
    assert runner.invoke(_cli(), ["mlflow-relocate"]).exit_code == 2
    assert runner.invoke(_cli(), ["mlflow-relocate", "--root", str(tmp_path / "none")]).exit_code == 2

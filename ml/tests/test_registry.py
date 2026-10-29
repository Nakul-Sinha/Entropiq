"""Tests of the model registry, its promotion gates and the register, golden and promote commands.

Everything is synthetic: a temporary MLflow file store with fake training runs that carry the dev bundle's
booster and calibrator, a fake data folder and two small scripted repositories as golden mirrors. No test
writes the tracked ``models/production.json`` or the shared data folders.
"""

from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import typer
from typer.testing import CliRunner, Result

import tdp_ml.registry as reg
from tdp_core.model.bundle import BUNDLE_FILES, load_bundle, make_dev_bundle
from tdp_core.model.calibration import IsotonicCalibrator
from tdp_core.model.synthetic import synthetic_dataset
from tdp_core.testing.repo_builder import RepoBuilder
from tdp_ml import tracking as trk
from tdp_ml.commands.registry import register
from tdp_ml.pipeline.config import ConfigError
from tdp_ml.pipeline.layout import DataLayout, write_json
from tdp_ml.train import TunedParams, file_sha256, monotone_columns, save_params

_MLFLOW_ENV = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_ALLOW_FILE_STORE",
    "MLFLOW_DISABLE_TELEMETRY",
    "MLFLOW_DISABLE_AGENT_HINT",
    "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR",
)
NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)
T0 = datetime(2022, 1, 3, 12, tzinfo=UTC)
MANIFEST_SHA = "a" * 64
MONOTONE = list(monotone_columns(("commits_decayed", "churn_decayed")))


def _lopo_metrics(*, ndcg: float, p: float, ece: float) -> dict[str, float]:
    return {
        "lopo_ndcg_10": ndcg,
        "lopo_ndcg_10_ci_low": ndcg - 0.08,
        "lopo_ndcg_10_ci_high": ndcg + 0.08,
        "lopo_precision_10": 0.5,
        "lopo_precision_10_ci_low": 0.4,
        "lopo_precision_10_ci_high": 0.6,
        "lopo_recall_20loc": 0.2,
        "lopo_popt": 0.55,
        "lopo_hotspot_ndcg_10": 0.70,
        "lopo_ndcg_10_p_greater_hotspot": p,
        "calibration_ece": ece,
        "calibration_ece_ci_low": ece / 2,
        "calibration_ece_ci_high": ece * 3,
        "calibration_brier": 0.07,
        "calibration_brier_skill": 0.25,
        "calibration_auc_pr": 0.5,
        "calibration_query_ece": 0.13,
    }


@dataclass(frozen=True)
class Store:
    root: Path
    uri: str
    dev: Path
    params: Path
    data: Path
    dataset_sha: str
    good: str
    bad: str
    no_calibrator: str
    temporal: str
    models: Path
    good_version: str
    bad_version: str


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    root = tmp_path_factory.mktemp("registry")
    with pytest.MonkeyPatch.context() as mp:
        for name in _MLFLOW_ENV:
            mp.setenv(name, "")
            mp.delenv(name)
        mp.setenv("TDP_CODE_SHA", "c0ffee")
        uri = (root / "mlruns").as_uri()
        dev = make_dev_bundle(root / "dev", version="dev-test")
        tuned = TunedParams(
            params={"objective": "rank:ndcg", "n_estimators": 50, "max_depth": 4, "learning_rate": 0.2},
            best_params={"max_depth": 4},
            best_value=0.7,
            fold_rounds=(45,),
            info={"seed": 42},
        )
        params = save_params(root / "art" / "ranker_params.json", tuned)

        data = root / "data"
        frame = synthetic_dataset(12, 20)
        data.mkdir()
        frame.to_parquet(data / "dataset.parquet", index=False)
        sha = file_sha256(data / "dataset.parquet")
        counts = frame["qid"].value_counts()
        queries = [
            {"repo": "x", "snapshot_idx": 0, "qid": int(q), "rows": int(counts[q]), "passed": True}
            for q in sorted(counts.index)[:10]
        ]
        write_json(
            DataLayout(data).run_record("check-leakage"),
            {"sample": 10, "seed": 42, "checked": 10, "passed": True, "queries": queries},
        )

        with trk.mlflow_run(
            "entropiq-ranker", "ranker-temporal", uri=uri, tags={"model": "ranker", "protocol": "temporal"}
        ) as run:
            run.log_metrics({"temporal_ndcg_10": 0.74, "temporal_ndcg_10_p_greater_hotspot": 0.2})
        temporal = run.run_id

        def lopo(metrics: dict[str, float], *, calibrator: bool = True, monotone: bool = True) -> str:
            tags = {
                "model": "ranker",
                "protocol": "lopo",
                "monotone": str(monotone).lower(),
                "dataset_sha256": sha,
                "manifest_sha256": MANIFEST_SHA,
                "code_git_sha": "feed",
                "tuned_in_run": temporal,
                "device": "cpu",
            }
            with trk.mlflow_run("entropiq-ranker", "ranker-lopo", uri=uri, tags=tags) as run:
                run.log_params(
                    {
                        "training_rows": len(frame),
                        "training_queries": 12,
                        "horizon_days": 365,
                        "n_estimators": 50,
                        "monotone": monotone,
                        "monotone_columns": json.dumps(MONOTONE) if monotone else "",
                    }
                )
                run.log_metrics(metrics)
                run.log_artifact(dev / "ranker.json", "model")
                if calibrator:
                    run.log_artifact(dev / "calibrator.json", "model")
                run.log_artifact(params, "results")
            return run.run_id

        good = lopo(_lopo_metrics(ndcg=0.75, p=0.01, ece=0.02))
        bad = lopo(_lopo_metrics(ndcg=0.60, p=0.5, ece=0.08), monotone=False)
        no_calibrator = lopo(_lopo_metrics(ndcg=0.7, p=0.01, ece=0.02), calibrator=False)

        models = root / "models"
        dataset = {
            "manifest_sha256": MANIFEST_SHA,
            "per_repo": [
                {"slug": "acme__alpha", "sha": "b" * 40, "passed": True, "rows": 120},
                {"slug": "acme__gone", "sha": "c" * 40, "passed": False, "rows": 0},
            ],
        }
        first = reg.register_run(good, models_dir=models, uri=uri, dataset=dataset, now=NOW)
        second = reg.register_run(bad, models_dir=models, uri=uri, now=NOW)
        yield Store(
            root=root,
            uri=uri,
            dev=dev,
            params=params,
            data=data,
            dataset_sha=sha,
            good=good,
            bad=bad,
            no_calibrator=no_calibrator,
            temporal=temporal,
            models=models,
            good_version=first.version,
            bad_version=second.version,
        )


@pytest.fixture
def models(store: Store, tmp_path: Path) -> Path:
    """A private copy of the registered models folder."""
    return Path(shutil.copytree(store.models, tmp_path / "models"))


def _lines(*rows: str) -> str:
    return "\n".join(rows) + "\n"


def _module(name: str, branches: int) -> str:
    body = [f"def {name}(x):"]
    for k in range(branches):
        body += [f"    if x > {k}:", f"        x = x - {k + 1}"]
    body += ["    return x", "", "", f"def {name}_twice(x):", f"    return {name}({name}(x))"]
    return _lines(*body)


def _golden_repo(path: Path, seed: int) -> str:
    repo = RepoBuilder(path)
    files = {f"pkg/m{i}.py": _module(f"f{i}", 1 + (i * (seed + 2)) % 6) for i in range(5)}
    files["pkg/__init__.py"] = "from pkg.m0 import f0\n"
    files["pkg/m1.py"] = "from pkg.m0 import f0\n\n\n" + files["pkg/m1.py"]
    repo.commit(files, "Initial import", T0)
    for day in range(1, 8):
        i = (day * (seed + 1)) % 5
        files[f"pkg/m{i}.py"] += f"\n\ndef extra_{day}(y):\n    return y + {day}\n"
        message = "Fix bug in helpers" if day % 3 == 0 else f"Add helper {day}"
        repo.commit({f"pkg/m{i}.py": files[f"pkg/m{i}.py"]}, message, T0 + timedelta(days=9 * day))
    return repo.head()


@pytest.fixture(scope="module")
def mirrors(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, tuple[reg.GoldenRepo, ...]]:
    root = tmp_path_factory.mktemp("mirrors")
    alpha = _golden_repo(root / "acme__alpha.git", 1)
    beta = _golden_repo(root / "acme__beta.git", 2)
    return root, (reg.GoldenRepo("acme/alpha", alpha), reg.GoldenRepo("acme/beta", beta))


def _golden_ok(version: str) -> dict[str, Any]:
    repo = {
        "name": "acme/alpha",
        "sha": "a" * 40,
        "passed": True,
        "errors": [],
        "files": 3,
        "priority_min": 0,
        "priority_max": 100,
        "top": [{"path": "pkg/m0.py", "band": "Critical", "priority": 100}],
        "seconds": 0.5,
    }
    return {"version": version, "passed": True, "repos": [repo]}


# ---------------------------------------------------------------------------------------------------------
# versions and tarballs


def test_parse_version() -> None:
    assert reg.parse_version("ranker-2026.10.1") == (2026, 10, 1)
    assert reg.parse_version("ranker-2027.01.12") == (2027, 1, 12)
    for bad in ("ranker-2026.13.1", "ranker-1999.01.1", "ranker-2026.10.0", "ranker-2026.1.1", "dev-1", ""):
        with pytest.raises(reg.RegistryError):
            reg.parse_version(bad)
    with pytest.raises(reg.RegistryError):
        reg.bundle_dir(Path("."), "../x")


def test_next_version(tmp_path: Path) -> None:
    assert reg.next_version(tmp_path / "missing", now=NOW) == "ranker-2026.10.1"
    assert reg.next_version(tmp_path, now=NOW) == "ranker-2026.10.1"
    (tmp_path / "ranker-2026.10.1").mkdir()
    (tmp_path / "ranker-2026.09.9").mkdir()
    (tmp_path / "ranker-2026.10.x").mkdir()
    (tmp_path / "junk").mkdir()
    (tmp_path / "ranker-2026.10.3.tar.gz").write_bytes(b"")
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "ranker-2026.10.4.json").write_text("{}", encoding="utf-8")
    (tmp_path / "registry" / "ranker-2026.10.99.txt").write_text("", encoding="utf-8")
    assert reg.next_version(tmp_path, now=NOW) == "ranker-2026.10.5"
    assert reg.next_version(tmp_path, now=datetime(2026, 11, 1, tzinfo=UTC)) == "ranker-2026.11.1"
    assert reg.next_version(tmp_path, now=datetime(2026, 9, 30, tzinfo=UTC)) == "ranker-2026.09.10"


def test_tarball_is_deterministic(tmp_path: Path) -> None:
    folder = tmp_path / "ranker-2026.10.7"
    folder.mkdir()
    (folder / "b.json").write_bytes(b'{"b": 1}\n')
    (folder / "a.txt").write_bytes(b"alpha\x00bytes")
    first = reg.write_tarball(folder, tmp_path / "one.tar.gz")
    (folder / "a.txt").touch()  # a new mtime must not change anything
    second = reg.write_tarball(folder, tmp_path / "two.tar.gz")
    assert first == second == file_sha256(tmp_path / "one.tar.gz")
    assert (tmp_path / "one.tar.gz").read_bytes() == (tmp_path / "two.tar.gz").read_bytes()
    raw = gzip.decompress((tmp_path / "one.tar.gz").read_bytes())
    assert (tmp_path / "one.tar.gz").read_bytes()[4:8] == b"\x00\x00\x00\x00"  # gzip mtime 0
    with tarfile.open(tmp_path / "one.tar.gz", "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        assert names == ["ranker-2026.10.7", "ranker-2026.10.7/a.txt", "ranker-2026.10.7/b.json"]
        assert all(m.mtime == 0 and m.uid == 0 and m.gid == 0 and m.uname == "" for m in members)
        assert [m.mode for m in members] == [0o755, 0o644, 0o644]
        tar.extractall(tmp_path / "out", filter="data")
    assert len(raw) % 512 == 0
    assert (tmp_path / "out" / "ranker-2026.10.7" / "a.txt").read_bytes() == b"alpha\x00bytes"
    assert reg.read_tarball(tmp_path / "one.tar.gz", root="ranker-2026.10.7") == {
        "a.txt": b"alpha\x00bytes",
        "b.json": b'{"b": 1}\n',
    }
    with pytest.raises(reg.RegistryError, match="top folder"):
        reg.read_tarball(tmp_path / "one.tar.gz", root="other")
    (folder / "sub").mkdir()
    with pytest.raises(reg.RegistryError, match="regular files"):
        reg.write_tarball(folder, tmp_path / "three.tar.gz")
    with pytest.raises(reg.RegistryError):
        reg.write_tarball(tmp_path / "missing", tmp_path / "four.tar.gz")


def test_read_tarball_rejects_unsafe_archives(tmp_path: Path) -> None:
    def build(name: str, members: list[tuple[str, bytes]]) -> Path:
        path = tmp_path / name
        with tarfile.open(path, "w:gz") as tar:
            for member, kind in members:
                info = tarfile.TarInfo(member)
                info.type = kind
                tar.addfile(info, None)
        return path

    cases = {
        "abs.tar.gz": [("/etc/x", tarfile.REGTYPE)],
        "dots.tar.gz": [("v/../x", tarfile.REGTYPE)],
        "link.tar.gz": [("v/x", tarfile.SYMTYPE)],
        "nested.tar.gz": [("v/a/b", tarfile.REGTYPE)],
        "two.tar.gz": [("v/a", tarfile.REGTYPE), ("w/b", tarfile.REGTYPE)],
    }
    for name, members in cases.items():
        with pytest.raises(reg.RegistryError):
            reg.read_tarball(build(name, members))
    (tmp_path / "junk.tar.gz").write_bytes(b"not a tarball")
    with pytest.raises(reg.RegistryError, match="readable"):
        reg.read_tarball(tmp_path / "junk.tar.gz")
    with pytest.raises(reg.RegistryError, match="not found"):
        reg.read_tarball(tmp_path / "missing.tar.gz")


# ---------------------------------------------------------------------------------------------------------
# metadata and model card


def _info(**changes: Any) -> reg.RunInfo:
    info = reg.RunInfo(
        run_id="r1",
        experiment_id="7",
        run_name="ranker-lopo",
        tracking_uri="file:///tmp/mlruns",
        model="ranker",
        params={
            "training_rows": "240",
            "training_queries": "12",
            "horizon_days": "365",
            "monotone": "true",
            "monotone_columns": json.dumps(MONOTONE),
        },
        metrics={**_lopo_metrics(ndcg=0.72, p=0.3, ece=0.01), "lopo_popt": float("nan")},
        tags={"dataset_sha256": "d" * 64, "manifest_sha256": "m" * 64, "code_git_sha": "feed"},
        temporal_run_id="t1",
        temporal_metrics={"temporal_ndcg_10": 0.73, "temporal_popt": float("inf")},
    )
    return replace(info, **changes)


TUNED = TunedParams(
    params={"objective": "rank:ndcg", "n_estimators": 145, "max_depth": 3, "learning_rate": 0.02},
    best_params={},
    best_value=None,
    fold_rounds=(),
    info={},
)


def test_collect_metrics_and_metadata() -> None:
    info = _info()
    metrics = reg.collect_metrics(info)
    assert set(metrics) == {"lopo", "temporal", "calibration"}
    assert metrics["lopo"]["ndcg_10"] == 0.72 and "popt" not in metrics["lopo"]
    assert metrics["temporal"] == {"ndcg_10": 0.73}
    assert metrics["calibration"]["ece"] == 0.01 and "query_ece" not in metrics["calibration"]
    meta = reg.bundle_metadata(info, TUNED, created_at=NOW, registry_code_sha="abc")
    assert meta["created_at"] == "2026-10-05T12:00:00Z"
    assert meta["training_rows"] == 240 and meta["training_queries"] == 12 and meta["horizon_days"] == 365
    assert meta["synthetic"] is False and meta["monotone"] is True
    assert meta["params"]["monotone_constraints"] == dict.fromkeys(MONOTONE, 1)
    assert meta["params"]["n_estimators"] == 145
    assert meta["dataset_sha256"] == "d" * 64 and meta["source"]["temporal_run_id"] == "t1"
    plain = reg.bundle_metadata(
        _info(params={"training_rows": "5", "monotone": "false", "monotone_columns": ""}),
        TUNED,
        created_at=NOW,
        registry_code_sha="abc",
    )
    assert "monotone_constraints" not in plain["params"] and plain["monotone"] is False
    assert plain["training_queries"] is None
    for params in (
        {"monotone": "false"},
        {"training_rows": "many"},
        {"training_rows": "5", "monotone": "true", "monotone_columns": '["nope"]'},
        {"training_rows": "5", "monotone": "true", "monotone_columns": "{"},
    ):
        with pytest.raises(reg.RegistryError):
            reg.bundle_metadata(_info(params=params), TUNED, created_at=NOW, registry_code_sha="abc")


def test_model_card() -> None:
    info = _info()
    meta = reg.bundle_metadata(info, TUNED, created_at=NOW, registry_code_sha="abc")
    dataset = {"per_repo": [{"slug": "acme__alpha", "sha": "b" * 40, "passed": True, "rows": 99}]}
    card = reg.model_card("ranker-2026.10.1", meta, info, dataset=dataset)
    for section in ("Intended use", "Model", "Training data", "Evaluation", "Limitations", "Provenance"):
        assert f"## {section}" in card
    assert card.startswith("# Entropiq risk model ranker-2026.10.1")
    assert "145 rounds, max depth 3, learning rate 0.0200" in card
    assert "Monotone increasing in: commits_decayed, churn_decayed" in card
    assert "acme__alpha at bbbbbbbbbbbb, 99 rows" in card
    assert "| LOPO | 0.720 [0.640, 0.800]" in card and "| temporal | 0.730 |" in card
    assert "Does not significantly beat the hotspot heuristic" in card and "p 0.3000" in card
    assert card.isascii() and chr(0x2014) not in card and chr(0x2013) not in card
    better = _info(metrics=_lopo_metrics(ndcg=0.8, p=0.01, ece=0.01))
    assert "Beats the hotspot heuristic" in reg.model_card("ranker-2026.10.1", meta, better)


# ---------------------------------------------------------------------------------------------------------
# registration


def test_register_run_builds_a_bundle(store: Store) -> None:
    version = store.good_version
    assert version == "ranker-2026.10.1" and store.bad_version == "ranker-2026.10.2"
    folder = store.models / version
    assert sorted(p.name for p in folder.iterdir()) == sorted(BUNDLE_FILES)
    bundle = load_bundle(folder)
    assert bundle.version == version
    assert (folder / "ranker.json").read_bytes() == (store.dev / "ranker.json").read_bytes()
    scores = np.linspace(-2, 2, 25)
    dev_cal = IsotonicCalibrator.from_json(json.loads((store.dev / "calibrator.json").read_text("utf-8")))
    for ours, theirs in zip(bundle.calibrator.predict(scores), dev_cal.predict(scores), strict=True):
        np.testing.assert_array_equal(ours, theirs)
    meta = bundle.metadata
    assert meta["synthetic"] is False and meta["training_rows"] == 240 and meta["model"] == "ranker"
    assert meta["metrics"]["lopo"]["ndcg_10_p_greater_hotspot"] == 0.01
    assert meta["metrics"]["temporal"]["ndcg_10"] == 0.74
    assert meta["source"]["temporal_run_id"] == store.temporal
    assert meta["registry_code_git_sha"] == "c0ffee"
    record = reg.read_record(store.models, version)
    assert record["sha256"] == file_sha256(store.models / f"{version}.tar.gz")
    assert record["files"]["metadata.json"] == file_sha256(folder / "metadata.json")
    assert record["source"]["run_id"] == store.good and record["promotions"] == []
    assert record["run_metrics"]["calibration_ece"] == 0.02
    card = (folder / "model_card.md").read_text(encoding="utf-8")
    assert "acme__alpha" in card and "acme__gone" not in card
    assert "acme__alpha" not in (store.models / store.bad_version / "model_card.md").read_text("utf-8")
    assert set(reg.read_tarball(store.models / f"{version}.tar.gz", root=version)) == set(BUNDLE_FILES)


def test_register_run_is_idempotent(store: Store, models: Path) -> None:
    again = reg.register_run(store.good, models_dir=models, uri=store.uri)
    assert not again.created and again.version == store.good_version
    rebuilt = reg.register_run(store.good, models_dir=models, uri=store.uri, force=True, now=NOW)
    assert rebuilt.created and rebuilt.version == store.good_version
    assert rebuilt.sha256 == file_sha256(models / f"{store.good_version}.tar.gz")
    assert reg.gate_integrity(models, rebuilt.version, reg.read_record(models, rebuilt.version)).passed
    with pytest.raises(reg.RegistryError, match="in production"):
        reg.register_run(
            store.good, models_dir=models, uri=store.uri, force=True, production_version=store.good_version
        )
    with pytest.raises(reg.RegistryError, match="already registered"):
        reg.register_run(store.good, models_dir=models, uri=store.uri, force=True, version="ranker-2026.10.9")
    with pytest.raises(reg.RegistryError, match="already exists"):
        reg.register_run(store.no_calibrator, models_dir=models, uri=store.uri, version=store.good_version)


@pytest.mark.parametrize(
    ("which", "message"),
    [("unknown", "not found"), ("temporal", "temporal run"), ("no_calibrator", "tdp-ml calibrate")],
)
def test_register_run_rejects_unsuitable_runs(store: Store, tmp_path: Path, which: str, message: str) -> None:
    run_id = {"unknown": "0" * 32, "temporal": store.temporal, "no_calibrator": store.no_calibrator}[which]
    with pytest.raises(reg.RegistryError, match=message):
        reg.register_run(run_id, models_dir=tmp_path / "models", uri=store.uri)
    assert not (tmp_path / "models").exists() or not any((tmp_path / "models").glob("ranker-*"))


def test_fetch_run_with_explicit_temporal_run(store: Store, tmp_path: Path) -> None:
    files = reg.fetch_run(store.good, tmp_path, uri=store.uri, temporal_run_id=store.temporal)
    assert files.info.temporal_run_id == store.temporal and files.params.name == "ranker_params.json"
    with pytest.raises(reg.RegistryError, match="not a FINISHED temporal run"):
        reg.fetch_run(store.good, tmp_path, uri=store.uri, temporal_run_id=store.bad)


# ---------------------------------------------------------------------------------------------------------
# golden


def test_load_golden_config(tmp_path: Path) -> None:
    shipped = reg.load_golden_config()
    assert [r.name for r in shipped] == ["pallets/click", "Kludex/starlette", "psf/requests"]
    assert all(len(r.sha) == 40 for r in shipped) and shipped[1].slug == "kludex__starlette"
    assert shipped[2].url == "https://github.com/psf/requests"
    sha = "a" * 40
    bad = {
        "sha.yaml": f"repos:\n  - {{name: a/b, sha: {sha[:-1]}}}\n",
        "dup.yaml": f"repos:\n  - {{name: a/b, sha: {sha}}}\n  - {{name: A/B, sha: {sha}}}\n",
        "key.yaml": f"repos:\n  - {{name: a/b, sha: {sha}, url: x}}\n",
        "top.yaml": f"repos:\n  - {{name: a/b, sha: {sha}}}\nother: 1\n",
        "empty.yaml": "repos: []\n",
        "name.yaml": f"repos:\n  - {{name: nope, sha: {sha}}}\n",
        "list.yaml": "- 1\n",
        "yaml.yaml": "repos: [\n",
    }
    for name, text in bad.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match=name):
            reg.load_golden_config(tmp_path / name)
    with pytest.raises(ConfigError):
        reg.load_golden_config(tmp_path / "missing.yaml")


def test_run_golden(store: Store, mirrors: tuple[Path, tuple[reg.GoldenRepo, ...]]) -> None:
    root, repos = mirrors
    broken = (reg.GoldenRepo("acme/alpha", "0" * 40), reg.GoldenRepo("acme/none", "1" * 40))
    report = reg.run_golden(store.dev, (*repos, *broken), mirrors=root, jobs=1, sha256="f" * 64, now=NOW)
    assert report["format"] == 1 and report["version"] == "dev-test" and report["bundle_sha256"] == "f" * 64
    assert report["created_at"] == "2026-10-05T12:00:00Z" and report["passed"] is False
    good, other, bad_sha, missing = report["repos"]
    for entry in (good, other):
        assert entry["passed"], entry["errors"]
        assert entry["files"] == 6 and 0 <= entry["priority_min"] <= entry["priority_max"] <= 100
        assert sum(entry["bands"].values()) == entry["files"]
        assert entry["additivity_max_error"] < reg.ADDITIVITY_TOL
        assert entry["top"][0]["reasons"] and len(entry["top"]) == 5
        assert entry["top"][0]["risk_low"] <= entry["top"][0]["risk"] <= entry["top"][0]["risk_high"]
    assert not bad_sha["passed"] and bad_sha["errors"] and bad_sha["files"] == 0
    assert missing["errors"] == [f"mirror not found: {(root / 'acme__none.git').as_posix()}"]
    gate = reg.gate_golden(report, "dev-test")
    assert not gate.passed and gate.value == 2 and "acme/none" in gate.detail
    passing = dict(report, repos=[good, other], passed=True)
    assert reg.gate_golden(passing, "dev-test").passed
    assert not reg.gate_golden(passing, "ranker-2026.10.1").passed


def test_check_result_flags_bad_results(
    store: Store, mirrors: tuple[Path, tuple[reg.GoldenRepo, ...]]
) -> None:
    from tdp_core.analyze import analyze
    from tdp_core.model.explain import Explainer

    root, repos = mirrors
    bundle = load_bundle(store.dev)
    explainer = Explainer(bundle)
    result = analyze(root / f"{repos[0].slug}.git", repos[0].sha, bundle, jobs=1)
    assert reg.check_result(result, repos[0], bundle, explainer)[0] == []
    first = result.files[0]
    broken = replace(
        result,
        sha="0" * 40,
        files=(replace(first, priority=101, risk=2.0, reasons=()), *result.files[1:]),
    )
    errors, _ = reg.check_result(broken, repos[0], bundle, explainer)
    assert any("analysed" in e for e in errors) and any("outside 0..100" in e for e in errors)
    assert any("risk" in e for e in errors) and any("no reason" in e for e in errors)
    shifted = replace(result, files=(replace(first, score=first.score + 1.0), *result.files[1:]))
    assert any("SHAP" in e for e in reg.check_result(shifted, repos[0], bundle, explainer)[0])
    fewer = replace(result, features=result.features.iloc[1:])
    assert any("feature rows" in e for e in reg.check_result(fewer, repos[0], bundle, explainer)[0])
    empty = replace(result, files=())
    assert reg.check_result(empty, repos[0], bundle, explainer) == (["no files were ranked"], None)


# ---------------------------------------------------------------------------------------------------------
# gates


def _meta(**metrics: dict[str, float]) -> dict[str, Any]:
    return {"synthetic": False, "training_rows": 100, "metrics": metrics}


def test_statistical_gates() -> None:
    good = _meta(lopo={"ndcg_10": 0.7, "hotspot_ndcg_10": 0.6, "ndcg_10_p_greater_hotspot": 0.01})
    hot = reg.gate_hotspot(good)
    assert hot.passed and hot.value == 0.01 and hot.waivable and "0.7000 vs hotspot 0.6000" in hot.detail
    assert not reg.gate_hotspot(_meta(lopo={"ndcg_10_p_greater_hotspot": 0.05})).passed
    assert not reg.gate_hotspot(_meta()).passed and reg.gate_hotspot(_meta()).value is None
    assert reg.gate_calibration(_meta(calibration={"ece": 0.05})).passed
    assert not reg.gate_calibration(_meta(calibration={"ece": 0.051})).passed
    missing = reg.gate_calibration({"metrics": "broken"})
    assert not missing.passed and "tdp-ml calibrate" in missing.detail
    real = reg.gate_real_model(good)
    assert real.passed and real.value == 100 and not real.waivable
    assert not reg.gate_real_model({"synthetic": True, "training_rows": 5}).passed
    for rows in (None, 0, True, "5"):
        assert not reg.gate_real_model({"synthetic": False, "training_rows": rows}).passed


def test_incumbent_gate(tmp_path: Path) -> None:
    candidate = _meta(lopo={"ndcg_10": 0.70})

    def incumbent(version: str, metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
        folder = tmp_path / version
        folder.mkdir()
        write_json(folder / "metadata.json", {"version": version, "metrics": metrics, **extra})
        return {"version": version, "uri": folder.as_posix(), "sha256": "e" * 64}

    assert (
        reg.gate_incumbent(candidate, "ranker-2026.10.2", None, tmp_path).detail == "no production model yet"
    )
    same = {"version": "ranker-2026.10.2", "uri": None, "sha256": None}
    assert reg.gate_incumbent(candidate, "ranker-2026.10.2", same, tmp_path).passed
    below = incumbent("ranker-2026.09.1", {"lopo": {"ndcg_10": 0.8, "ndcg_10_ci_low": 0.75}})
    result = reg.gate_incumbent(candidate, "ranker-2026.10.2", below, tmp_path)
    assert not result.passed and result.waivable and "CI low 0.7500" in result.detail
    within = incumbent("ranker-2026.09.2", {"lopo": {"ndcg_10": 0.74, "ndcg_10_ci_low": 0.66}})
    assert reg.gate_incumbent(candidate, "ranker-2026.10.2", within, tmp_path).passed
    point = incumbent("ranker-2026.09.3", {"lopo": {"ndcg_10": 0.71}})
    assert not reg.gate_incumbent(candidate, "ranker-2026.10.2", point, tmp_path).passed
    gone = {"version": "ranker-2026.08.1", "uri": "https://example.com/x.tar.gz", "sha256": "e" * 64}
    missing = reg.gate_incumbent(candidate, "ranker-2026.10.2", gone, tmp_path)
    assert not missing.passed and "not found on this machine" in missing.detail
    by_name = {"version": "ranker-2026.09.2", "uri": None, "sha256": None}  # found in models_dir/<version>
    assert reg.gate_incumbent(candidate, "ranker-2026.10.2", by_name, tmp_path).passed
    synthetic = incumbent("dev-1", {}, synthetic=True)
    assert reg.gate_incumbent(candidate, "ranker-2026.10.2", synthetic, tmp_path).passed
    empty = incumbent("ranker-2026.09.4", {})
    assert reg.gate_incumbent(candidate, "ranker-2026.10.2", empty, tmp_path).passed
    assert not reg.gate_incumbent(_meta(), "ranker-2026.10.2", within, tmp_path).passed


def test_leakage_gate(store: Store, tmp_path: Path) -> None:
    layout = DataLayout(store.data)
    meta = {"dataset_sha256": store.dataset_sha}
    good = reg.gate_leakage(layout, meta)
    assert good.passed and good.value == 10 and not good.waivable
    original = json.loads(layout.run_record("check-leakage").read_text(encoding="utf-8"))

    def variant(name: str, **changes: Any) -> Path:
        return write_json(tmp_path / f"{name}.json", {**original, **changes})

    queries = original["queries"]
    cases = {
        "failed": variant("failed", passed=False),
        "few": variant("few", checked=9, queries=queries[:9]),
        "rows": variant("rows", queries=[{**queries[0], "rows": 1}, *queries[1:]]),
        "query": variant("query", queries=[{**queries[0], "passed": False}, *queries[1:]]),
        "shape": write_json(tmp_path / "shape.json", [1, 2]),
    }
    for name, path in cases.items():
        assert not reg.gate_leakage(layout, meta, report=path).passed, name
    missing = reg.gate_leakage(layout, meta, report=tmp_path / "none.json")
    assert not missing.passed and "pytest -m leakage" in missing.detail
    changed = reg.gate_leakage(layout, {"dataset_sha256": "0" * 64})
    assert not changed.passed and "training dataset" in changed.detail
    few_queries = reg.gate_leakage(layout, meta, min_queries=11)
    assert not few_queries.passed  # 12 queries in the dataset, so 11 are needed


def test_integrity_gate(store: Store, models: Path) -> None:
    version = store.good_version
    record = reg.read_record(models, version)
    good = reg.gate_integrity(models, version, record)
    assert good.passed and good.value == 7 and not good.waivable
    assert not reg.gate_integrity(models, version, {**record, "sha256": "0" * 64}).passed
    card = models / version / "model_card.md"
    card.write_text(card.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    tampered = reg.gate_integrity(models, version, record)
    assert not tampered.passed and "model_card.md" in tampered.detail
    (models / version / "extra.txt").write_text("x", encoding="utf-8")
    assert "differ from folder files" in reg.gate_integrity(models, version, record).detail
    (models / f"{version}.tar.gz").unlink()
    assert "tarball not found" in reg.gate_integrity(models, version, record).detail


# ---------------------------------------------------------------------------------------------------------
# promotion


def _promote(store: Store, models: Path, tmp_path: Path, version: str, **kwargs: Any) -> reg.PromotionOutcome:
    options: dict[str, Any] = {
        "models_dir": models,
        "results_dir": tmp_path / "results",
        "layout": DataLayout(store.data),
        "production_files": [tmp_path / "repo" / "production.json", models / "production.json"],
        "golden": (),
        "mirrors": tmp_path / "mirrors",
        "golden_report": _golden_ok(version),
        "now": NOW,
    }
    options.update(kwargs)
    return reg.promote(version, **options)


def test_promote_publishes_when_every_gate_passes(store: Store, models: Path, tmp_path: Path) -> None:
    version = store.good_version
    outcome = _promote(store, models, tmp_path, version)
    assert outcome.promoted and outcome.accepted and not outcome.failed and outcome.previous is None
    assert [g.name for g in outcome.gates] == list(reg.GATES)
    record = reg.read_record(models, version)
    for path in (tmp_path / "repo" / "production.json", models / "production.json"):
        pointer = json.loads(path.read_text(encoding="utf-8"))
        assert list(pointer) == ["version", "uri", "sha256"]
        assert pointer["version"] == version and pointer["sha256"] == record["sha256"]
        assert Path(pointer["uri"]) == (models / version).absolute()
        assert path.read_text(encoding="utf-8").endswith("}\n")
    assert reg.read_pointer(models / "production.json") == outcome.pointer
    assert len(outcome.written) == 2 and len(record["promotions"]) == 1
    assert record["promotions"][0]["waivers"] == [] and record["promotions"][0]["pointer"] == outcome.pointer
    report = json.loads(outcome.report.read_text(encoding="utf-8"))
    assert report["promoted"] is True and report["failed"] == [] and len(report["gates"]) == 7

    # A worse model is refused against the incumbent; re-promoting the same version passes.
    refused = _promote(store, models, tmp_path, store.bad_version)
    assert {g.name for g in refused.failed} == {"incumbent", "hotspot", "calibration"}
    assert refused.previous == outcome.pointer
    again = _promote(store, models, tmp_path, version)
    assert again.promoted and len(reg.read_record(models, version)["promotions"]) == 2

    # A later dry run of the same version keeps the report of the real promotion.
    dry = _promote(store, models, tmp_path, version, dry_run=True)
    assert dry.report == tmp_path / "results" / f"promotion_{version}_dry_run.json"
    assert json.loads(again.report.read_text(encoding="utf-8"))["promoted"] is True
    assert again.report.name == f"promotion_{version}.json"


def test_promote_dry_run_and_refusal_write_no_pointer(store: Store, models: Path, tmp_path: Path) -> None:
    pointer = tmp_path / "repo" / "production.json"
    pointer.parent.mkdir()
    pointer.write_text('{"version": null, "uri": null, "sha256": null}\n', encoding="utf-8")
    dry = _promote(store, models, tmp_path, store.good_version, dry_run=True)
    assert dry.accepted and not dry.promoted and dry.written == ()
    assert reg.read_record(models, store.good_version)["promotions"] == []
    refused = _promote(store, models, tmp_path, store.bad_version)
    assert not refused.promoted and {g.name for g in refused.failed} == {"hotspot", "calibration"}
    assert pointer.read_text(encoding="utf-8") == '{"version": null, "uri": null, "sha256": null}\n'
    assert not (models / "production.json").exists()
    report = json.loads(refused.report.read_text(encoding="utf-8"))
    assert report["promoted"] is False and report["failed"] == ["hotspot", "calibration"]
    assert refused.report.name == f"promotion_{store.bad_version}_refused.json"
    assert dry.report.name == f"promotion_{store.good_version}_dry_run.json"
    bad_golden = _promote(store, models, tmp_path, store.good_version, golden_report={"version": "x"})
    assert [g.name for g in bad_golden.failed] == ["golden"]


def test_promote_with_waivers(store: Store, models: Path, tmp_path: Path) -> None:
    reason = "Ties the heuristic; adds calibrated risk."
    outcome = _promote(
        store,
        models,
        tmp_path,
        store.bad_version,
        waive=["calibration", "hotspot", "incumbent"],
        reason=reason,
    )
    assert outcome.promoted and outcome.waivers == ("incumbent", "hotspot", "calibration")
    assert {g.name for g in outcome.gates if g.waived} == {"hotspot", "calibration"}
    promotion = reg.read_record(models, store.bad_version)["promotions"][0]
    assert promotion["reason"] == reason and promotion["waivers"] == ["incumbent", "hotspot", "calibration"]
    assert [g["name"] for g in promotion["gates"] if g["waived"]] == ["hotspot", "calibration"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"waive": ["integrity"], "reason": "x"}, "cannot be waived"),
        ({"waive": ["speed"], "reason": "x"}, "unknown gate"),
        ({"waive": ["hotspot"]}, "needs --reason"),
        ({"waive": ["hotspot"], "reason": "two\nlines"}, "one line"),
        ({"reason": "why"}, "only used with --waive"),
        ({"production_files": []}, "no production pointer"),
    ],
)
def test_promote_rejects_bad_input(
    store: Store, models: Path, tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(reg.RegistryError, match=message):
        _promote(store, models, tmp_path, store.good_version, **kwargs)


def test_promote_unknown_version(store: Store, models: Path, tmp_path: Path) -> None:
    with pytest.raises(reg.RegistryError, match="not registered"):
        _promote(store, models, tmp_path, "ranker-2026.10.9")
    (models / "production.json").write_text("[]", encoding="utf-8")
    with pytest.raises(reg.RegistryError, match="JSON object"):
        reg.read_pointer(models / "production.json")


def test_promote_end_to_end_with_golden(
    store: Store, models: Path, tmp_path: Path, mirrors: tuple[Path, tuple[reg.GoldenRepo, ...]]
) -> None:
    root, repos = mirrors
    outcome = _promote(
        store,
        models,
        tmp_path,
        store.good_version,
        golden=repos[:1],
        mirrors=root,
        golden_report=None,
        jobs=1,
    )
    assert outcome.promoted, [g.detail for g in outcome.failed]
    assert outcome.golden_report == tmp_path / "results" / f"golden_{store.good_version}.json"
    golden = json.loads(outcome.golden_report.read_text(encoding="utf-8"))
    assert golden["passed"] and golden["bundle_sha256"] == outcome.sha256


def test_default_production_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TDP_MODELS_DIR", str(tmp_path))
    files = reg.default_production_files()
    assert files[-1] == tmp_path / "production.json"
    assert files[0].parent.name == "models" and files[0].name == "production.json" and len(files) == 2


# ---------------------------------------------------------------------------------------------------------
# commands


def _cli() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        """Test app."""

    register(app)
    return app


@pytest.fixture
def cli(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Callable[..., Result], Path]]:
    results = tmp_path / "results"
    results.mkdir()
    write_json(
        results / "ranker_lopo_run.json", {"run_id": store.good, "protocol": "lopo", "model": "ranker"}
    )
    models = tmp_path / "models"
    monkeypatch.setenv("TDP_MODELS_DIR", str(models))
    monkeypatch.setenv("TDP_RESULTS_DIR", str(results))
    monkeypatch.setenv("TDP_DATA_DIR", str(store.data))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store.uri)
    monkeypatch.setenv("TDP_CODE_SHA", "c0ffee")
    pointer = tmp_path / "repo" / "production.json"
    monkeypatch.setattr(reg, "default_production_files", lambda: [pointer])

    def fake_golden(bundle_path: Path, repos: Any, **_: Any) -> dict[str, Any]:
        return _golden_ok(load_bundle(bundle_path).version)

    monkeypatch.setattr(reg, "run_golden", fake_golden)
    app = _cli()
    runner = CliRunner()

    def invoke(*args: str) -> Result:
        return runner.invoke(app, list(args))

    yield invoke, tmp_path


def test_cli_register_golden_promote(
    cli: tuple[Callable[..., Result], Path], store: Store, mirrors: tuple[Path, tuple[reg.GoldenRepo, ...]]
) -> None:
    invoke, tmp_path = cli
    result = invoke("register")
    assert result.exit_code == 0, result.output
    version = reg.next_version(tmp_path / "models-none")
    assert f"registered {version} from run {store.good} (ranker)" in result.output
    assert "LOPO NDCG@10 0.750 [0.670, 0.830], hotspot p 0.010, ECE 0.0200" in result.output
    again = invoke("register")
    assert again.exit_code == 0 and f"already registered as {version}" in again.output
    other = invoke("register", "--run-id", store.bad)
    assert other.exit_code == 0 and "ranker" in other.output
    bad_version = other.output.split()[1]

    report = tmp_path / "golden.json"
    golden = invoke("golden", "--version", version, "--out", str(report))
    assert golden.exit_code == 0, golden.output
    assert "golden: passed" in golden.output and json.loads(report.read_text("utf-8"))["passed"]
    assert invoke("golden").exit_code == 2
    assert invoke("golden", "--bundle", str(tmp_path / "nothing")).exit_code == 2

    pointer = tmp_path / "repo" / "production.json"
    dry = invoke("promote", "--version", version, "--dry-run", "--production-file", str(pointer))
    assert dry.exit_code == 0, dry.output
    assert "Dry run: all gates pass" in dry.output and not pointer.exists()
    refused = invoke("promote", "--version", bad_version, "--production-file", str(pointer))
    assert (
        refused.exit_code == 1 and "PROMOTION REFUSED: failed gates: hotspot, calibration" in refused.output
    )
    assert "FAIL" in refused.output and not pointer.exists()
    assert invoke("promote", "--version", version, "--waive", "integrity", "--reason", "x").exit_code == 2
    done = invoke("promote", "--version", version)
    assert done.exit_code == 0, done.output
    assert f"Promoted {version}: wrote {pointer.as_posix()}" in done.output
    assert json.loads(pointer.read_text("utf-8"))["version"] == version
    forced = invoke("register", "--force")
    assert forced.exit_code == 2 and "in production" in forced.output


def test_cli_golden_with_real_analysis(
    cli: tuple[Callable[..., Result], Path],
    store: Store,
    mirrors: tuple[Path, tuple[reg.GoldenRepo, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke, tmp_path = cli
    root, repos = mirrors
    monkeypatch.setattr(reg, "run_golden", _REAL_RUN_GOLDEN)
    config = tmp_path / "golden.yaml"
    config.write_text(f"repos:\n  - {{name: {repos[0].name}, sha: {repos[0].sha}}}\n", encoding="utf-8")
    common = ("--bundle", str(store.dev), "--config", str(config), "--jobs", "1")
    result = invoke("golden", *common, "--mirrors", str(root))
    assert result.exit_code == 0, result.output
    assert f"acme/alpha {repos[0].sha[:7]}: 6 files" in result.output
    assert (tmp_path / "results" / "golden_dev-test.json").is_file()
    failed = invoke("golden", *common, "--mirrors", str(tmp_path / "nowhere"))
    assert failed.exit_code == 1 and "FAILED: mirror not found" in failed.output
    (tmp_path / "bad.yaml").write_text("repos: []\n", encoding="utf-8")
    assert invoke("golden", "--bundle", str(store.dev), "--config", str(tmp_path / "bad.yaml")).exit_code == 2


_REAL_RUN_GOLDEN = reg.run_golden

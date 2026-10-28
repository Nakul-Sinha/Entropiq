"""Tests of the risk calibration library and the ``tdp-ml calibrate`` command (synthetic data only)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner, Result

import tdp_ml.calibrate as cal
from tdp_core.model.calibration import IsotonicCalibrator
from tdp_ml import tracking as trk
from tdp_ml.commands.calibrate import register
from tdp_ml.eval import auc_pr, brier, ece, reliability_table, temporal_split

START = pd.Timestamp("2018-01-01", tz="UTC")
_MLFLOW_ENV = ("MLFLOW_ALLOW_FILE_STORE", "MLFLOW_DISABLE_TELEMETRY", "MLFLOW_DISABLE_AGENT_HINT")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _scores_frame(
    n_repos: int = 6,
    n_snapshots: int = 5,
    n_files: int = 200,
    *,
    offsets: bool = False,
    seed: int = 0,
    model: str = "ranker",
) -> pd.DataFrame:
    """Score rows in the layout of ``<model>_oof.parquet`` with a known true probability.

    Labels are ``Bernoulli(sigmoid(2 * score - 2 + offset))``; ``offset`` is 0, or a per-repository draw
    when ``offsets`` is true. The true probability is kept in an extra column ``p_true``.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for r in range(n_repos):
        offset = rng.normal(0.0, 1.5) if offsets else 0.0
        base = rng.normal(0.0, 1.0, size=n_files)
        for s in range(n_snapshots):
            score = base + rng.normal(0.0, 0.3, size=n_files)
            p_true = _sigmoid(2.0 * score - 2.0 + offset)
            label = rng.random(n_files) < p_true
            defects = np.where(label, rng.integers(1, 4, size=n_files), 0)
            parts.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "protocol": "lopo",
                        "split": f"synthetic__repo{r:02d}",
                        "qid": r * n_snapshots + s,
                        "repo": f"synthetic__repo{r:02d}",
                        "snapshot_idx": s,
                        "snapshot_time": START + pd.Timedelta(days=23 * r + 182 * s),
                        "path": [f"src/mod_{i:03d}.py" for i in range(n_files)],
                        "sloc": rng.integers(10, 500, size=n_files),
                        "grade": np.minimum(defects, 4),
                        "pain": defects + rng.random(n_files),
                        "latent_defects": defects,
                        "has_latent_defect": label,
                        "is_test": s == n_snapshots - 1,
                        "score": score,
                        "p_true": p_true,
                    }
                )
            )
    out = pd.concat(parts, ignore_index=True)
    for col in ("model", "protocol", "split", "repo", "path"):
        out[col] = out[col].astype("string")
    return out.astype(
        {
            "qid": "int64",
            "snapshot_idx": "int16",
            "snapshot_time": "datetime64[us, UTC]",
            "sloc": "float32",
            "grade": "int8",
            "pain": "float32",
            "latent_defects": "int32",
            "has_latent_defect": "bool",
            "is_test": "bool",
            "score": "float64",
        }
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _scores_frame()


@pytest.fixture(scope="module")
def outcome(frame: pd.DataFrame) -> cal.CalibrationOutcome:
    temporal = frame[frame["is_test"]].assign(score=lambda d: d["score"] * 1.1)
    return cal.calibrate(frame, temporal_scores=temporal, n_boot=20, ci_boot=30)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _MLFLOW_ENV:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


# ---------------------------------------------------------------------------------------------------------
# inputs


def test_check_scores_accepts_valid_frames(frame: pd.DataFrame) -> None:
    cal.check_scores(frame)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.drop(columns=["score", "sloc"]), "missing columns ['sloc', 'score']"),
        (lambda d: d.iloc[:0], "no rows"),
        (lambda d: d.assign(qid=d["qid"].astype("float64")), "integer dtype"),
        (lambda d: d.assign(repo=d["repo"].where(d.index != 3)), "'repo' contains nulls"),
        (lambda d: d.assign(score=d["score"].where(d.index != 3)), "'score' contains nulls"),
        (lambda d: d.assign(score=d["score"].where(d.index != 3, np.inf)), "finite"),
        (lambda d: d.assign(has_latent_defect=np.where(d.index == 3, 2, 0)), "only 0 and 1"),
        (lambda d: pd.concat([d, d.iloc[[5]]]), "duplicate (qid, path)"),
    ],
)
def test_check_scores_errors(
    frame: pd.DataFrame, change: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=r"^oof: ") as info:
        cal.check_scores(change(frame.copy()), name="oof")
    assert message in str(info.value)


def test_load_scores_round_trip(frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "s.parquet"
    frame.sample(frac=1.0, random_state=1).to_parquet(path, index=False)
    loaded = cal.load_scores(path)
    assert isinstance(loaded.index, pd.RangeIndex)
    expected = frame.sort_values(["qid", "path"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(loaded, expected, check_dtype=False)
    with pytest.raises(FileNotFoundError):
        cal.load_scores(tmp_path / "missing.parquet")
    frame.drop(columns=["score"]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        cal.load_scores(path)


# ---------------------------------------------------------------------------------------------------------
# fitting


def test_row_bootstrap_equals_core_fit(frame: pd.DataFrame) -> None:
    s, y = frame["score"].to_numpy(), frame["has_latent_defect"].to_numpy()
    ours = cal.fit_calibrator(s, y, bootstrap="row", n_boot=15, seed=3)
    assert ours.to_json() == IsotonicCalibrator.fit(s, y, n_boot=15, seed=3).to_json()


def test_repo_bootstrap(frame: pd.DataFrame) -> None:
    s, y, g = frame["score"].to_numpy(), frame["has_latent_defect"].to_numpy(), frame["repo"]
    fit = cal.fit_calibrator(s, y, groups=g, n_boot=25, seed=1)
    plain = IsotonicCalibrator.fit(s, y, n_boot=0)
    np.testing.assert_array_equal(fit.x, plain.x)
    np.testing.assert_array_equal(fit.y, plain.y)
    assert np.all(fit.y_low <= fit.y) and np.all(fit.y <= fit.y_high)
    assert np.all(np.diff(fit.y_low) >= 0) and np.all(np.diff(fit.y_high) >= 0)
    assert np.any(fit.y_high > fit.y_low)
    again = cal.fit_calibrator(s, y, groups=g, n_boot=25, seed=1)
    assert again.to_json() == fit.to_json()
    other = cal.fit_calibrator(s, y, groups=g, n_boot=25, seed=2)
    assert other.to_json() != fit.to_json()
    zero = cal.fit_calibrator(s, y, groups=g, n_boot=0)
    assert zero.to_json() == plain.to_json()
    single = cal.fit_calibrator(s, y, groups=np.full(s.size, "one"), n_boot=10)
    np.testing.assert_array_equal(single.y_low, single.y)
    np.testing.assert_array_equal(single.y_high, single.y)


def test_repo_bootstrap_errors(frame: pd.DataFrame) -> None:
    s, y = frame["score"].to_numpy(), frame["has_latent_defect"].to_numpy()
    with pytest.raises(ValueError, match="needs groups"):
        cal.fit_calibrator(s, y)
    with pytest.raises(ValueError, match="groups has"):
        cal.fit_calibrator(s, y, groups=frame["repo"].iloc[:-1])
    nulls = frame["repo"].astype(object).where(frame.index != 0, None)
    with pytest.raises(ValueError, match="nulls"):
        cal.fit_calibrator(s, y, groups=nulls)
    with pytest.raises(ValueError, match="bootstrap must be one of"):
        cal.fit_calibrator(s, y, groups=frame["repo"], bootstrap="file")
    with pytest.raises(ValueError, match="n_boot"):
        cal.fit_calibrator(s, y, groups=frame["repo"], n_boot=-1)
    with pytest.raises(ValueError, match="n_boot"):
        cal.fit_calibrator(s, y, groups=frame["repo"], n_boot=True)


def test_repo_bands_are_wider_on_clustered_data() -> None:
    data = _scores_frame(n_repos=8, n_snapshots=3, n_files=150, offsets=True, seed=5)
    s, y = data["score"].to_numpy(), data["has_latent_defect"].to_numpy()
    by_repo = cal.fit_calibrator(s, y, groups=data["repo"], n_boot=40)
    by_row = cal.fit_calibrator(s, y, bootstrap="row", n_boot=40)

    def width(c: IsotonicCalibrator) -> float:
        _, low, high = c.predict(s)
        return float(np.mean(high - low))

    assert width(by_repo) > 1.5 * width(by_row)


# ---------------------------------------------------------------------------------------------------------
# metrics


def test_wilson_interval() -> None:
    low, high = cal.wilson_interval([0, 5, 10, 0], [10, 10, 10, 0])
    np.testing.assert_allclose(low[:3], [0.0, 0.2366, 0.7225], atol=1e-4)
    np.testing.assert_allclose(high[:3], [0.2775, 0.7634, 1.0], atol=1e-4)
    assert math.isnan(low[3]) and math.isnan(high[3])
    lo, hi = cal.wilson_interval(5, 10)
    assert float(lo) == pytest.approx(0.2366, abs=1e-4) and float(hi) == pytest.approx(0.7634, abs=1e-4)
    with pytest.raises(ValueError, match="trials"):
        cal.wilson_interval(0, -1)
    with pytest.raises(ValueError, match="successes"):
        cal.wilson_interval(11, 10)
    with pytest.raises(ValueError, match="successes"):
        cal.wilson_interval(-1, 10)
    with pytest.raises(ValueError, match="z must be"):
        cal.wilson_interval(1, 10, z=0)


def test_reliability_matches_eval_table() -> None:
    prob = np.array([0.05, 0.07, 0.31, 0.33, 0.35, 0.95])
    y = np.array([0, 0, 1, 0, 1, 1])
    table = cal.reliability(prob, y, bins=10)
    assert list(table.columns) == list(cal.RELIABILITY_COLUMNS[1:])
    pd.testing.assert_frame_equal(
        table[list(reliability_table(prob, y, 10).columns)], reliability_table(prob, y)
    )
    filled = table[table["count"] > 0]
    assert np.all(filled["ci_low"] <= filled["frac_positive"])
    assert np.all(filled["frac_positive"] <= filled["ci_high"])
    empty = table[table["count"] == 0]
    assert len(empty) == 7 and empty["ci_low"].isna().all() and empty["ci_high"].isna().all()


def test_pooled_metrics(frame: pd.DataFrame) -> None:
    p, y = frame["p_true"].to_numpy(), frame["has_latent_defect"].to_numpy()
    groups = frame["repo"]
    base = float(y.mean())
    out = cal.pooled_metrics(p, y, groups=groups, base_rate=base, n_boot=50)
    assert out["brier"] == brier(p, y) and out["ece"] == ece(p, y) and out["auc_pr"] == auc_pr(p, y)
    for m in ("brier", "ece", "auc_pr"):
        assert out[f"{m}_ci_low"] <= out[f"{m}_ci_high"]
    assert out["brier_base_rate"] == pytest.approx(brier(np.full(y.size, base), y))
    assert out["brier_skill"] == pytest.approx(1 - out["brier"] / out["brier_base_rate"])
    assert all(type(v) is float for v in out.values())
    assert cal.pooled_metrics(p, y, groups=groups, base_rate=base, n_boot=50) == out
    assert out["ece"] < 0.06
    skewed = cal.pooled_metrics(np.sqrt(p), y, groups=groups, base_rate=base, n_boot=10)
    assert skewed["ece"] > out["ece"] + 0.1
    with pytest.raises(ValueError, match="n_boot"):
        cal.pooled_metrics(p, y, groups=groups, base_rate=base, n_boot=0)
    with pytest.raises(ValueError, match="base_rate"):
        cal.pooled_metrics(p, y, groups=groups, base_rate=1.5)


def test_pooled_metrics_without_positives() -> None:
    out = cal.pooled_metrics([0.1, 0.2, 0.3], [0, 0, 0], groups=["a", "a", "b"], base_rate=0.0, n_boot=5)
    assert math.isnan(out["auc_pr"]) and math.isnan(out["auc_pr_ci_low"])
    assert math.isnan(out["brier_skill"])


# ---------------------------------------------------------------------------------------------------------
# procedure


def test_calibrate_uses_the_embargo(frame: pd.DataFrame, outcome: cal.CalibrationOutcome) -> None:
    sorted_frame = frame.sort_values(["qid", "path"]).reset_index(drop=True)
    train_idx, test_idx = temporal_split(sorted_frame, 365)
    assert outcome.fit_rows.rows == len(train_idx)
    assert outcome.primary.rows.rows == len(test_idx)
    assert outcome.final_rows.rows == len(frame)
    flipped = sorted_frame.copy()
    flipped.loc[test_idx, "has_latent_defect"] = ~flipped.loc[test_idx, "has_latent_defect"]
    other = cal.calibrate(flipped, n_boot=20, ci_boot=30)
    assert other.eval_calibrator.to_json() == outcome.eval_calibrator.to_json()
    assert other.primary.metrics["brier"] != outcome.primary.metrics["brier"]
    rows = frame[frame["is_test"]]
    reference = cal.fit_calibrator(
        frame["score"], frame["has_latent_defect"], groups=frame["repo"], n_boot=20
    ).to_json()
    assert outcome.calibrator.to_json() == reference
    assert outcome.primary.rows.positive_rate == pytest.approx(rows["has_latent_defect"].mean())


def test_calibrate_horizon_zero_uses_every_older_snapshot(frame: pd.DataFrame) -> None:
    out = cal.calibrate(frame, horizon_days=0, n_boot=0, ci_boot=5)
    assert out.fit_rows.rows == int((frame["snapshot_idx"] < 4).sum())


def test_calibrate_errors(frame: pd.DataFrame) -> None:
    single = _scores_frame(n_snapshots=1, n_files=20)
    with pytest.raises(ValueError, match="no rows to fit on"):
        cal.calibrate(single, n_boot=0, ci_boot=5)
    with pytest.raises(ValueError, match="model column"):
        cal.calibrate(frame, model="ranker_monotone", n_boot=0, ci_boot=5)
    with pytest.raises(ValueError, match="temporal_scores: the model column"):
        cal.calibrate(frame, temporal_scores=frame.assign(model="other"), n_boot=0, ci_boot=5)
    with pytest.raises(ValueError, match="model name"):
        cal.calibrate(frame, model="Bad-Name")
    with pytest.raises(ValueError, match="ci_boot"):
        cal.calibrate(frame, ci_boot=0)
    with pytest.raises(ValueError, match="bootstrap"):
        cal.calibrate(frame, bootstrap="x")


def test_evaluation_sets(frame: pd.DataFrame, outcome: cal.CalibrationOutcome) -> None:
    assert [e.name for e in outcome.evaluations] == ["oof", "temporal_model"]
    for e in outcome.evaluations:
        assert tuple(e.metrics) == cal.SET_METRICS
        assert tuple(e.reliability.columns) == cal.RELIABILITY_COLUMNS
        assert set(e.reliability["set"]) == {e.name}
        assert 0.0 <= e.metrics["ece"] <= 1.0
        assert e.metrics["query_ece"] >= 0.0
    assert outcome.primary is outcome.evaluations[0]
    assert outcome.meets_target == (outcome.primary.metrics["ece"] <= cal.ECE_TARGET)
    alone = cal.calibrate(frame, n_boot=0, ci_boot=5)
    assert [e.name for e in alone.evaluations] == ["oof"]
    assert set(outcome.final_stats) == set(cal.FINAL_STATS)
    assert outcome.settings["bootstrap"] == "repo" and outcome.settings["target_ece"] == 0.05


# ---------------------------------------------------------------------------------------------------------
# outputs


def _strict(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value}")


def test_write_outputs(outcome: cal.CalibrationOutcome, tmp_path: Path) -> None:
    provenance = {"created_at": "2026-01-01T00:00:00Z", "scores": {"path": "x", "sha256": "ab"}}
    paths = cal.write_outputs(outcome, tmp_path, provenance=provenance)
    assert paths == cal.output_paths(tmp_path, "ranker")
    shipped = IsotonicCalibrator.from_json(json.loads(paths["calibrator"].read_text(encoding="utf-8")))
    assert shipped.to_json() == outcome.calibrator.to_json()
    record = json.loads(paths["record"].read_text(encoding="utf-8"), parse_constant=_strict)
    expected = {"format", "created_at", "model", "provenance", "settings", "target_ece", "meets_target"}
    assert expected | {"fit", "evaluation", "calibrator", "seconds"} == set(record)
    assert record["created_at"] == "2026-01-01T00:00:00Z"
    assert record["evaluation"]["oof"]["ece"] == outcome.primary.metrics["ece"]
    assert record["calibrator"]["thresholds"] == outcome.calibrator.x.size
    assert record["fit"]["rows"] == outcome.fit_rows.rows
    tables = pd.read_parquet(paths["reliability"])
    assert tuple(tables.columns) == cal.RELIABILITY_COLUMNS and len(tables) == 20
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "ECE" in summary and "ECE target 0.05" in summary
    assert chr(0x2014) not in summary and chr(0x2013) not in summary
    assert paths["figure"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert paths["figure"] == tmp_path / "figures" / "reliability.png"
    assert not list(tmp_path.rglob("*.tmp*"))


def test_output_paths() -> None:
    root = Path("r")
    assert (
        cal.output_paths(root, "ranker_monotone")["figure"]
        == root / "figures" / "ranker_monotone_reliability.png"
    )
    assert cal.output_paths(root, "ranker")["calibrator"] == root / "ranker_calibrator.json"
    for bad in ("Ranker", "1x", "a-b", "", "../x"):
        with pytest.raises(ValueError, match="model name"):
            cal.output_paths(root, bad)


def test_record_nan_becomes_null(outcome: cal.CalibrationOutcome) -> None:
    record = cal.calibration_record(outcome, provenance={"value": float("nan"), "n": np.int64(3)})
    assert record["provenance"] == {"value": None, "n": 3}
    json.dumps(record, allow_nan=False)


def test_calibrator_fits_a_bundle(outcome: cal.CalibrationOutcome, tmp_path: Path) -> None:
    from tdp_core.model.bundle import load_bundle, make_dev_bundle

    bundle = make_dev_bundle(tmp_path / "dev")
    (bundle / "calibrator.json").write_text(json.dumps(outcome.calibrator.to_json()), encoding="utf-8")
    assert load_bundle(bundle).calibrator.to_json() == outcome.calibrator.to_json()


def test_run_payload(outcome: cal.CalibrationOutcome) -> None:
    metrics, tags = cal.run_payload(outcome, provenance={"scores": {"sha256": "ab"}, "code_git_sha": "c0"})
    assert metrics["calibration_ece"] == outcome.primary.metrics["ece"]
    assert "calibration_temporal_model_ece" in metrics
    assert metrics["calibration_fit_rows"] == outcome.fit_rows.rows
    assert all(math.isfinite(v) for v in metrics.values())
    assert tags["calibration_scores_sha256"] == "ab" and tags["calibration_code_git_sha"] == "c0"
    assert tags["calibration_meets_target"] in ("true", "false")
    assert tags["calibration_bootstrap"] == "repo" and tags["calibration_n_boot"] == "20"


# ---------------------------------------------------------------------------------------------------------
# MLflow


def _finished_run(uri: str, tmp_path: Path) -> str:
    note = tmp_path / "params.json"
    note.write_text("{}", encoding="utf-8")
    with trk.mlflow_run("calibration-test", "ranker-lopo", uri=uri) as run:
        run.log_params({"max_depth": 3})
        run.log_artifact(note, "results")
    return run.run_id


def test_log_to_run(tmp_path: Path, clean_env: None) -> None:
    from mlflow.tracking import MlflowClient

    uri = (tmp_path / "mlruns").as_uri()
    run_id = _finished_run(uri, tmp_path)
    client = MlflowClient(tracking_uri=uri)
    before = client.get_run(run_id)
    source = tmp_path / "ranker_calibrator.json"
    source.write_text('{"x": [0.0]}', encoding="utf-8")
    info = cal.find_run(run_id, uri=uri)
    assert info["status"] == "FINISHED" and info["run_name"] == "ranker-lopo" and info["tracking_uri"] == uri
    cal.log_to_run(
        run_id,
        artifacts={"model/calibrator.json": source, "top.json": source},
        metrics={"calibration_ece": 0.02, "calibration_bad": float("nan")},
        tags={"calibration_bootstrap": "repo"},
        uri=uri,
    )
    after = client.get_run(run_id)
    assert after.info.status == "FINISHED" and after.info.end_time == before.info.end_time
    assert after.data.params == {"max_depth": "3"}
    assert after.data.metrics == {"calibration_ece": 0.02}
    assert after.data.tags["calibration_bootstrap"] == "repo"
    assert {a.path for a in client.list_artifacts(run_id, "model")} == {"model/calibrator.json"}
    assert "top.json" in {a.path for a in client.list_artifacts(run_id)}
    cal.log_to_run(
        run_id,
        artifacts={"model/calibrator.json": source},
        metrics={"calibration_ece": 0.03},
        tags={"calibration_bootstrap": "row"},
        uri=uri,
    )
    again = client.get_run(run_id)
    assert again.data.metrics["calibration_ece"] == 0.03 and again.data.tags["calibration_bootstrap"] == "row"
    assert again.info.status == "FINISHED"
    with pytest.raises(ValueError, match="not found"):
        cal.find_run("0" * 32, uri=uri)
    for bad in ("/abs/x.json", "../x.json", "model/", "a\\b.json", "model/../x.json"):
        with pytest.raises(ValueError, match="artifact key"):
            cal.log_to_run(run_id, artifacts={bad: source}, metrics={}, tags={}, uri=uri)
    with pytest.raises(FileNotFoundError):
        cal.log_to_run(run_id, artifacts={"x.json": tmp_path / "none"}, metrics={}, tags={}, uri=uri)


def test_default_run_id(tmp_path: Path) -> None:
    path = tmp_path / "ranker_lopo_run.json"
    with pytest.raises(ValueError, match="--run-id or --no-tracking"):
        cal.default_run_id(tmp_path, "ranker")
    path.write_text(json.dumps({"run_id": "abc", "protocol": "lopo", "model": "ranker"}), encoding="utf-8")
    assert cal.default_run_id(tmp_path, "ranker") == "abc"
    path.write_text(
        json.dumps({"run_id": "abc", "protocol": "temporal", "model": "ranker"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="--run-id or --no-tracking"):
        cal.default_run_id(tmp_path, "ranker")
    monotone = tmp_path / "ranker_monotone_lopo_run.json"
    monotone.write_text(
        json.dumps({"run_id": "abc", "protocol": "lopo", "model": "ranker"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected 'lopo' and 'ranker_monotone'"):
        cal.default_run_id(tmp_path, "ranker_monotone")
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        cal.default_run_id(tmp_path, "ranker")


# ---------------------------------------------------------------------------------------------------------
# command


def _cli() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        """Test app."""

    register(app)
    return app


@pytest.fixture
def cli_env(
    tmp_path: Path, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> Iterator[tuple[Callable[..., Result], Path, str]]:
    results = tmp_path / "results"
    results.mkdir()
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("TDP_RESULTS_DIR", str(results))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("TDP_CODE_SHA", "c0ffee")
    frame.drop(columns=["p_true"]).to_parquet(results / "ranker_oof.parquet", index=False)
    test = frame[frame["is_test"]].drop(columns=["p_true"]).assign(protocol="temporal", split="test")
    test.to_parquet(results / "ranker_scores_temporal.parquet", index=False)
    app = _cli()
    runner = CliRunner()

    def invoke(*args: str) -> Result:
        return runner.invoke(app, ["calibrate", *args, "--n-boot", "10", "--ci-boot", "20"])

    yield invoke, results, uri


def test_cli_without_tracking(cli_env: tuple[Callable[..., Result], Path, str]) -> None:
    invoke, results, _ = cli_env
    result = invoke("--no-tracking")
    assert result.exit_code == 0, result.output
    assert "ECE target 0.05" in result.output and "temporal_model" in result.output
    assert "MLflow tracking off" in result.output
    for path in cal.output_paths(results, "ranker").values():
        assert path.is_file(), path
    record = json.loads((results / "ranker_calibration.json").read_text(encoding="utf-8"))
    assert record["provenance"]["code_git_sha"] == "c0ffee"
    assert record["provenance"]["temporal_scores"]["path"].endswith("ranker_scores_temporal.parquet")
    assert record["provenance"]["run_id"] is None
    assert set(record["evaluation"]) == {"oof", "temporal_model"}


def test_cli_skips_missing_temporal_scores(cli_env: tuple[Callable[..., Result], Path, str]) -> None:
    invoke, results, _ = cli_env
    (results / "ranker_scores_temporal.parquet").unlink()
    result = invoke("--no-tracking")
    assert result.exit_code == 0, result.output
    assert "skipping that check" in result.output
    record = json.loads((results / "ranker_calibration.json").read_text(encoding="utf-8"))
    assert set(record["evaluation"]) == {"oof"} and record["provenance"]["temporal_scores"] is None


def test_cli_logs_to_the_lopo_run(cli_env: tuple[Callable[..., Result], Path, str], tmp_path: Path) -> None:
    from mlflow.tracking import MlflowClient

    invoke, results, uri = cli_env
    run_id = _finished_run(uri, tmp_path)
    record = {"run_id": run_id, "protocol": "lopo", "model": "ranker"}
    (results / "ranker_lopo_run.json").write_text(json.dumps(record), encoding="utf-8")
    result = invoke()
    assert result.exit_code == 0, result.output
    assert f"MLflow run {run_id} updated" in result.output
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert "calibration_ece" in run.data.metrics and run.data.tags["calibration_code_git_sha"] == "c0ffee"
    assert {a.path for a in client.list_artifacts(run_id, "model")} == {"model/calibrator.json"}
    assert "figures/reliability.png" in {a.path for a in client.list_artifacts(run_id, "figures")}
    names = {a.path for a in client.list_artifacts(run_id, "results")}
    assert {"results/ranker_calibration.json", "results/ranker_reliability.parquet"} <= names


def test_cli_errors(cli_env: tuple[Callable[..., Result], Path, str], tmp_path: Path) -> None:
    invoke, results, _ = cli_env
    missing = invoke("--no-tracking", "--scores", str(tmp_path / "none.parquet"))
    assert missing.exit_code == 2 and "not found" in missing.output
    no_record = invoke()
    assert no_record.exit_code == 2 and "--run-id or --no-tracking" in no_record.output
    unknown = invoke("--run-id", "0" * 32)
    assert unknown.exit_code == 2 and "not found" in unknown.output
    assert invoke("--no-tracking", "--model", "Bad-Name").exit_code == 2
    temporal = invoke("--no-tracking", "--temporal-scores", str(tmp_path / "none.parquet"))
    assert temporal.exit_code == 2 and "temporal scores not found" in temporal.output
    assert invoke("--no-tracking", "--bootstrap", "foo").exit_code == 2
    assert not (results / "ranker_calibrator.json").exists()


def test_cli_help_lists_calibrate() -> None:
    from tdp_ml.cli import app, load_commands

    load_commands()
    top = CliRunner().invoke(app, ["--help"])
    assert top.exit_code == 0 and "calibrate" in top.output
    assert CliRunner().invoke(app, ["calibrate", "--help"]).exit_code == 0


# ---------------------------------------------------------------------------------------------------------
# integration with the trainer's score layout


def test_calibrates_lopo_scores_from_the_trainer() -> None:
    from tdp_ml.eval import make_synthetic_dataset
    from tdp_ml.train import SCORE_COLUMNS, run_lopo

    data = make_synthetic_dataset(n_repos=3, n_snapshots=4, n_files=30)
    params = {
        "objective": "rank:ndcg",
        "tree_method": "hist",
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    result = run_lopo(data, params, nthread=2)
    assert tuple(result.scores.columns) == SCORE_COLUMNS
    out = cal.calibrate(result.scores, n_boot=5, ci_boot=10)
    assert out.final_rows.rows == len(data)
    assert out.fit_rows.rows == int((data["snapshot_idx"] == 0).sum())

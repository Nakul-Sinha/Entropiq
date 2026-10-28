"""Tests of the ranker training library and the ``tdp-ml train`` command (synthetic data only)."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import typer
import yaml
from typer.testing import CliRunner, Result

import tdp_ml.train as tr
from tdp_core.contracts import FEATURE_INPUT_COLUMNS
from tdp_core.model.ranker import Ranker
from tdp_ml.commands.train import register
from tdp_ml.eval import (
    QUERY_COLUMNS,
    RANKING_METRICS,
    RUN_COLUMNS,
    TEST_COLUMNS,
    evaluate,
    make_synthetic_dataset,
    per_query_path,
    run_protocol,
    save_per_query,
)
from tdp_ml.pipeline.config import ConfigError
from tdp_ml.pipeline.layout import PipelineError

PARAMS: dict[str, Any] = {
    "objective": "rank:ndcg",
    "lambdarank_pair_method": "topk",
    "lambdarank_num_pair_per_sample": 10,
    "tree_method": "hist",
    "n_estimators": 60,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_synthetic_dataset(n_repos=4, n_snapshots=8, n_files=40, seed=42)


@pytest.fixture(scope="module")
def small_config() -> tr.RankerConfig:
    cfg = tr.load_ranker_config()
    fixed = {**cfg.fixed, "n_estimators": 200, "early_stopping_rounds": 20}
    return dataclasses.replace(cfg, fixed=fixed, device="cpu", nthread=2)


@pytest.fixture(scope="module")
def temporal(frame: pd.DataFrame) -> tr.ProtocolResult:
    return tr.run_temporal(frame, PARAMS, nthread=2)


@pytest.fixture(scope="module")
def lopo(frame: pd.DataFrame) -> tr.ProtocolResult:
    return tr.run_lopo(frame, PARAMS, nthread=2)


@pytest.fixture(scope="module")
def tuned(frame: pd.DataFrame) -> tuple[tr.TuningResult, float]:
    started = time.perf_counter()
    result = tr.tune(frame, tr.load_ranker_config(), n_trials=3, device="cpu", nthread=2)
    return result, time.perf_counter() - started


def _write_config(path: Path, cfg: tr.RankerConfig, **changes: Any) -> Path:
    data = cfg.to_json() | changes
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# ---------- configuration ----------


def test_default_config_matches_the_plan() -> None:
    cfg = tr.load_ranker_config()
    assert dict(cfg.fixed) == {
        "objective": "rank:ndcg",
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 10,
        "tree_method": "hist",
        "eval_metric": ["ndcg@10"],
        "early_stopping_rounds": 100,
        "n_estimators": 2000,
    }
    assert (cfg.seed, cfg.n_trials, cfg.n_folds, cfg.rounds_factor) == (42, 50, 2, 1.1)
    assert (cfg.experiment, cfg.model_name, cfg.device, cfg.nthread) == (
        "entropiq-ranker",
        "ranker",
        "auto",
        4,
    )
    assert [(p.name, p.kind, p.low, p.high, p.log) for p in cfg.search] == [
        ("max_depth", "int", 3, 10, False),
        ("learning_rate", "float", 0.01, 0.2, True),
        ("min_child_weight", "float", 1.0, 20.0, False),
        ("subsample", "float", 0.5, 1.0, False),
        ("colsample_bytree", "float", 0.5, 1.0, False),
        ("reg_lambda", "float", 0.001, 10.0, True),
        ("reg_alpha", "float", 0.001, 10.0, True),
    ]
    columns = tr.monotone_columns(cfg.monotone_features)
    assert len(columns) == 8
    assert columns[:4] == ("commits_decayed", "churn_decayed", "prior_latent_bugs", "transitive_dependents")
    assert tr.monotone_constraints(columns) == dict.fromkeys(columns, 1)


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    cfg = tr.load_ranker_config()
    assert tr.load_ranker_config(_write_config(tmp_path / "c.yaml", cfg)) == cfg


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"unknown": 1}, "unknown"),
        (
            {
                "fixed": {
                    "objective": "rank:ndcg",
                    "n_estimators": 5,
                    "early_stopping_rounds": 2,
                    "device": "cuda",
                }
            },
            "device",
        ),
        ({"fixed": {"objective": "rank:ndcg", "n_estimators": 5}}, "early_stopping_rounds"),
        ({"search": {"max_depth": {"type": "int", "low": 5, "high": 5}}}, "low < high"),
        ({"search": {"reg_alpha": {"type": "float", "low": 0, "high": 1, "log": True}}}, "log"),
        ({"search": {"max_depth": {"type": "int", "low": 2.5, "high": 5}}}, "integral"),
        ({"search": {"n_estimators": {"type": "int", "low": 2, "high": 5}}}, "n_estimators"),
        ({"search": {"max_depth": {"type": "cat", "low": 2, "high": 5}}}, "type"),
        ({"search": {}}, "search"),
        ({"monotone_features": ["no_such_feature"]}, "no_such_feature"),
        ({"monotone_features": ["sloc", "sloc"]}, "twice"),
        ({"device": "gpu"}, "device"),
        ({"seed": True}, "seed"),
        ({"n_trials": 0}, "n_trials"),
        ({"rounds_factor": 0}, "rounds_factor"),
        ({"model_name": "Ranker"}, "model_name"),
        ({"experiment": ""}, "experiment"),
    ],
)
def test_bad_config_raises(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    path = _write_config(tmp_path / "bad.yaml", tr.load_ranker_config(), **change)
    with pytest.raises(ConfigError, match=message) as info:
        tr.load_ranker_config(path)
    assert "bad.yaml" in str(info.value)


def test_unreadable_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        tr.load_ranker_config(tmp_path / "missing.yaml")
    (tmp_path / "list.yaml").write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        tr.load_ranker_config(tmp_path / "list.yaml")


# ---------- data ----------


def test_training_rows_drop_all_zero_queries(frame: pd.DataFrame) -> None:
    qid = int(frame["qid"].iloc[0])
    data = frame.copy()
    data.loc[data["qid"] == qid, "grade"] = 0
    data = data.sample(frac=1.0, random_state=1)
    rows = tr.training_rows(data)
    assert qid not in set(rows["qid"])
    assert set(rows["qid"]) == set(data["qid"]) - {qid}
    expected = data.loc[data["qid"] != qid].sort_values(["qid", "path"], kind="stable")
    assert rows.index.equals(expected.index)
    assert tr.training_rows(data.iloc[:0]).empty


def test_load_dataset(tmp_path: Path, frame: pd.DataFrame) -> None:
    with pytest.raises(PipelineError, match="tdp-ml dataset"):
        tr.load_dataset(tmp_path / "none.parquet")
    bad = frame.copy()
    bad["sloc"] = bad["sloc"].astype("float64")
    bad.to_parquet(tmp_path / "bad.parquet", index=False)
    with pytest.raises(PipelineError, match="float32"):
        tr.load_dataset(tmp_path / "bad.parquet")
    frame.sample(frac=1.0, random_state=3).to_parquet(tmp_path / "ok.parquet", index=False)
    loaded = tr.load_dataset(tmp_path / "ok.parquet")
    assert isinstance(loaded.index, pd.RangeIndex)
    expected = pd.read_parquet(tmp_path / "ok.parquet").sort_values(["qid", "path"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(loaded, expected)
    assert loaded[["qid", "path"]].equals(frame[["qid", "path"]])
    assert len(tr.file_sha256(tmp_path / "ok.parquet")) == 64


def test_test_mask_marks_last_snapshots(frame: pd.DataFrame) -> None:
    mask = tr.test_mask(frame)
    last = frame.groupby("repo")["snapshot_idx"].transform("max")
    assert mask.equals((frame["snapshot_idx"] == last).rename("is_test"))


def test_code_git_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TDP_CODE_SHA", "abc123")
    assert tr.code_git_sha() == "abc123"
    monkeypatch.delenv("TDP_CODE_SHA")
    sha = tr.code_git_sha()
    assert sha == "unknown" or len(sha) == 40

    def boom(*args: Any, **kwargs: Any) -> bytes:
        raise OSError("no git")

    monkeypatch.setattr(tr, "run_git", boom)
    assert tr.code_git_sha() == "unknown"


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tr.resolve_device("cpu") == "cpu"
    monkeypatch.setattr(tr, "_cuda_probe", lambda: False)
    assert tr.resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        tr.resolve_device("cuda")
    with pytest.raises(ValueError, match="device"):
        tr.resolve_device("gpu")
    monkeypatch.setattr(tr, "_cuda_probe", lambda: True)
    assert tr.resolve_device("auto") == "cuda"
    assert tr.resolve_device("cuda") == "cuda"


def test_cuda_probe_returns_a_bool() -> None:
    assert isinstance(tr._cuda_probe(), bool)


# ---------- fitting ----------


def _folds(frame: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    return tr._tuning_folds(frame, 365, 2)


def test_fit_ranker_with_early_stopping(frame: pd.DataFrame, small_config: tr.RankerConfig) -> None:
    train, valid = _folds(frame)[1]
    params = tr.tuning_params(small_config, {"max_depth": 4, "learning_rate": 0.1})
    fit = tr.fit_ranker(train, params, valid=valid, nthread=2)
    assert fit.best_rounds is not None and 1 <= fit.best_rounds <= 200
    assert fit.booster.num_features() == 89
    assert fit.booster.feature_names == list(FEATURE_INPUT_COLUMNS)
    config = json.loads(fit.booster.save_config())
    assert config["learner"]["generic_param"]["device"] == "cpu"
    assert (fit.rows, fit.queries) == (len(train), train["qid"].nunique())
    again = tr.fit_ranker(train, params, valid=valid, nthread=2)
    first = tr.score_rows(fit.booster, valid, rounds=fit.best_rounds)
    second = tr.score_rows(again.booster, valid, rounds=again.best_rounds)
    assert first.equals(second)
    assert first.index.equals(valid.index) and first.dtype == np.float64 and first.name == "score"


def test_fit_ranker_without_validation_ignores_early_stopping(
    frame: pd.DataFrame, small_config: tr.RankerConfig
) -> None:
    params = tr.tuning_params(small_config, {"max_depth": 3, "learning_rate": 0.1}) | {"n_estimators": 25}
    fit = tr.fit_ranker(frame, params, nthread=2)
    assert fit.best_rounds is None
    assert fit.booster.num_boosted_rounds() == 25
    scores = tr.score_rows(fit.booster, frame)
    assert np.array_equal(scores.to_numpy(), Ranker(fit.booster).score(frame[list(FEATURE_INPUT_COLUMNS)]))
    assert tr.score_rows(fit.booster, frame.iloc[:0]).empty


def test_fit_ranker_errors(frame: pd.DataFrame) -> None:
    zero = frame.assign(grade=np.int8(0))
    with pytest.raises(ValueError, match="no training query"):
        tr.fit_ranker(zero, PARAMS)
    with pytest.raises(ValueError, match="no validation query"):
        tr.fit_ranker(frame, PARAMS | {"early_stopping_rounds": 5}, valid=zero)
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        tr.fit_ranker(frame, PARAMS, valid=frame)


def test_rounds_to_estimators() -> None:
    assert tr.rounds_to_estimators((89, 71), 1.1) == 88
    assert tr.rounds_to_estimators((3,), 1.1) == 3
    assert tr.rounds_to_estimators((1,), 0.4) == 1
    assert tr.rounds_to_estimators((5, 6), 1.0) == 6  # 5.5 rounds half up
    with pytest.raises(ValueError):
        tr.rounds_to_estimators((), 1.1)
    with pytest.raises(ValueError):
        tr.rounds_to_estimators((0, 3), 1.1)
    with pytest.raises(ValueError):
        tr.rounds_to_estimators((3,), 0.0)


def test_final_params() -> None:
    cfg = tr.load_ranker_config()
    params = tr.final_params(cfg, {"max_depth": 5}, 77)
    assert not set(tr.TUNING_ONLY_PARAMS) & set(params)
    assert (params["random_state"], params["n_estimators"], params["max_depth"]) == (42, 77, 5)
    assert params["objective"] == "rank:ndcg"
    assert tr.tuning_params(cfg, {"max_depth": 5})["early_stopping_rounds"] == 100
    assert tr.with_monotone(params, ["sloc"])["monotone_constraints"] == {"sloc": 1}


# ---------- tuning ----------


def test_tune_default_config_three_trials(tuned: tuple[tr.TuningResult, float]) -> None:
    result, seconds = tuned
    assert seconds < 60
    assert len(result.trials) == 3 and len(result.folds) == 2 and len(result.fold_rounds) == 2
    cfg = tr.load_ranker_config()
    for p in cfg.search:
        assert p.low <= result.best_params[p.name] <= p.high
    assert isinstance(result.best_params["max_depth"], int)
    assert result.n_estimators == tr.rounds_to_estimators(result.fold_rounds, 1.1)
    expected = {"number", "state", "value", "seconds", *(p.name for p in cfg.search)}
    expected |= {"fold0_ndcg_10", "fold1_ndcg_10", "fold0_rounds", "fold1_rounds"}
    assert expected <= set(result.trials.columns)
    assert (result.trials["state"] == "COMPLETE").all()
    assert math.isclose(result.best_value, float(result.trials["value"].max()))
    assert result.folds[0]["train_queries"] == 4 and result.folds[1]["train_queries"] == 8


def test_tune_is_deterministic(frame: pd.DataFrame, small_config: tr.RankerConfig) -> None:
    first = tr.tune(frame, small_config, n_trials=2)
    second = tr.tune(frame, small_config, n_trials=2)
    assert first.best_params == second.best_params
    assert first.best_value == second.best_value
    assert first.fold_rounds == second.fold_rounds


def test_tune_errors(frame: pd.DataFrame, small_config: tr.RankerConfig) -> None:
    with pytest.raises(ValueError, match="n_trials"):
        tr.tune(frame, small_config, n_trials=0)
    one_snapshot = frame.loc[frame["snapshot_idx"] == frame["snapshot_idx"].max()]
    with pytest.raises(ValueError, match="forward fold"):
        tr.tune(one_snapshot, small_config, n_trials=1)


def test_monotone_tuning_runs(frame: pd.DataFrame, small_config: tr.RankerConfig) -> None:
    result = tr.tune(frame, small_config, n_trials=1, monotone=True)
    assert len(result.trials) == 1


# ---------- params file ----------


def test_tuned_params_round_trip(tmp_path: Path, tuned: tuple[tr.TuningResult, float]) -> None:
    cfg = tr.load_ranker_config()
    params = tr.tuned_params(cfg, tuned[0], info={"run_id": "r1", "monotone": False})
    assert params.n_estimators == tuned[0].n_estimators
    assert params.info["n_trials"] == 3 and params.info["run_id"] == "r1"
    assert not set(tr.TUNING_ONLY_PARAMS) & set(params.params)
    path = tr.save_params(tmp_path / "p.json", params)
    assert tr.load_params(path) == params

    data = json.loads(path.read_text(encoding="utf-8"))
    for change, message in (
        ({"format": 2}, "format"),
        ({"params": {**data["params"], "device": "cuda"}}, "device"),
        ({"params": {**data["params"], "n_estimators": 0}}, "n_estimators"),
        ({"best_value": "x"}, "best_value"),
        ({"fold_rounds": ["a"]}, "fold_rounds"),
    ):
        (tmp_path / "bad.json").write_text(json.dumps(data | change), encoding="utf-8")
        with pytest.raises(ConfigError, match=message):
            tr.load_params(tmp_path / "bad.json")
    with pytest.raises(ConfigError, match="does not exist"):
        tr.load_params(tmp_path / "missing.json")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError):
        tr.load_params(tmp_path / "broken.json")
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(params, best_value=math.nan).to_json()


# ---------- protocols ----------


def test_run_temporal(
    frame: pd.DataFrame, temporal: tr.ProtocolResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[pd.DataFrame] = []
    original = tr.fit_ranker

    def spy(train: pd.DataFrame, *args: Any, **kwargs: Any) -> tr.FitResult:
        seen.append(train)
        return original(train, *args, **kwargs)

    monkeypatch.setattr(tr, "fit_ranker", spy)
    tr.run_temporal(frame, PARAMS | {"n_estimators": 5}, nthread=2)
    assert len(seen) == 1
    last = frame.groupby("repo")["snapshot_time"].max()
    train = seen[0]
    assert (train["snapshot_time"] + pd.Timedelta(days=365) <= train["repo"].map(last)).all()

    pq = temporal.per_query
    assert list(pq.columns) == [*RUN_COLUMNS, *QUERY_COLUMNS, *RANKING_METRICS]
    assert (pq["model"] == "ranker").all() and (pq["protocol"] == "temporal").all()
    assert set(pq["qid"]) == set(frame.loc[tr.test_mask(frame), "qid"])
    scores = temporal.scores
    assert list(scores.columns) == list(tr.SCORE_COLUMNS)
    assert scores["is_test"].all() and len(scores) == int(tr.test_mask(frame).sum())
    rows = frame.set_index(["qid", "path"]).loc[list(zip(scores["qid"], scores["path"], strict=True))]
    expected = Ranker(temporal.booster).score(rows[list(FEATURE_INPUT_COLUMNS)])
    assert np.array_equal(scores["score"].to_numpy(), expected)
    assert pq["ndcg_10"].mean() > 0.4
    assert temporal.training_rows == len(tr.training_rows(train))


def test_run_lopo(frame: pd.DataFrame, lopo: tr.ProtocolResult, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[set[str]] = []
    original = tr.fit_ranker

    def spy(train: pd.DataFrame, *args: Any, **kwargs: Any) -> tr.FitResult:
        seen.append(set(train["repo"]))
        return original(train, *args, **kwargs)

    monkeypatch.setattr(tr, "fit_ranker", spy)
    tr.run_lopo(frame, PARAMS | {"n_estimators": 5}, nthread=2)
    repos = sorted(set(frame["repo"]))
    assert len(seen) == 5
    for held, used in zip(repos, seen[:4], strict=True):
        assert used == set(repos) - {held}
    assert seen[4] == set(repos)

    assert set(lopo.per_query["split"]) == set(repos)
    oof = lopo.scores
    assert list(oof.columns) == list(tr.SCORE_COLUMNS)
    assert len(oof) == len(frame) and not oof.duplicated(["qid", "path"]).any()
    assert (oof["split"] == oof["repo"]).all()
    merged = frame.merge(oof[["qid", "path", "is_test"]], on=["qid", "path"])
    assert merged["is_test"].equals(tr.test_mask(merged).rename("is_test"))
    test_rows = oof.loc[oof["is_test"]]
    again = evaluate(test_rows, "score")
    assert np.allclose(again["ndcg_10"].to_numpy(), lopo.per_query["ndcg_10"].to_numpy(), equal_nan=True)
    assert lopo.booster.num_boosted_rounds() == PARAMS["n_estimators"]
    assert lopo.training_rows == len(tr.training_rows(frame))


def test_run_lopo_leak_guard(frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    def leaky(df: pd.DataFrame, protocol: str, fit_predict: Callable[..., Any], **kwargs: Any) -> Any:
        held = df.loc[df["repo"] == df["repo"].iloc[0]]
        return fit_predict(df, held)

    monkeypatch.setattr(tr, "run_protocol", leaky)
    with pytest.raises(RuntimeError, match="LOPO leak"):
        tr.run_lopo(frame, PARAMS, nthread=2)


def test_monotone_fit_is_monotone(frame: pd.DataFrame) -> None:
    columns = tr.monotone_columns(tr.load_ranker_config().monotone_features)
    fit = tr.fit_ranker(frame, tr.with_monotone(PARAMS, columns), nthread=2)
    rows = frame.sample(50, random_state=7)
    raised = rows.copy()
    raised["commits_decayed"] = raised["commits_decayed"] + np.float32(5.0)
    raised["pct_commits_decayed"] = np.minimum(
        raised["pct_commits_decayed"] + np.float32(0.3), np.float32(1.0)
    )
    before = tr.score_rows(fit.booster, rows).to_numpy()
    after = tr.score_rows(fit.booster, raised).to_numpy()
    assert (after >= before).all()


# ---------- comparison ----------


def _hotspot_scores(train: pd.DataFrame, test: pd.DataFrame) -> Any:
    return test["pct_churn_decayed"].to_numpy(np.float64) * test["pct_cc_sum"].to_numpy(np.float64)


def _hotspot_run(frame: pd.DataFrame) -> pd.DataFrame:
    return run_protocol(frame, "temporal", _hotspot_scores, model="hotspot").per_query


def test_comparison_runs(
    tmp_path: Path, frame: pd.DataFrame, temporal: tr.ProtocolResult, caplog: pytest.LogCaptureFixture
) -> None:
    pq = temporal.per_query
    alone = tr.comparison_runs(frame, "temporal", pq, results_dir=tmp_path, horizon_days=365)
    assert list(alone) == ["hotspot"]
    pd.testing.assert_frame_equal(alone["hotspot"], _hotspot_run(frame))

    loc = run_protocol(frame, "temporal", lambda tr_, te: te["sloc"], model="loc").per_query
    baselines = pd.concat([loc, _hotspot_run(frame)], ignore_index=True)
    path = save_per_query(baselines, per_query_path("baselines", "temporal", root=tmp_path))
    found = tr.comparison_runs(frame, "temporal", pq, results_dir=tmp_path, horizon_days=365)
    assert list(found) == ["loc", "hotspot"]
    pd.testing.assert_frame_equal(found["loc"], loc, check_dtype=False)
    path.with_suffix(".json").write_text(json.dumps({"dataset_sha256": "a" * 64}), encoding="utf-8")
    same = tr.comparison_runs(
        frame, "temporal", pq, results_dir=tmp_path, horizon_days=365, dataset_sha256="a" * 64
    )
    assert list(same) == ["loc", "hotspot"]
    with caplog.at_level(logging.WARNING, logger="tdp_ml.train"):
        other = tr.comparison_runs(
            frame, "temporal", pq, results_dir=tmp_path, horizon_days=365, dataset_sha256="b" * 64
        )
    assert list(other) == ["hotspot"] and "different dataset" in caplog.text
    path.with_suffix(".json").unlink()

    save_per_query(baselines.iloc[1:], path)
    with caplog.at_level(logging.WARNING, logger="tdp_ml.train"):
        ignored = tr.comparison_runs(frame, "temporal", pq, results_dir=tmp_path, horizon_days=365)
    assert list(ignored) == ["hotspot"]
    assert "differ" in caplog.text

    extra = save_per_query(pq.assign(model="other"), tmp_path / "other_temporal.parquet")
    with_extra = tr.comparison_runs(
        frame, "temporal", pq, results_dir=tmp_path, horizon_days=365, extra={"other": extra}
    )
    assert list(with_extra) == ["hotspot", "other"]


def test_compare_and_report(temporal: tr.ProtocolResult, frame: pd.DataFrame) -> None:
    loc = run_protocol(frame, "temporal", lambda tr_, te: te["sloc"], model="loc").per_query
    others = {"hotspot": _hotspot_run(frame), "loc": loc}
    summary, tests = tr.compare(temporal.per_query, others, n_boot=50)
    ndcg = summary.loc[summary["metric"] == "ndcg_10"]
    assert set(ndcg["model"]) == {"ranker", "hotspot", "loc"}
    assert len(tests) == len(others) * len(tr.COMPARE_METRICS)
    assert list(tests.columns) == list(TEST_COLUMNS)
    text = tr.summary_report("temporal", "ranker", summary, tests, notes=["a note"])
    assert text.startswith("# Ranker results (temporal)")
    assert "NDCG@10" in text and "hotspot" in text and "- a note" in text
    empty_summary, empty_tests = tr.compare(temporal.per_query, {}, n_boot=10)
    assert empty_tests.empty and list(empty_tests.columns) == list(TEST_COLUMNS)
    assert set(empty_summary["model"]) == {"ranker"}
    assert 0.0 <= tr.p_greater(temporal.per_query, others["hotspot"]) <= 1.0


# ---------- orchestration ----------


def test_train_writes_every_output(
    tmp_path: Path, frame: pd.DataFrame, small_config: tr.RankerConfig
) -> None:
    with pytest.raises(ValueError, match="protocol"):
        tr.train(frame, small_config, protocol="nope", results_dir=tmp_path)
    first = tr.train(frame, small_config, protocol="temporal", results_dir=tmp_path, n_trials=2, run_id="r1")
    names = {p.name for p in first.files.values()}
    assert names == {
        "ranker_params.json",
        "ranker_trials.parquet",
        "ranker_temporal.parquet",
        "ranker_scores_temporal.parquet",
        "ranker_temporal_summary.md",
    }
    assert all(p.is_file() for p in first.files.values())
    assert first.tuning is not None and first.params.info["run_id"] == "r1"
    assert set(first.seconds) == {"tuning", "fit", "total"}

    params_path = tmp_path / "ranker_params.json"
    mtime = params_path.stat().st_mtime_ns
    loaded = tr.load_params(params_path)
    second = tr.train(
        frame, small_config, protocol="lopo", results_dir=tmp_path, params=loaded, params_path=params_path
    )
    assert params_path.stat().st_mtime_ns == mtime
    assert second.tuning is None and second.files["params"] == params_path
    assert {p.name for p in second.files.values()} == {
        "ranker_params.json",
        "ranker_lopo.parquet",
        "ranker_oof.parquet",
        "ranker_lopo_summary.md",
    }
    assert len(pd.read_parquet(second.files["scores"])) == len(frame)

    third = tr.train(frame, small_config, protocol="lopo", results_dir=tmp_path, params=loaded, monotone=True)
    assert third.model == "ranker_monotone"
    assert (third.result.per_query["model"] == "ranker_monotone").all()
    assert {"hotspot", "ranker"} <= set(third.others)
    assert (tmp_path / "ranker_monotone_lopo.parquet").is_file()
    assert (tmp_path / "ranker_monotone_oof.parquet").is_file()
    assert "ranker_monotone" in (tmp_path / "ranker_monotone_lopo_summary.md").read_text(encoding="utf-8")


# ---------- command ----------


def _cli() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        """Test app."""

    register(app)
    return app


@pytest.fixture
def cli_env(
    tmp_path: Path, frame: pd.DataFrame, small_config: tr.RankerConfig, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Callable[..., Result], Path, Path, Path]]:
    results = tmp_path / "results"
    monkeypatch.setenv("TDP_RESULTS_DIR", str(results))
    dataset = tmp_path / "dataset.parquet"
    frame.to_parquet(dataset, index=False)
    (tmp_path / "dataset.json").write_text(json.dumps({"manifest_sha256": "m" * 64}), encoding="utf-8")
    config = _write_config(tmp_path / "ranker.yaml", small_config)
    app = _cli()
    runner = CliRunner()

    def invoke(*args: str) -> Result:
        return runner.invoke(app, list(args))

    yield invoke, results, dataset, config


def test_cli_trains_both_protocols(cli_env: tuple[Callable[..., Result], Path, Path, Path]) -> None:
    invoke, results, dataset, config = cli_env
    common = ["--dataset", str(dataset), "--config", str(config), "--device", "cpu", "--no-tracking"]
    first = invoke("train", "--protocol", "temporal", "--trials", "3", *common)
    assert first.exit_code == 0, first.output
    assert "NDCG@10" in first.output and "hotspot" in first.output
    params = results / "ranker_params.json"
    info = json.loads(params.read_text(encoding="utf-8"))["info"]
    assert info["n_trials"] == 3 and info["manifest_sha256"] == "m" * 64 and info["device"] == "cpu"
    second = invoke("train", "--protocol", "lopo", "--params", str(params), "--jobs", "2", *common)
    assert second.exit_code == 0, second.output
    oof = pd.read_parquet(results / "ranker_oof.parquet")
    assert len(oof) == len(pd.read_parquet(dataset))


def test_cli_errors(cli_env: tuple[Callable[..., Result], Path, Path, Path], tmp_path: Path) -> None:
    invoke, _, dataset, config = cli_env
    common = ["--config", str(config), "--device", "cpu", "--no-tracking"]
    missing = invoke("train", "--protocol", "temporal", "--dataset", str(tmp_path / "no.parquet"), *common)
    assert missing.exit_code == 2 and "tdp-ml dataset" in missing.output
    assert invoke("train", "--protocol", "foo", "--dataset", str(dataset), *common).exit_code == 2
    no_params = invoke(
        "train",
        "--protocol",
        "lopo",
        "--dataset",
        str(dataset),
        "--params",
        str(tmp_path / "p.json"),
        *common,
    )
    assert no_params.exit_code == 2 and "does not exist" in no_params.output


def test_cli_logs_to_mlflow(
    cli_env: tuple[Callable[..., Result], Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoke, results, dataset, config = cli_env
    for name in ("MLFLOW_ALLOW_FILE_STORE", "MLFLOW_DISABLE_TELEMETRY", "MLFLOW_DISABLE_AGENT_HINT"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("TDP_CODE_SHA", "c0ffee")
    args = ["--dataset", str(dataset), "--config", str(config), "--device", "cpu", "--trials", "2"]
    result = invoke("train", "--protocol", "temporal", *args)
    assert result.exit_code == 0, result.output
    record = json.loads((results / "ranker_temporal_run.json").read_text(encoding="utf-8"))
    assert "model/ranker_temporal.json" in record["artifacts"]

    import mlflow
    from mlflow.tracking import MlflowClient

    run = MlflowClient().get_run(record["run_id"])
    assert run.info.status == "FINISHED"
    assert "max_depth" in run.data.params and run.data.params["device"] == "cpu"
    assert "temporal_ndcg_10" in run.data.metrics
    assert "temporal_ndcg_10_p_greater_hotspot" in run.data.metrics
    assert run.data.tags["dataset_sha256"] == tr.file_sha256(dataset)
    assert run.data.tags["code_git_sha"] == "c0ffee"
    assert run.data.tags["tuned_in_run"] == record["run_id"]
    local = mlflow.artifacts.download_artifacts(
        run_id=record["run_id"], artifact_path="model/ranker_temporal.json", dst_path=str(tmp_path / "dl")
    )
    assert Ranker.from_json(Path(local)).booster.num_features() == 89
    names = {a.path for a in MlflowClient().list_artifacts(record["run_id"], "results")}
    assert "results/ranker_temporal.parquet" in names
    assert "results/ranker_params.json" in names


def test_cli_help_lists_commands() -> None:
    from tdp_ml.cli import app, load_commands

    load_commands()
    top = CliRunner().invoke(app, ["--help"])
    assert top.exit_code == 0 and "train" in top.output and "mlflow-relocate" in top.output
    assert CliRunner().invoke(app, ["train", "--help"]).exit_code == 0

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tdp_ml.baselines import (
    BASELINES,
    baseline_fit_predict,
    check_baselines,
    complexity_scores,
    hotspot_scores,
    loc_scores,
    order_per_query,
    prior_bugs_scores,
    rf_fit_predict,
    run_baseline,
    run_baselines,
    run_random,
    xgb_point_fit_predict,
)
from tdp_ml.cli import app, load_commands
from tdp_ml.commands.baselines import (
    PROVENANCE_KEYS,
    file_sha256,
    provenance_path,
    read_provenance,
    same_provenance,
    summary_paths,
    write_provenance,
    write_summary,
)
from tdp_ml.eval import (
    QUERY_COLUMNS,
    RANKING_METRICS,
    RUN_COLUMNS,
    load_per_query,
    make_synthetic_dataset,
    per_query_path,
    run_protocol,
    summarize,
)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_synthetic_dataset(n_repos=4, n_snapshots=5, n_files=50, seed=7)


@pytest.fixture(scope="module")
def lopo_all(data: pd.DataFrame) -> pd.DataFrame:
    return run_baselines(data, "lopo", n_jobs=2, n_shuffles=10)


def _mean(per_query: pd.DataFrame, model: str, metric: str) -> float:
    return float(per_query.loc[per_query["model"] == model, metric].mean())


def test_heuristic_scores_use_the_documented_columns() -> None:
    test = pd.DataFrame(
        {
            "sloc": [10, 200, 3],
            "cc_sum": [4, 1, 9],
            "pct_churn_decayed": [0.5, 1.0, 0.2],
            "pct_cc_sum": [0.4, 0.1, 1.0],
        }
    )
    np.testing.assert_array_equal(loc_scores(test), [10.0, 200.0, 3.0])
    np.testing.assert_array_equal(complexity_scores(test), [4.0, 1.0, 9.0])
    np.testing.assert_allclose(hotspot_scores(test), [0.2, 0.1, 0.2])
    assert loc_scores(test).dtype == np.float64


def test_prior_bugs_orders_by_latent_bugs_then_bugfix_commits() -> None:
    test = pd.DataFrame(
        {
            "prior_latent_bugs": [1, 0, 1, 2, 1, 0],
            "prior_bugfix_commits": [0, 5, 3, 0, 3, 0],
        }
    )
    scores = prior_bugs_scores(test)
    np.testing.assert_array_equal(scores, [2.0, 1.0, 3.0, 4.0, 3.0, 0.0])
    assert prior_bugs_scores(test.iloc[:0]).shape == (0,)


def test_heuristic_scores_reject_missing_columns_and_nan() -> None:
    with pytest.raises(ValueError, match="sloc"):
        loc_scores(pd.DataFrame({"cc_sum": [1]}))
    with pytest.raises(ValueError, match="NaN"):
        complexity_scores(pd.DataFrame({"cc_sum": [1.0, np.nan]}))
    with pytest.raises(ValueError, match="prior_bugfix_commits"):
        prior_bugs_scores(pd.DataFrame({"prior_latent_bugs": [1]}))


def test_baseline_fit_predict_rejects_random_and_unknown_names() -> None:
    with pytest.raises(ValueError, match="run_random"):
        baseline_fit_predict("random")
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_fit_predict("oracle")
    fit_predict = baseline_fit_predict("loc")
    test = pd.DataFrame({"sloc": [5, 1]})
    np.testing.assert_array_equal(np.asarray(fit_predict(test.iloc[:0], test)), [5.0, 1.0])


def test_check_baselines_orders_and_deduplicates() -> None:
    assert check_baselines(["rf", "loc", "rf", "random"]) == ["random", "loc", "rf"]
    assert check_baselines(BASELINES) == list(BASELINES)
    with pytest.raises(ValueError, match="unknown"):
        check_baselines(["loc", "magic"])
    with pytest.raises(ValueError, match="no baseline"):
        check_baselines([])


def test_run_random_is_the_mean_of_seeded_shuffles(data: pd.DataFrame) -> None:
    averaged = run_random(data, "temporal", n_shuffles=3, seed=5)
    singles = []
    for i in range(3):
        rng = np.random.default_rng([5, i])
        run = run_protocol(
            data,
            "temporal",
            lambda train, test, rng=rng: rng.permutation(len(test)).astype(np.float64),
            model="random",
        )
        singles.append(run.per_query)
    expected = pd.concat([s[list(RANKING_METRICS)] for s in singles]).groupby(level=0).mean()
    np.testing.assert_allclose(averaged[list(RANKING_METRICS)].to_numpy(), expected.to_numpy())
    pd.testing.assert_frame_equal(
        averaged[[*RUN_COLUMNS, *QUERY_COLUMNS]], singles[0][[*RUN_COLUMNS, *QUERY_COLUMNS]]
    )
    assert (averaged["model"] == "random").all()
    again = run_random(data, "temporal", n_shuffles=3, seed=5)
    pd.testing.assert_frame_equal(averaged, again)


def test_run_random_rejects_bad_shuffle_counts(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_shuffles"):
        run_random(data, "lopo", n_shuffles=0)


def test_rf_returns_probabilities_and_is_deterministic(data: pd.DataFrame) -> None:
    train = data[data["repo"] != data["repo"].iloc[0]]
    test = data[data["repo"] == data["repo"].iloc[0]]
    first = rf_fit_predict(train, test, n_jobs=2, seed=3)
    second = rf_fit_predict(train, test, n_jobs=1, seed=3)
    assert first.shape == (len(test),)
    assert ((first >= 0) & (first <= 1)).all()
    np.testing.assert_allclose(first, second)
    defective = test["has_latent_defect"].to_numpy(dtype=bool)
    assert first[defective].mean() > first[~defective].mean()


def test_rf_with_one_training_class_gives_constant_scores(data: pd.DataFrame) -> None:
    train = data.head(30).assign(has_latent_defect=False)
    scores = rf_fit_predict(train, data.tail(5))
    np.testing.assert_array_equal(scores, np.zeros(5))
    with pytest.raises(ValueError, match="training row"):
        rf_fit_predict(data.iloc[:0], data.tail(5))


def test_xgb_point_predicts_pain(data: pd.DataFrame) -> None:
    train = data[data["repo"] != data["repo"].iloc[0]]
    test = data[data["repo"] == data["repo"].iloc[0]]
    pred = xgb_point_fit_predict(train, test, n_jobs=2)
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()
    assert np.corrcoef(pred, test["pain"].to_numpy(dtype=np.float64))[0, 1] > 0.3
    with pytest.raises(ValueError, match="feature column"):
        xgb_point_fit_predict(train, test.drop(columns=["sloc"]))


def test_run_baselines_covers_every_baseline_and_query(data: pd.DataFrame, lopo_all: pd.DataFrame) -> None:
    assert list(dict.fromkeys(lopo_all["model"])) == list(BASELINES)
    test_qids = sorted(
        data.loc[
            data.groupby("repo")["snapshot_idx"].transform("max") == data["snapshot_idx"], "qid"
        ].unique()
    )
    for _, group in lopo_all.groupby("model"):
        assert group["qid"].tolist() == test_qids
        assert set(group["split"]) == set(data["repo"])
    assert (lopo_all["protocol"] == "lopo").all()
    summary = summarize(lopo_all, n_boot=50)
    assert set(summary["model"]) == set(BASELINES)


def test_learned_and_heuristic_baselines_beat_random(lopo_all: pd.DataFrame) -> None:
    random_ndcg = _mean(lopo_all, "random", "ndcg_10")
    for model in ("hotspot", "prior_bugs", "rf", "xgb_point"):
        assert _mean(lopo_all, model, "ndcg_10") > random_ndcg, model


def test_run_baseline_temporal_matches_the_harness(data: pd.DataFrame) -> None:
    per_query = run_baseline(data, "hotspot", "temporal")
    direct = run_protocol(
        data, "temporal", lambda train, test: hotspot_scores(test), model="hotspot"
    ).per_query
    pd.testing.assert_frame_equal(per_query, direct)


def test_order_per_query_puts_unknown_models_last() -> None:
    frame = pd.DataFrame(
        {
            "model": ["ranker", "rf", "loc", "rf"],
            "protocol": ["lopo"] * 4,
            "qid": [1, 2, 1, 1],
        }
    )
    ordered = order_per_query(frame)
    assert list(zip(ordered["model"], ordered["qid"], strict=True)) == [
        ("loc", 1),
        ("rf", 1),
        ("rf", 2),
        ("ranker", 1),
    ]


def _write_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.parquet"
    make_synthetic_dataset(n_repos=3, n_snapshots=4, n_files=30, seed=11).to_parquet(path, index=False)
    return path


def test_cli_writes_per_query_files_and_summary(tmp_path: Path) -> None:
    load_commands()
    runner = CliRunner()
    dataset = _write_dataset(tmp_path)
    out = tmp_path / "results"
    args = [
        "baselines",
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out),
        "--shuffles",
        "3",
        "--n-boot",
        "20",
    ]
    result = runner.invoke(app, [*args, "--model", "loc", "--model", "random"])
    assert result.exit_code == 0, result.output
    for protocol in ("temporal", "lopo"):
        stored = load_per_query(per_query_path("baselines", protocol, root=out))
        assert list(dict.fromkeys(stored["model"])) == ["random", "loc"]
    parquet_path, md_path = summary_paths(out)
    assert parquet_path.exists()
    assert "NDCG@10" in md_path.read_text(encoding="utf-8")
    assert "loc" in result.output

    again = runner.invoke(app, [*args, "--model", "loc", "--protocol", "lopo"])
    assert again.exit_code == 0, again.output
    assert "already in" in again.output

    merged = runner.invoke(app, [*args, "--model", "hotspot", "--model", "loc", "--protocol", "lopo"])
    assert merged.exit_code == 0, merged.output
    lopo = load_per_query(per_query_path("baselines", "lopo", root=out))
    assert list(dict.fromkeys(lopo["model"])) == ["random", "loc", "hotspot"]
    temporal = load_per_query(per_query_path("baselines", "temporal", root=out))
    assert set(temporal["model"]) == {"random", "loc"}

    forced = runner.invoke(app, [*args, "--model", "loc", "--protocol", "temporal", "--force"])
    assert forced.exit_code == 0, forced.output
    temporal = load_per_query(per_query_path("baselines", "temporal", root=out))
    assert list(dict.fromkeys(temporal["model"])) == ["random", "loc"]
    summary = pd.read_parquet(parquet_path)
    assert {("hotspot", "lopo"), ("loc", "temporal")} <= set(
        zip(summary["model"], summary["protocol"], strict=True)
    )


def test_cli_reports_bad_input(tmp_path: Path) -> None:
    load_commands()
    runner = CliRunner()
    missing = runner.invoke(app, ["baselines", "--dataset", str(tmp_path / "none.parquet")])
    assert missing.exit_code == 2
    assert "Dataset not found" in missing.output
    dataset = _write_dataset(tmp_path)
    bad_protocol = runner.invoke(app, ["baselines", "--dataset", str(dataset), "--protocol", "kfold"])
    assert bad_protocol.exit_code != 0
    bad_model = runner.invoke(app, ["baselines", "--dataset", str(dataset), "--model", "magic"])
    assert bad_model.exit_code != 0
    bad_jobs = runner.invoke(app, ["baselines", "--dataset", str(dataset), "--jobs", "0"])
    assert bad_jobs.exit_code != 0


def test_write_summary_without_results_returns_none(tmp_path: Path) -> None:
    assert write_summary(tmp_path) is None
    assert not summary_paths(tmp_path)[1].exists()


def _overwrite_dataset(path: Path) -> None:
    make_synthetic_dataset(n_repos=3, n_snapshots=4, n_files=30, seed=12).to_parquet(path, index=False)


def test_provenance_helpers(tmp_path: Path) -> None:
    note_path = provenance_path(tmp_path, "lopo")
    assert note_path == tmp_path / "baselines_lopo.json"
    assert read_provenance(note_path) is None

    note = {
        "dataset": "a/b.parquet",
        "dataset_sha256": "ab" * 32,
        "rows": 10,
        "horizon_days": 365,
        "seed": 42,
        "shuffles": 100,
    }
    assert write_provenance(note_path, note) == note_path
    assert read_provenance(note_path) == note
    assert not note_path.with_name(note_path.name + ".tmp").exists()

    assert set(PROVENANCE_KEYS) == {"dataset_sha256", "horizon_days", "seed", "shuffles"}
    assert same_provenance(note, {**note, "rows": 11, "dataset": "elsewhere.parquet"})
    for key, value in [("seed", 7), ("horizon_days", 180), ("shuffles", 3), ("dataset_sha256", "cd" * 32)]:
        assert not same_provenance(note, {**note, key: value})
    assert not same_provenance(None, note)
    assert not same_provenance({k: v for k, v in note.items() if k != "seed"}, note)

    note_path.write_text("{not json", encoding="utf-8")
    assert read_provenance(note_path) is None
    note_path.write_text("[1, 2]", encoding="utf-8")
    assert read_provenance(note_path) is None

    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"entropiq" * 200_000)
    assert file_sha256(blob) == hashlib.sha256(blob.read_bytes()).hexdigest()


def test_cli_reruns_when_the_dataset_or_options_change(tmp_path: Path) -> None:
    load_commands()
    runner = CliRunner()
    dataset = _write_dataset(tmp_path)
    out = tmp_path / "results"
    args = [
        "baselines",
        "--dataset",
        str(dataset),
        "--out-dir",
        str(out),
        "--n-boot",
        "20",
        "--protocol",
        "lopo",
    ]
    stored_path = per_query_path("baselines", "lopo", root=out)
    note_path = provenance_path(out, "lopo")

    first = runner.invoke(app, [*args, "--model", "loc", "--model", "hotspot"])
    assert first.exit_code == 0, first.output
    note = read_provenance(note_path)
    assert note is not None
    assert note["dataset_sha256"] == file_sha256(dataset)
    assert note["rows"] == len(pd.read_parquet(dataset))
    assert (note["horizon_days"], note["seed"], note["shuffles"]) == (365, 42, 100)

    again = runner.invoke(app, [*args, "--model", "loc"])
    assert again.exit_code == 0, again.output
    assert "already in" in again.output

    reseeded = runner.invoke(app, [*args, "--model", "loc", "--seed", "7"])
    assert reseeded.exit_code == 0, reseeded.output
    assert "another dataset or other options" in reseeded.output
    assert set(load_per_query(stored_path)["model"]) == {"loc"}

    _overwrite_dataset(dataset)
    rebuilt = runner.invoke(app, [*args, "--model", "loc", "--seed", "7"])
    assert rebuilt.exit_code == 0, rebuilt.output
    assert "another dataset or other options" in rebuilt.output
    stored = load_per_query(stored_path)
    expected = run_baseline(pd.read_parquet(dataset), "loc", "lopo", seed=7)
    assert list(stored["qid"]) == list(expected["qid"])
    for metric in RANKING_METRICS:
        np.testing.assert_allclose(
            stored[metric].to_numpy(dtype=float), expected[metric].to_numpy(dtype=float), equal_nan=True
        )
    note = read_provenance(note_path)
    assert note is not None
    assert note["dataset_sha256"] == file_sha256(dataset)
    assert note["seed"] == 7

    note_path.unlink()
    unknown = runner.invoke(app, [*args, "--model", "loc", "--seed", "7"])
    assert unknown.exit_code == 0, unknown.output
    assert "another dataset or other options" in unknown.output
    assert note_path.exists()


def test_summary_leaves_out_stale_protocol_files(tmp_path: Path) -> None:
    load_commands()
    runner = CliRunner()
    dataset = _write_dataset(tmp_path)
    out = tmp_path / "results"
    args = ["baselines", "--dataset", str(dataset), "--out-dir", str(out), "--n-boot", "20", "--model", "loc"]
    both = runner.invoke(app, args)
    assert both.exit_code == 0, both.output

    _overwrite_dataset(dataset)
    temporal = runner.invoke(app, [*args, "--protocol", "temporal"])
    assert temporal.exit_code == 0, temporal.output
    parquet_path, md_path = summary_paths(out)
    assert set(pd.read_parquet(parquet_path)["protocol"]) == {"temporal"}
    markdown = md_path.read_text(encoding="utf-8")
    assert file_sha256(dataset)[:16] in markdown
    assert "- temporal: dataset sha256" in markdown
    assert "- lopo:" not in markdown

    assert write_summary(out, n_boot=20) is not None
    assert set(pd.read_parquet(parquet_path)["protocol"]) == {"temporal", "lopo"}

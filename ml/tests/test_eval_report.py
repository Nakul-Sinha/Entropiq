import math
from pathlib import Path

import pandas as pd
import pytest

from tdp_ml.eval.harness import run_protocol
from tdp_ml.eval.report import (
    comparison_markdown,
    load_per_query,
    per_query_path,
    save_per_query,
    summary_markdown,
)
from tdp_ml.eval.stats import summarize, wilcoxon_holm
from tdp_ml.eval.synthetic import make_synthetic_dataset

DASHES = (chr(0x2013), chr(0x2014))  # en and em dash


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    df = make_synthetic_dataset(n_repos=3, n_snapshots=3, n_files=25, seed=6)
    hot = run_protocol(df, "lopo", lambda tr, te: te["pct_churn_decayed"] * te["pct_cc_sum"], model="hotspot")
    loc = run_protocol(df, "lopo", lambda tr, te: te["sloc"], model="loc")
    return pd.concat([hot.per_query, loc.per_query], ignore_index=True)


def test_per_query_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert per_query_path("baselines", "lopo", root=tmp_path) == tmp_path / "baselines_lopo.parquet"
    monkeypatch.setenv("TDP_RESULTS_DIR", str(tmp_path / "results"))
    assert per_query_path("ranker", "temporal") == tmp_path / "results" / "ranker_temporal.parquet"
    for bad in ("a/b", "", "x y", "..\\up"):
        with pytest.raises(ValueError, match="must match"):
            per_query_path(bad, "lopo")
    with pytest.raises(ValueError, match="protocol"):
        per_query_path("baselines", "lo/po")


def test_save_and_load_roundtrip(runs: pd.DataFrame, tmp_path: Path) -> None:
    path = per_query_path("baselines", "lopo", root=tmp_path / "nested" / "out")
    assert save_per_query(runs, path) == path
    assert sorted(p.name for p in path.parent.iterdir()) == ["baselines_lopo.parquet"]  # no .tmp left
    loaded = load_per_query(path)
    pd.testing.assert_frame_equal(loaded, runs, check_dtype=False)
    for col in ("qid", "n_files", "ndcg_10"):
        assert loaded[col].dtype == runs[col].dtype
    assert isinstance(loaded["snapshot_time"].dtype, pd.DatetimeTZDtype)


def test_save_rejects_bad_frames(runs: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    with pytest.raises(ValueError, match="qid"):
        save_per_query(runs.drop(columns=["qid"]), path)
    with pytest.raises(ValueError, match="metric"):
        save_per_query(runs[["model", "protocol", "qid", "repo"]], path)
    with pytest.raises(ValueError, match="duplicate qids"):
        save_per_query(pd.concat([runs, runs.head(1)]), path)
    # the same qid in two different runs is fine
    assert runs["qid"].duplicated().any()
    assert not list(tmp_path.iterdir())


def test_save_removes_tmp_file_on_failure(
    runs: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(src: object, dst: object) -> None:
        assert Path(str(src)).name == "x.parquet.tmp" and Path(str(src)).exists()
        raise OSError("disk full")

    monkeypatch.setattr("tdp_ml.eval.report.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        save_per_query(runs, tmp_path / "x.parquet")
    assert not list(tmp_path.iterdir())


def test_summary_markdown(runs: pd.DataFrame) -> None:
    summary = summarize(runs, n_boot=200)
    text = summary_markdown(summary, metrics=["ndcg_10", "popt", "recall_20loc"])
    lines = text.strip().splitlines()
    assert lines[0] == "| model | protocol | NDCG@10 | Popt | Recall@20%LOC |"
    assert lines[1] == "|---|---|---|---|---|"
    assert lines[2].startswith("| hotspot | lopo | ") and lines[3].startswith("| loc | lopo | ")
    row = summary[(summary["model"] == "hotspot") & (summary["metric"] == "ndcg_10")].iloc[0]
    assert f"{row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]" in lines[2]
    default = summary_markdown(summary)
    assert "NDCG@5" in default and "Spearman" in default
    assert not any(dash in text + default for dash in DASHES)


def test_summary_markdown_nan_and_escaping() -> None:
    summary = pd.DataFrame(
        {
            "model": ["a|b", "a|b", "c"],
            "metric": ["ndcg_10", "popt", "ndcg_10"],
            "mean": [0.5, math.nan, 0.25],
            "ci_low": [0.4, math.nan, math.nan],
            "ci_high": [0.6, math.nan, math.nan],
        }
    )
    lines = summary_markdown(summary, digits=2).strip().splitlines()
    assert lines[0] == "| model | NDCG@10 | Popt |"
    assert lines[2] == "| a\\|b | 0.50 [0.40, 0.60] | n/a |"
    assert lines[3] == "| c | 0.25 | n/a |"  # missing metric and NaN interval
    with pytest.raises(ValueError, match="missing"):
        summary_markdown(summary.drop(columns=["ci_low"]))
    with pytest.raises(ValueError, match="no metric"):
        summary_markdown(summary.iloc[:0])


def test_comparison_markdown(runs: pd.DataFrame) -> None:
    hot = runs[runs["model"] == "hotspot"]
    loc = runs[runs["model"] == "loc"]
    tests = wilcoxon_holm(hot, {"loc": loc, "none": loc.iloc[:0]}, "ndcg_10")
    lines = comparison_markdown(tests).strip().splitlines()
    assert lines[0] == (
        "| Metric | Baseline | n | Mean (model) | Mean (baseline) | Mean diff | p | p (Holm) "
        "| Cliff's delta | Magnitude |"
    )
    assert lines[2].startswith("| NDCG@10 | loc | 3 | ")
    assert f"| {tests.loc[0, 'p_value']:.3g} |" in lines[2]
    assert lines[3] == "| NDCG@10 | none | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    assert len(comparison_markdown(tests.iloc[:0]).strip().splitlines()) == 2
    assert not any(dash in line for line in lines for dash in DASHES)
    with pytest.raises(ValueError, match="missing"):
        comparison_markdown(tests.drop(columns=["p_holm"]))

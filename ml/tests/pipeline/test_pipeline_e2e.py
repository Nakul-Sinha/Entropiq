"""The whole pipeline over two small repositories built from one source repository."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tdp_core.contracts import FEATURE_INPUT_COLUMNS, INDEX_COLUMNS
from tdp_core.labeling.labels import LABEL_FRAME_COLUMNS
from tdp_ml.pipeline.dataset import DATASET_COLUMNS
from tdp_ml.pipeline.layout import DataLayout
from tdp_ml.pipeline.manifest import read_manifest
from tdp_ml.pipeline.stages import read_plan

SLUGS = ("local__alpha", "local__beta")
KEPT = [2, 3, 4]


def test_manifest_pins_both_repositories(pipeline_run: Any) -> None:
    manifest = read_manifest(DataLayout(pipeline_run.data))
    assert [r.slug for r in manifest.repos] == list(SLUGS)
    assert {r.sha for r in manifest.repos} == {pipeline_run.source.head}
    assert {r.default_branch for r in manifest.repos} == {"main"}
    assert [r.issues for r in manifest.repos] == [False, True]
    assert manifest.settings.horizon_days == 60
    assert manifest.settings.history_window_days is None
    assert manifest.max_snapshots == 3
    again = type(manifest).from_json(manifest.to_json())
    assert again == manifest


def test_plan_keeps_the_newest_snapshots(pipeline_run: Any) -> None:
    layout = DataLayout(pipeline_run.data)
    for slug in SLUGS:
        plan = read_plan(layout, slug)
        assert plan["schedule"] == 5
        assert plan["kept"] == KEPT
        assert plan["dropped_for_budget"] == []
        assert plan["label_fixes"]
        assert plan["feature_fixes"]
        assert [e["index"] for e in plan["snapshots"]] == KEPT


def test_label_bugfix_tables_use_issues_only_for_crawled_repositories(pipeline_run: Any) -> None:
    layout = DataLayout(pipeline_run.data)
    alpha_kw = pq.read_table(layout.bugfix("local__alpha", keyword=True))
    alpha_lab = pq.read_table(layout.bugfix("local__alpha"))
    assert alpha_kw.equals(alpha_lab)
    beta_kw = pq.read_table(layout.bugfix("local__beta", keyword=True)).to_pandas()
    beta_lab = pq.read_table(layout.bugfix("local__beta")).to_pandas()
    fix = pipeline_run.source.shas["d152"]
    assert not beta_kw.loc[beta_kw["sha"] == fix, "is_bugfix"].item()
    row = beta_lab.loc[beta_lab["sha"] == fix]
    assert row["is_bugfix"].item() and row["source"].item() == "issue"
    assert fix in set(pq.read_table(layout.szz("local__beta")).column("fix_sha").to_pylist())
    assert fix not in set(pq.read_table(layout.szz("local__alpha")).column("fix_sha").to_pylist())


def test_features_and_labels_per_kept_snapshot(pipeline_run: Any) -> None:
    layout = DataLayout(pipeline_run.data)
    latent = 0
    for slug in SLUGS:
        for index in KEPT:
            features = pd.read_parquet(layout.features(slug, index))
            assert list(features.columns) == list(INDEX_COLUMNS + FEATURE_INPUT_COLUMNS)
            assert not features["path"].str.startswith("tests/").any()
            assert "pkg/core.py" in set(features["path"])
            assert layout.lineage(slug, index).is_file()
            meta = json.loads(layout.features_meta(slug, index).read_text(encoding="utf-8"))
            assert meta["rows"] == len(features)
            labels = pd.read_parquet(layout.labels(slug, index))
            assert list(labels.columns) == list(LABEL_FRAME_COLUMNS)
            assert labels["path"].tolist() == features["path"].tolist()
            latent += int(labels["latent_defects"].sum())
    assert latent > 0


def test_dataset_frame_and_summary(pipeline_run: Any) -> None:
    layout = DataLayout(pipeline_run.data)
    frame = pd.read_parquet(layout.dataset)
    assert tuple(frame.columns) == DATASET_COLUMNS
    assert set(frame["qid"]) == {102, 103, 104, 202, 203, 204}
    ordered = frame.sort_values(["qid", "path"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(frame, ordered)
    assert frame["qid"].dtype == np.int64
    assert all(frame[c].dtype == np.float32 for c in FEATURE_INPUT_COLUMNS)
    assert frame["grade"].dtype == np.int8
    assert frame["has_latent_defect"].dtype == bool
    summary = json.loads(layout.dataset_summary.read_text(encoding="utf-8"))
    assert summary["queries"] == 6 and summary["repos"] == 2 and summary["rows"] == len(frame)
    assert [e["slug"] for e in summary["per_repo"]] == list(SLUGS)
    assert all(e["passed"] for e in summary["per_repo"])
    beta = summary["per_repo"][1]
    assert beta["values"]["bug_issues"] == 2
    assert summary["per_repo"][0]["values"]["bug_issues"] is None
    stats = json.loads(layout.stats.read_text(encoding="utf-8"))
    assert [r["slug"] for r in stats["repos"]] == list(SLUGS)
    assert stats["passing_repos"] == 2


def test_second_run_skips_finished_work(pipeline_run: Any) -> None:
    layout = DataLayout(pipeline_run.data)
    watched = [layout.commits("local__alpha"), layout.features("local__beta", 3), layout.dataset]
    before = [p.stat().st_mtime_ns for p in watched]
    for stage in ("mine", "features"):
        result = pipeline_run.invoke(stage)
        assert "0 done, 2 skipped, 0 failed" in result.output
    assert "dataset exists" in pipeline_run.invoke("dataset").output
    assert [p.stat().st_mtime_ns for p in watched] == before


def test_force_rebuilds_one_repository_identically(pipeline_run: Any) -> None:
    layout = DataLayout(pipeline_run.data)
    old = pd.read_parquet(layout.features("local__alpha", 4))
    beta_mtime = layout.features("local__beta", 4).stat().st_mtime_ns
    result = pipeline_run.invoke("features", "--force", "--repo", "local__alpha")
    assert "1 done, 0 skipped, 0 failed" in result.output
    pd.testing.assert_frame_equal(pd.read_parquet(layout.features("local__alpha", 4)), old)
    assert layout.features("local__beta", 4).stat().st_mtime_ns == beta_mtime


def test_parallel_mine_gives_equal_tables(pipeline_run: Any) -> None:
    layout = DataLayout(pipeline_run.data)
    old = [pq.read_table(layout.commits(s)) for s in SLUGS]
    result = pipeline_run.invoke("mine", "--force", "--jobs", "2")
    assert "2 done, 0 skipped, 0 failed" in result.output
    assert all(pq.read_table(layout.commits(s)).equals(t) for s, t in zip(SLUGS, old, strict=True))


def test_unknown_repository_is_rejected(pipeline_run: Any) -> None:
    result = pipeline_run.invoke("mine", "--repo", "nope", code=None)
    assert result.exit_code == 2
    assert "nope" in result.output


def test_check_leakage_passes(pipeline_run: Any) -> None:
    result = pipeline_run.invoke("check-leakage", "--sample", "2")
    assert "2 of 2 queries match" in result.output
    report = json.loads(DataLayout(pipeline_run.data).run_record("check-leakage").read_text(encoding="utf-8"))
    assert report["passed"] and report["checked"] == 2


def test_check_leakage_detects_a_changed_feature(pipeline_run: Any) -> None:
    """Last test of the module: it edits the stored dataset and restores it afterwards."""
    layout = DataLayout(pipeline_run.data)
    original = layout.dataset.read_bytes()
    try:
        frame = pd.read_parquet(layout.dataset)
        frame["sloc"] = (frame["sloc"] + 1.0).astype(np.float32)
        frame.to_parquet(layout.dataset, index=False)
        result = pipeline_run.invoke("check-leakage", "--sample", "1", code=1)
        assert "sloc: stored" in result.output
    finally:
        layout.dataset.write_bytes(original)

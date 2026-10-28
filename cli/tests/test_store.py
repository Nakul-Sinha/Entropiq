from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tdp_cli.errors import CliError
from tdp_cli.store import AnalysisMeta, AnalysisStore, LoadedAnalysis
from tdp_core.contracts import FEATURE_INPUT_COLUMNS, AnalysisResult, GraphData, Plan
from tdp_core.serialize import to_jsonable


def _meta(meta: AnalysisMeta, **changes: object) -> AnalysisMeta:
    return dataclasses.replace(meta, **changes)  # type: ignore[arg-type]


def test_round_trip(tmp_path: Path, sample_loaded: LoadedAnalysis) -> None:
    store = AnalysisStore(tmp_path / "analyses")
    path = store.save(sample_loaded)
    meta = sample_loaded.meta
    assert path == tmp_path / "analyses" / meta.slug / f"{meta.sha}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == 1
    assert store.get(meta.slug, meta.sha) == meta
    assert store.get(meta.slug, "0" * 40) is None
    loaded = store.load(meta)
    assert loaded.meta == meta
    assert to_jsonable(loaded.result) == to_jsonable(sample_loaded.result)
    pd.testing.assert_frame_equal(loaded.result.features, sample_loaded.result.features)
    assert loaded.result.features.index.name == "path"
    assert str(loaded.result.features.dtypes.iloc[0]) == "float32"
    top = sample_loaded.result.files[0]
    assert loaded.file(top.path) == top
    assert loaded.file("missing.py") is None
    assert meta.n_files == len(sample_loaded.result.files)
    assert sum(meta.counts.values()) == meta.n_files
    assert list(meta.counts) == ["Critical", "High", "Medium", "Low"]


def test_meta_dict_round_trip(sample_loaded: LoadedAnalysis) -> None:
    data = sample_loaded.meta.to_dict()
    assert data["analyzed_at"].endswith("Z")
    assert AnalysisMeta.from_dict(json.loads(json.dumps(data))) == sample_loaded.meta
    for bad in ({**data, "kind": "ftp"}, {**data, "n_files": "3"}, {**data, "counts": {"x": 1.5}}, [1]):
        with pytest.raises((ValueError, TypeError)):
            AnalysisMeta.from_dict(bad)
    with pytest.raises(KeyError):
        AnalysisMeta.from_dict({k: v for k, v in data.items() if k != "sha"})


def _populate(tmp_path: Path, loaded: LoadedAnalysis) -> tuple[AnalysisStore, list[AnalysisMeta]]:
    store = AnalysisStore(tmp_path / "analyses")
    base = loaded.meta
    metas = [
        _meta(base, sha="aaaa1" + "1" * 35, analyzed_at=datetime(2024, 1, 1, tzinfo=UTC)),
        _meta(base, sha="aaaa2" + "2" * 35, analyzed_at=datetime(2024, 3, 1, tzinfo=UTC)),
        _meta(
            base, sha="bbbb3" + "3" * 35, analyzed_at=datetime(2024, 2, 1, tzinfo=UTC), repo="other", slug="o"
        ),
    ]
    for meta in metas:
        result = dataclasses.replace(loaded.result, sha=meta.sha)
        store.save(LoadedAnalysis(meta, result))
    return store, metas


def test_entries_and_find(tmp_path: Path, sample_loaded: LoadedAnalysis) -> None:
    store, metas = _populate(tmp_path, sample_loaded)
    entries = store.entries()
    assert [m.sha for m in entries] == [metas[1].sha, metas[2].sha, metas[0].sha]
    assert store.find("1") == metas[1]
    assert store.find(" 3 ") == metas[0]
    assert store.find("BBBB") == metas[2]
    assert store.find("aaaa2") == metas[1]
    with pytest.raises(CliError, match="2 saved analyses match aaaa") as info:
        store.find("aaaa")
    assert info.value.exit_code == 2
    with pytest.raises(CliError, match="no analysis number 4"):
        store.find("4")
    with pytest.raises(CliError, match="No saved analysis has a commit"):
        store.find("ffff")
    with pytest.raises(CliError, match="at least 4 characters"):
        store.find("abc")
    with pytest.raises(CliError, match="at least 4 characters"):
        store.find("hello")


def test_find_on_empty_store(tmp_path: Path) -> None:
    store = AnalysisStore(tmp_path / "none")
    assert store.entries() == []
    with pytest.raises(CliError, match="no saved analyses"):
        store.find("1")


def test_partial_and_corrupt_entries_are_skipped(tmp_path: Path, sample_loaded: LoadedAnalysis) -> None:
    store, metas = _populate(tmp_path, sample_loaded)
    folder = store.folder(metas[0].slug)
    (folder / f"{metas[0].sha}.json").unlink()
    (folder / "junk.meta.json").write_text("{nope", encoding="utf-8")
    (folder / "old.meta.json").write_text(json.dumps({"schema": 0}), encoding="utf-8")
    assert {m.sha for m in store.entries()} == {metas[1].sha, metas[2].sha}

    (folder / f"{metas[1].sha}.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(CliError, match="cannot be read; run /analyze again"):
        store.load(metas[1])


def test_missing_features_file_gives_empty_frame(tmp_path: Path, sample_loaded: LoadedAnalysis) -> None:
    store = AnalysisStore(tmp_path / "analyses")
    store.save(sample_loaded)
    meta = sample_loaded.meta
    (store.folder(meta.slug) / f"{meta.sha}.features.parquet").unlink()
    loaded = store.load(meta)
    assert list(loaded.result.features.columns) == list(FEATURE_INPUT_COLUMNS)
    assert len(loaded.result.features) == 0


def test_zero_file_result_round_trip(tmp_path: Path, sample_loaded: LoadedAnalysis) -> None:
    empty = AnalysisResult(
        sha="c" * 40,
        analyzed_at=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=1),
        model_version="dev-x",
        extractor_version="1",
        files=(),
        features=pd.DataFrame(
            {c: pd.Series(dtype="float32") for c in FEATURE_INPUT_COLUMNS},
            index=pd.Index([], dtype=str, name="path"),
        ),
        extras={},
        graph=GraphData((), ()),
        plan=Plan(40.0, (), 0.0, 0.0),
    )
    meta = _meta(
        sample_loaded.meta,
        sha=empty.sha,
        n_files=0,
        counts=dict.fromkeys(("Critical", "High", "Medium", "Low"), 0),
    )
    store = AnalysisStore(tmp_path / "analyses")
    store.save(LoadedAnalysis(meta, empty))
    loaded = store.load(meta)
    assert loaded.result.files == ()
    assert len(loaded.result.features) == 0

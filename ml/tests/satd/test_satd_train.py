import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from tdp_core.features.satd_model import JsonSatdClassifier
from tdp_ml.commands import satd as satd_commands
from tdp_ml.satd import train
from tdp_ml.satd.train import (
    CSV_NAME,
    ZIP_NAME,
    BinaryCounts,
    LopoResult,
    binary_counts,
    candidate_mask,
    download_dataset,
    extract_csv,
    git_blob_sha1,
    keyword_baseline,
    load_dataset,
    lopo_evaluate,
    parity_texts,
    run_training,
)

DEBT = ("todo fix this later", "hack around the bug", "ugly workaround for now", "fixme when possible")
PLAIN = ("returns the value", "parse the input", "user name and path", "list of rows", "default config")
FILLER = ("table", "cache", "item", "index", "data", "file", "loop", "count", "query", "result")


def synthetic_frame(projects: int = 3, rows: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []
    for p in range(projects):
        for i in range(rows):
            debt = i % 3 == 0
            base = rng.choice(DEBT if debt else PLAIN)
            text = f"{base} {rng.choice(FILLER)} {rng.choice(FILLER)}"
            records.append(
                {
                    "project": f"proj{p}",
                    "classification": "DESIGN" if debt else train.NEGATIVE_CLASS,
                    "text": text,
                    "is_debt": debt,
                }
            )
    return pd.DataFrame.from_records(records)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    out = df.rename(columns={"project": "projectname", "text": "commenttext"})[
        ["projectname", "classification", "commenttext"]
    ]
    out.to_csv(path, index=False, lineterminator="\n")
    return path


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


CSV_BYTES = (
    b"projectname,classification,commenttext\nant,DESIGN,// TODO hack\nant,WITHOUT_CLASSIFICATION,ok\n"
)


def test_git_blob_sha1_matches_git() -> None:
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_extract_csv_reads_only_the_csv_member(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "a.zip",
        {
            CSV_NAME: CSV_BYTES,
            f"__MACOSX/._{CSV_NAME}": b"\x00\x05mac",
            "../evil.txt": b"nope",
            "other/readme.txt": b"hi",
        },
    )
    dest = tmp_path / "out"
    csv_path = extract_csv(archive, dest)
    assert csv_path == dest / CSV_NAME
    assert csv_path.read_bytes() == CSV_BYTES
    assert sorted(p.name for p in dest.iterdir()) == [CSV_NAME]
    assert not (tmp_path / "evil.txt").exists()

    nested = make_zip(
        tmp_path / "nested.zip", {f"dataset/{CSV_NAME}": CSV_BYTES, f"__MACOSX/dataset/._{CSV_NAME}": b"x"}
    )
    assert extract_csv(nested, tmp_path / "nested").read_bytes() == CSV_BYTES

    with pytest.raises(ValueError):
        extract_csv(make_zip(tmp_path / "none.zip", {"other.csv": b"x"}), tmp_path / "none")
    with pytest.raises(ValueError):
        extract_csv(
            make_zip(tmp_path / "two.zip", {f"a/{CSV_NAME}": b"x", f"b/{CSV_NAME}": b"y"}), tmp_path / "two"
        )
    with pytest.raises(ValueError, match="bytes"):
        extract_csv(archive, tmp_path / "small", max_bytes=10)
    not_zip = tmp_path / "bad.zip"
    not_zip.write_bytes(b"not a zip")
    with pytest.raises(ValueError):
        extract_csv(not_zip, tmp_path / "bad")


def test_download_dataset_from_file_url(tmp_path: Path) -> None:
    source = make_zip(tmp_path / "source.zip", {CSV_NAME: CSV_BYTES})
    sha = git_blob_sha1(source.read_bytes())
    dest = tmp_path / "satd"

    csv_path = download_dataset(dest, url=source.as_uri(), expected_blob_sha1=sha, wait_s=0)
    assert csv_path == dest / CSV_NAME
    assert csv_path.read_bytes() == CSV_BYTES
    assert (dest / ZIP_NAME).read_bytes() == source.read_bytes()

    # A verified zip is reused, so an unreachable URL does not matter.
    missing = (tmp_path / "missing.zip").as_uri()
    assert download_dataset(dest, url=missing, expected_blob_sha1=sha, retries=0, wait_s=0) == csv_path

    # With force, the bad URL is fetched (and fails after the retries).
    with pytest.raises(OSError):
        download_dataset(dest, url=missing, expected_blob_sha1=sha, retries=1, wait_s=0, force=True)

    wrong = tmp_path / "wrong"
    with pytest.raises(ValueError, match="verification"):
        download_dataset(wrong, url=source.as_uri(), expected_blob_sha1="0" * 40, wait_s=0)
    assert not (wrong / ZIP_NAME).exists()
    assert not (wrong / f"{ZIP_NAME}.part").exists()
    assert not (wrong / CSV_NAME).exists()


@pytest.mark.parametrize("url", ["http://example.com/x.zip", "ftp://example.com/x.zip", "x.zip", ""])
def test_download_rejects_other_schemes(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="https"):
        download_dataset(tmp_path, url=url)
    assert not list(tmp_path.iterdir())


def test_load_dataset_normalises_rows(tmp_path: Path) -> None:
    raw = (
        b"\xef\xbb\xbf ProjectName ,Classification,CommentText\n"
        b'ant,design,"// TODO: hack\nsecond line"\n'
        b"ant,WITHOUT_CLASSIFICATION,NA\n"
        b"jedit , WITHOUT_CLASSIFICATION ,   \n"
        b"jedit,DEFECT,caf\xe9 broken\n"
        b"jedit,WITHOUT_CLASSIFICATION,null\n"
    )
    path = tmp_path / "d.csv"
    path.write_bytes(raw)
    df = load_dataset(path)
    assert list(df.columns) == ["project", "classification", "text", "is_debt"]
    assert df["project"].tolist() == ["ant", "ant", "jedit", "jedit"]
    assert df["classification"].tolist() == [
        "DESIGN",
        "WITHOUT_CLASSIFICATION",
        "DEFECT",
        "WITHOUT_CLASSIFICATION",
    ]
    assert df["text"].tolist() == ["// TODO: hack\nsecond line", "NA", "caf\ufffd broken", "null"]
    assert df["is_debt"].tolist() == [True, False, True, False]
    assert df["is_debt"].dtype == bool
    assert df.index.tolist() == [0, 1, 2, 3]
    assert df.attrs == {"rows_read": 5, "dropped_empty": 1, "replacement_chars": 1}

    (tmp_path / "m.csv").write_bytes(b"projectname,commenttext\nant,x\n")
    with pytest.raises(ValueError, match="classification"):
        load_dataset(tmp_path / "m.csv")
    (tmp_path / "e.csv").write_bytes(b"projectname,classification,commenttext\nant, ,x\n")
    with pytest.raises(ValueError, match="empty classification"):
        load_dataset(tmp_path / "e.csv")
    (tmp_path / "p.csv").write_bytes(b"projectname,classification,commenttext\n,DESIGN,x\n")
    with pytest.raises(ValueError, match="empty projectname"):
        load_dataset(tmp_path / "p.csv")


def test_binary_counts_edge_cases() -> None:
    counts = binary_counts([True, True, False, False, True], [True, False, True, False, True])
    assert counts == BinaryCounts(n=5, positives=3, tp=2, fp=1, fn=1)
    assert counts.precision == pytest.approx(2 / 3)
    assert counts.recall == pytest.approx(2 / 3)
    assert counts.f1 == pytest.approx(2 / 3)
    assert counts.predicted == 3
    none = binary_counts([False, False], [False, False])
    assert (none.precision, none.recall, none.f1) == (0.0, 0.0, 0.0)
    empty = binary_counts([], [])
    assert empty == BinaryCounts(0, 0, 0, 0, 0)
    assert empty.f1 == 0.0
    assert (counts + none).n == 7
    assert counts.to_dict()["f1"] == pytest.approx(0.666667)
    with pytest.raises(ValueError):
        binary_counts([True], [True, False])


def test_candidate_mask_follows_the_keyword_rule() -> None:
    texts = ["TODO later", "a hack", "Work around it", "todos", "hacker", "temp fix", "fine", "XXX", "ugly!"]
    assert candidate_mask(texts).tolist() == [True, True, True, False, False, True, False, True, True]
    assert candidate_mask([]).shape == (0,)


def test_lopo_evaluate_synthetic() -> None:
    df = synthetic_frame()
    result = lopo_evaluate(df, (10.0, 1.0, 0.1, 1.0), max_features=200)
    assert result.projects == ("proj0", "proj1", "proj2")
    assert result.c_grid == (0.1, 1.0, 10.0)
    for c in result.c_grid:
        assert set(result.per_project[c]) == set(result.projects)
        assert len(result.n_iter[c]) == 3
        assert not np.isnan(result.proba[c]).any()
        assert result.proba[c].shape == (len(df),)
        assert sum(b.n for b in result.per_project[c].values()) == len(df)
        assert result.pooled(c).n == len(df)
        assert result.macro_f1(c) > 0.8
    assert result.selected_c() in result.c_grid
    again = lopo_evaluate(df, (0.1, 1.0, 10.0), max_features=200)
    for c in result.c_grid:
        assert np.array_equal(again.proba[c], result.proba[c])
        assert again.per_project[c] == result.per_project[c]

    same = BinaryCounts(30, 10, 8, 2, 2)
    tied = LopoResult(
        projects=("a", "b"),
        c_grid=(0.1, 1.0, 10.0),
        threshold=0.5,
        proba={},
        per_project={c: {"a": same, "b": same} for c in (0.1, 1.0, 10.0)},
        n_iter={},
    )
    assert tied.selected_c() == 0.1
    better = LopoResult(
        projects=("a",),
        c_grid=(0.1, 1.0),
        threshold=0.5,
        proba={},
        per_project={0.1: {"a": BinaryCounts(30, 10, 5, 5, 5)}, 1.0: {"a": same}},
        n_iter={},
    )
    assert better.selected_c() == 1.0
    assert better.macro_precision(1.0) == pytest.approx(0.8)
    assert better.macro_recall(1.0) == pytest.approx(0.8)


def test_lopo_needs_two_projects_and_both_classes() -> None:
    df = synthetic_frame(projects=1)
    with pytest.raises(ValueError, match="two projects"):
        lopo_evaluate(df)
    df = synthetic_frame(projects=2)
    df.loc[df["project"] == "proj1", "is_debt"] = False
    with pytest.raises(ValueError, match="one class"):
        lopo_evaluate(df)
    with pytest.raises(ValueError, match="positive"):
        lopo_evaluate(synthetic_frame(), (0.0,))
    with pytest.raises(ValueError, match="lacks"):
        lopo_evaluate(df.drop(columns=["text"]))


def test_keyword_baseline_and_parity_texts() -> None:
    df = synthetic_frame()
    baseline = keyword_baseline(df)
    assert sorted(baseline) == ["proj0", "proj1", "proj2"]
    assert all(b.n == 90 for b in baseline.values())
    df["text"] = [f"{text} row{i}" for i, text in enumerate(df["text"])]  # unique, so order is checkable
    sample = parity_texts(df, n=20, seed=1)
    assert len(sample) == 20
    assert sample == parity_texts(df, n=20, seed=1)
    debt_texts = set(df.loc[df["is_debt"], "text"])
    assert sum(t in debt_texts for t in sample) == 10
    order = {t: i for i, t in enumerate(df["text"])}
    assert [order[t] for t in sample] == sorted(order[t] for t in sample)
    small = synthetic_frame(projects=1, rows=6)
    assert len(parity_texts(small, n=100)) == 6


def test_run_training_end_to_end(tmp_path: Path) -> None:
    csv_path = write_csv(synthetic_frame(projects=3, rows=60, seed=4), tmp_path / CSV_NAME)
    out = tmp_path / "model" / "satd_model.json"
    report_path = tmp_path / "results" / "lopo_report.json"
    report = run_training(csv_path, out, report_path, c_grid=(0.1, 1.0), max_features=500, jobs=1)

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    for key in ("lopo", "keyword_baseline", "candidates", "final", "export", "dataset", "config"):
        assert key in report
    assert report["lopo"]["selected_c"] in (0.1, 1.0)
    assert set(report["lopo"]["by_c"]) == {"0.1", "1.0"}
    assert set(report["lopo"]["thresholds"]) == {repr(t) for t in train.THRESHOLDS}
    assert report["dataset"]["rows"] == 180
    assert report["dataset"]["pinned_dataset"] is False
    assert report["export"]["parity_max_abs_diff"] < 1e-6
    assert report["export"]["bytes"] == out.stat().st_size <= train.MAX_MODEL_BYTES
    assert report["final"]["vocabulary_size"] <= 500

    clf = JsonSatdClassifier.from_json(out)
    assert clf.n_features == report["final"]["vocabulary_size"]
    assert clf.source_sha256 == report["export"]["sha256"]
    assert clf.metadata["dataset"]["url"] is None
    assert clf.metadata["lopo"]["selected_c"] == report["lopo"]["selected_c"]
    p = clf.predict_proba(["todo fix this later table", "returns the value cache"])
    assert p[0] > 0.5 > p[1]

    before = out.read_bytes()
    with pytest.raises(ValueError, match="max-features"):
        run_training(
            csv_path, out, tmp_path / "r2.json", c_grid=(1.0,), max_features=500, max_bytes=100, jobs=1
        )
    assert out.read_bytes() == before


def make_cli() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        """Test root."""

    satd_commands.register(app)
    return app


@pytest.mark.parametrize(
    "args", [["satd", "--help"], ["satd", "train", "--help"], ["satd", "download", "--help"]]
)
def test_cli_registers_satd_commands(args: list[str]) -> None:
    result = CliRunner().invoke(make_cli(), args)
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_cli_train_reports_missing_csv(tmp_path: Path) -> None:
    result = CliRunner().invoke(make_cli(), ["satd", "train", "--csv", str(tmp_path / "nope.csv")])
    assert result.exit_code == 1
    assert "satd download" in result.output


def test_cli_download_and_train(tmp_path: Path) -> None:
    source = make_zip(
        tmp_path / "src.zip", {CSV_NAME: write_csv(synthetic_frame(), tmp_path / "s.csv").read_bytes()}
    )
    csv_path = download_dataset(
        tmp_path / "satd", url=source.as_uri(), expected_blob_sha1=git_blob_sha1(source.read_bytes())
    )
    out = tmp_path / "m.json"
    report = tmp_path / "r.json"
    args = ["satd", "train", "--csv", str(csv_path), "--out", str(out), "--report", str(report)]
    args += ["--c", "1.0", "--max-features", "300", "--jobs", "1"]
    result = CliRunner().invoke(make_cli(), args)
    assert result.exit_code == 0, result.output
    assert "macro F1" in result.output
    assert "proj2" in result.output
    assert out.is_file()
    assert report.is_file()

    too_small = CliRunner().invoke(
        make_cli(), [*args, "--max-bytes", "10", "--out", str(tmp_path / "x.json")]
    )
    assert too_small.exit_code == 1
    assert not (tmp_path / "x.json").exists()


@pytest.mark.network
def test_download_real_dataset(tmp_path: Path) -> None:
    csv_path = download_dataset(tmp_path, wait_s=5)
    assert csv_path.read_bytes().startswith(b"projectname,classification,commenttext")
    df = load_dataset(csv_path)
    assert len(df) > 60_000
    assert df["project"].nunique() == 10


def test_cli_download_prints_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = write_csv(synthetic_frame(projects=2, rows=30), tmp_path / CSV_NAME)
    calls: list[tuple[Path, bool]] = []

    def fake_download(dest: Path, *, force: bool = False) -> Path:
        calls.append((dest, force))
        return csv_path

    monkeypatch.setattr(train, "download_dataset", fake_download)
    result = CliRunner().invoke(make_cli(), ["satd", "download", "--dest", str(tmp_path / "d"), "--force"])
    assert result.exit_code == 0, result.output
    assert calls == [(tmp_path / "d", True)]
    assert "rows: 60" in result.output
    assert "debt rows: 20" in result.output
    assert "projects: 2" in result.output

    def broken(dest: Path, *, force: bool = False) -> Path:
        raise ValueError("checksum mismatch")

    monkeypatch.setattr(train, "download_dataset", broken)
    failed = CliRunner().invoke(make_cli(), ["satd", "download", "--dest", str(tmp_path / "d")])
    assert failed.exit_code == 1
    assert "checksum mismatch" in failed.output

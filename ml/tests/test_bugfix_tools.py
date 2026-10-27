import json
import math
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import typer
from typer.testing import CliRunner

from tdp_core.contracts import COMMITS_SCHEMA, FILE_CHANGES_SCHEMA
from tdp_core.labeling.bugfix import classify_bugfix
from tdp_core.labeling.github_issues import ISSUES_SCHEMA
from tdp_core.testing import RepoBuilder
from tdp_ml.commands import bugfix_eval
from tdp_ml.tools import github
from tdp_ml.tools.sample_bugfix import (
    KEY_COLUMNS,
    SAMPLE_COLUMNS,
    allocate,
    key_path,
    sample_bugfix,
    write_sample,
)
from tdp_ml.tools.score_bugfix import read_key, read_labels, score_bugfix, wilson_interval

_T0 = datetime(2021, 1, 1, tzinfo=UTC)


def _tables(specs: Sequence[tuple[str, str, str]]) -> tuple[pa.Table, pa.Table]:
    """``(sha, message, path)`` rows -> commits and file_changes tables (one change per commit)."""
    n = len(specs)
    commits = pa.Table.from_pydict(
        {
            "sha": [s[0] for s in specs],
            "author_name": ["Ana"] * n,
            "author_email": ["ana@example.com"] * n,
            "author_time": [_T0 + timedelta(days=i) for i in range(n)],
            "parents": [[] for _ in range(n)],
            "subject": [s[1].split("\n")[0] for s in specs],
            "message": [s[1] for s in specs],
            "files_touched": [1] * n,
            "is_bulk": [False] * n,
        },
        schema=COMMITS_SCHEMA,
    )
    changes = pa.Table.from_pydict(
        {
            "sha": [s[0] for s in specs] + [specs[0][0]],
            "path": [s[2] for s in specs] + ["CHANGES.rst"],
            "old_path": [None] * (n + 1),
            "added": [3] * (n + 1),
            "deleted": [1] * (n + 1),
            "is_binary": [False] * (n + 1),
        },
        schema=FILE_CHANGES_SCHEMA,
    )
    return commits, changes


def _synthetic() -> tuple[pa.Table, pa.Table, pa.Table]:
    specs: list[tuple[str, str, str]] = []
    specs += [(f"i{k:03d}", f"Handle #{k + 1}", "pkg/a.py") for k in range(5)]
    specs += [(f"k{k:03d}", f"Fix crash number {k}", "pkg/a.py") for k in range(30)]
    specs += [(f"n{k:03d}", f"Add feature {k}", "pkg/b.py") for k in range(60)]
    specs += [(f"t{k:03d}", f"Fix test {k}", "tests/test_a.py") for k in range(10)]  # ineligible
    commits, changes = _tables(specs)
    issues = pa.Table.from_pydict(
        {
            "number": list(range(1, 6)),
            "created_at": [_T0] * 5,
            "closed_at": [None] * 5,
            "labels": [["bug"]] * 5,
            "is_bug": [True] * 5,
        },
        schema=ISSUES_SCHEMA,
    )
    return commits, changes, classify_bugfix(commits, changes, issues)


# ------------------------------------------------------------------ allocation and sampling
def test_allocate_shares_spare_slots() -> None:
    assert allocate({"issue": 3, "keyword": 50, "none": 500}, 20) == {"issue": 3, "keyword": 9, "none": 8}
    assert allocate({"issue": 3, "keyword": 5, "none": 0}, 100) == {"issue": 3, "keyword": 5, "none": 0}
    assert allocate({"a": 10, "b": 10, "c": 10}, 2) == {"a": 1, "b": 1, "c": 0}
    with pytest.raises(ValueError):
        allocate({"issue": 3}, 0)
    with pytest.raises(ValueError):
        allocate({"issue": -1}, 3)


def test_sample_is_stratified_blind_and_deterministic() -> None:
    commits, changes, bugfix = _synthetic()
    sample = sample_bugfix(commits, changes, bugfix, n=20, seed=7, url_prefix="https://github.com/o/r")
    rows, key = sample.rows, sample.key
    assert tuple(rows.columns) == SAMPLE_COLUMNS
    assert tuple(key.columns) == KEY_COLUMNS
    assert rows["row"].tolist() == list(range(1, 21))
    assert not rows["sha"].str.startswith("t").any()  # test-only commits are not in the population
    assert key["sha"].tolist() == sorted(rows["sha"])
    counts = key.groupby("stratum").size().to_dict()
    assert (
        counts
        == allocate({"issue": 5, "keyword": 30, "none": 60}, 20)
        == {"issue": 5, "keyword": 8, "none": 7}
    )
    sizes = key.groupby("stratum")["stratum_size"].first().to_dict()
    assert sizes == {"issue": 5, "keyword": 30, "none": 60}
    assert set(key.groupby("stratum")["sample_size"].first().to_dict().items()) == set(counts.items())
    assert dict(zip(key["stratum"], key["predicted"], strict=True)) == {"issue": 1, "keyword": 1, "none": 0}
    assert (rows["label"] == "").all()
    assert (rows["note"] == "").all()
    first = rows.iloc[0]
    assert first["url"] == f"https://github.com/o/r/commit/{first['sha']}"
    assert first["date"].startswith("2021-")
    i000 = rows.set_index("sha").loc["i000"]  # the issue stratum is small enough to be sampled whole
    assert i000["subject"] == "Handle #1"
    assert i000["source_files"] == "pkg/a.py (+3/-1)"
    assert i000["other_files"] == "CHANGES.rst (+3/-1)"

    again = sample_bugfix(commits, changes, bugfix, n=20, seed=7, url_prefix="https://github.com/o/r")
    assert again.rows.equals(rows)
    assert again.key.equals(key)
    other = sample_bugfix(commits, changes, bugfix, n=20, seed=8)
    assert other.rows["sha"].tolist() != rows["sha"].tolist()
    assert (other.rows["url"] == "").all()


def test_sample_takes_everything_when_n_is_large() -> None:
    commits, changes, bugfix = _synthetic()
    sample = sample_bugfix(commits, changes, bugfix, n=1000)
    assert len(sample.rows) == 95
    assert set(sample.key["sample_size"]) == {5, 30, 60}


def test_sample_errors() -> None:
    commits, changes, bugfix = _synthetic()
    with pytest.raises(ValueError, match="missing"):
        sample_bugfix(commits, changes, bugfix.slice(1), n=20)
    with pytest.raises(ValueError, match="strata"):
        sample_bugfix(commits, changes, bugfix, n=2)


def test_long_file_lists_are_cut() -> None:
    commits, _ = _tables([("a", "Fix crash", "pkg/a.py")])
    changes = pa.Table.from_pydict(
        {
            "sha": ["a"] * 12,
            "path": [f"pkg/m{i:02d}.py" for i in range(12)],
            "old_path": ["pkg/old.py"] + [None] * 11,
            "added": [1] * 12,
            "deleted": [0] * 12,
            "is_binary": [False] * 12,
        },
        schema=FILE_CHANGES_SCHEMA,
    )
    sample = sample_bugfix(commits, changes, classify_bugfix(commits, changes), n=1)
    listed = sample.rows.iloc[0]["source_files"]
    assert listed.startswith("pkg/old.py => pkg/m00.py (+1/-0); pkg/m01.py (+1/-0)")
    assert listed.endswith("; ... 2 more")


def test_write_sample_protects_labels(tmp_path: Path) -> None:
    commits, changes, bugfix = _synthetic()
    sample = sample_bugfix(commits, changes, bugfix, n=10)
    out = tmp_path / "eval" / "o__r.sample.csv"
    paths = write_sample(sample, out)
    assert paths == (out, tmp_path / "eval" / "o__r.sample.key.csv")
    assert key_path(out) == paths[1]
    with pytest.raises(FileExistsError):
        write_sample(sample, out)
    write_sample(sample, out, force=True)
    assert b"\r\n" not in out.read_bytes()
    assert read_key(paths[1]).equals(
        sample.key.astype({"stratum_size": int, "sample_size": int, "predicted": int})
    )


# ------------------------------------------------------------------ scoring
def _key_and_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    design = {"issue": (10, [1, 1]), "keyword": (40, [1, 1, 1, 0]), "none": (150, [1, 0, 0, 0, 0])}
    key_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for stratum, (size, labels) in design.items():
        for i, label in enumerate(labels):
            sha = f"{stratum}{i}"
            key_rows.append(
                {
                    "sha": sha,
                    "stratum": stratum,
                    "stratum_size": size,
                    "sample_size": len(labels),
                    "predicted": int(stratum != "none"),
                }
            )
            label_rows.append({"sha": sha, "label": label})
    return pd.DataFrame(key_rows), pd.DataFrame(label_rows)


def test_score_uses_design_weights() -> None:
    key, labels = _key_and_labels()
    score = score_bugfix(labels, key)
    assert score.labelled == 11
    assert score.precision == pytest.approx(0.8)  # TP 40 / PP 50
    assert score.recall == pytest.approx(40 / 70)  # AP 70
    assert score.f1 == pytest.approx(2 / 3)
    by_name = {s.stratum: s for s in score.strata}
    assert [s.stratum for s in score.strata] == ["issue", "keyword", "none"]
    assert (by_name["keyword"].labelled, by_name["keyword"].bugs, by_name["keyword"].rate) == (4, 3, 0.75)
    assert by_name["none"].population == 150
    assert 0 < by_name["none"].ci_low < 0.2 < by_name["none"].ci_high < 1
    data = score.to_dict()
    assert data["strata"][0]["stratum"] == "issue"
    json.dumps(data, allow_nan=False)


def test_rescore_with_new_predictions() -> None:
    key, labels = _key_and_labels()
    predicted = {sha: sha in {"issue0", "issue1", "keyword0", "none0"} for sha in key["sha"]}
    score = score_bugfix(labels, key, predicted)
    # TP = 5 + 5 + 10 + 30 = 50, PP = 50, AP = 70
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(50 / 70)
    with pytest.raises(ValueError, match="predicted"):
        score_bugfix(labels, key, {"issue0": True})


def test_score_errors() -> None:
    key, labels = _key_and_labels()
    with pytest.raises(ValueError, match="none"):
        score_bugfix(labels[labels["sha"] != "none0"].iloc[:6], key)
    with pytest.raises(ValueError, match="missing from the key"):
        score_bugfix(pd.concat([labels, pd.DataFrame([{"sha": "zzz", "label": 1}])]), key)


def test_score_without_predictions_or_bugs_is_nan() -> None:
    key, labels = _key_and_labels()
    labels = labels.assign(label=0)
    score = score_bugfix(labels, key)
    assert math.isnan(score.recall)
    assert score.precision == 0.0
    assert score.to_dict()["recall"] is None


def test_wilson_interval() -> None:
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_read_labels(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    pd.DataFrame(
        {"row": [1, 2, 3], "sha": ["a", "b", "c"], "label": [" 1", "", "0 "], "note": ["", "", ""]}
    ).to_csv(path, index=False)
    labels = read_labels(path)
    assert labels.to_dict("list") == {"sha": ["a", "c"], "label": [1, 0]}
    pd.DataFrame({"row": [1, 2], "sha": ["a", "b"], "label": ["1", "2"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="bad rows: 2"):
        read_labels(path)


# ------------------------------------------------------------------ token helper
def test_github_token_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "  from-env  ")
    assert github.github_token() == "from-env"


def test_github_token_from_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "gh-binary")
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="tok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert github.github_token() == "tok"
    assert calls == [["gh-binary", "auth", "token"]]

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 1, stdout="", stderr="no"),
    )
    assert github.github_token() is None
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert github.github_token() is None


def test_split_slug() -> None:
    assert github.split_slug("psf__requests") == ("psf", "requests")
    assert github.split_slug("encode__django-rest-framework") == ("encode", "django-rest-framework")
    for bad in ("psf", "__requests", "psf__"):
        with pytest.raises(ValueError):
            github.split_slug(bad)


# ------------------------------------------------------------------ commands end to end
def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        """Test app."""

    bugfix_eval.register(app)
    return app


def _history(root: Path) -> Path:
    builder = RepoBuilder(root)
    day = timedelta(days=1)
    builder.commit({"pkg/a.py": "A = 1\n", "tests/test_a.py": "def test():\n    pass\n"}, "Add a", _T0)
    builder.commit({"pkg/a.py": "A = 2\n"}, "Fix wrong value of A (#1)", _T0 + day)
    builder.commit({"pkg/b.py": "B = 1\n"}, "Add b", _T0 + 2 * day)
    builder.commit({"pkg/b.py": "B = 2\n"}, "Fix crash in b", _T0 + 3 * day)
    builder.commit({"tests/test_a.py": "def test():\n    assert 1\n"}, "Fix test", _T0 + 4 * day)
    builder.commit({"pkg/c.py": "C = 1\n"}, "Add c", _T0 + 5 * day)
    return builder.path


def test_sample_and_score_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _history(tmp_path / "repo")
    monkeypatch.setenv("TDP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TDP_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(bugfix_eval, "github_token", lambda: pytest.fail("no token lookup expected"))
    runner = CliRunner()
    args = ["sample-bugfix", "--repo", "test__repo", "--repo-dir", str(repo), "--no-issues", "--size", "4"]
    result = runner.invoke(_app(), args)
    assert result.exit_code == 0, result.output
    sample_csv = tmp_path / "results" / "bugfix_eval" / "test__repo.sample.csv"
    assert sample_csv.exists()
    assert key_path(sample_csv).exists()
    assert "keyword  population      2" in result.output

    again = runner.invoke(_app(), args)
    assert again.exit_code == 1

    rows = pd.read_csv(sample_csv, dtype=str, keep_default_na=False)
    rows["label"] = ["1" if s.startswith("Fix") else "0" for s in rows["subject"]]
    rows.to_csv(sample_csv, index=False, lineterminator="\n")
    scored = runner.invoke(_app(), ["score-bugfix", str(sample_csv)])
    assert scored.exit_code == 0, scored.output
    report = json.loads((tmp_path / "results" / "bugfix_eval" / "test__repo.sample.score.json").read_text())
    assert report["precision"] == pytest.approx(1.0)
    assert report["recall"] == pytest.approx(1.0)
    assert report["labelled"] == 4

    bugfix_parquet = tmp_path / "none.parquet"
    pq.write_table(pa.table({"sha": rows["sha"].tolist(), "is_bugfix": [False] * len(rows)}), bugfix_parquet)
    rescored = runner.invoke(_app(), ["score-bugfix", str(sample_csv), "--bugfix", str(bugfix_parquet)])
    assert rescored.exit_code == 0, rescored.output
    assert "recall 0.000" in rescored.output


def test_sample_command_uses_cached_issues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _history(tmp_path / "repo")
    data = tmp_path / "data"
    monkeypatch.setenv("TDP_DATA_DIR", str(data))
    monkeypatch.setenv("TDP_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(bugfix_eval, "github_token", lambda: pytest.fail("no token lookup expected"))
    cache = data / "issues" / "test__repo.parquet"
    cache.parent.mkdir(parents=True)
    issues = pa.Table.from_pydict(
        {"number": [1], "created_at": [_T0], "closed_at": [None], "labels": [["bug"]], "is_bug": [True]},
        schema=ISSUES_SCHEMA,
    )
    pq.write_table(issues, cache)
    result = CliRunner().invoke(_app(), ["sample-bugfix", "--repo", "test__repo", "--repo-dir", str(repo)])
    assert result.exit_code == 0, result.output
    assert "issue    population      1" in result.output


def test_sample_command_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TDP_DATA_DIR", str(tmp_path / "data"))
    runner = CliRunner()
    missing = runner.invoke(_app(), ["sample-bugfix", "--repo", "a__b", "--no-issues"])
    assert missing.exit_code == 2
    bad = runner.invoke(_app(), ["sample-bugfix", "--repo", "nounderscore", "--no-issues"])
    assert bad.exit_code == 2

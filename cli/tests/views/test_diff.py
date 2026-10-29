from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tdp_cli.commands.diff import (
    ChangedFile,
    DiffRange,
    complete_diff,
    diff_json,
    diff_markdown,
    diff_range,
    diff_view,
    parse_name_status,
    split_changes,
)
from tdp_cli.errors import CliError, UsageError
from tdp_cli.main import build_app
from tdp_cli.session import Session
from tdp_cli.shell import dispatch
from tdp_cli.store import LoadedAnalysis
from tdp_core.testing import RepoBuilder

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RANKED_CHANGES = {"app/api/helpers.py", "app/api/routes.py", "app/api/schema.py", "app/views.py"}


def _out(session: Session) -> str:
    text = session.console.file.getvalue()  # type: ignore[attr-defined]
    session.console.file.seek(0)  # type: ignore[attr-defined]
    session.console.file.truncate()  # type: ignore[attr-defined]
    return text


def test_parse_name_status() -> None:
    raw = (
        b"R100\x00old/a.py\x00new/a.py\x00"
        b"C75\x00src.py\x00copy.py\x00"
        b"M\x00b.py\x00A\x00c.py\x00D\x00d.py\x00T\x00e.py\x00X\x00f.py\x00"
        b"M\x00caf\xe9.py\x00"
    )
    changes = parse_name_status(raw)
    assert changes == [
        ChangedFile("b.py", "modified"),
        ChangedFile("c.py", "added"),
        ChangedFile("caf�.py", "modified"),
        ChangedFile("copy.py", "copied", "src.py"),
        ChangedFile("d.py", "deleted"),
        ChangedFile("e.py", "type changed"),
        ChangedFile("f.py", "modified"),
        ChangedFile("new/a.py", "renamed", "old/a.py"),
    ]
    assert parse_name_status(b"") == []
    assert parse_name_status(b"M\x00a.py\x00R90\x00only-old.py\x00") == [ChangedFile("a.py", "modified")]
    assert parse_name_status(b"M\x00") == []


def test_diff_range(linked_repo: Path, linked_base: str, linked_loaded: LoadedAnalysis) -> None:
    head = linked_loaded.result.sha
    diff = diff_range(linked_repo, "HEAD~8", head, display="linked")
    assert diff.base == "HEAD~8" and diff.base_sha == linked_base and diff.merge_base == linked_base
    assert diff.head_sha == head
    assert [(c.path, c.change, c.old_path) for c in diff.changes] == [
        ("app/api/helpers.py", "renamed", "app/helpers.py"),
        ("app/api/routes.py", "modified", None),
        ("app/api/schema.py", "modified", None),
        ("app/legacy.py", "deleted", None),
        ("app/views.py", "modified", None),
        ("docs.txt", "added", None),
        ("tests/test_core.py", "modified", None),
    ]
    ranked, unranked = split_changes(linked_loaded.result, diff.changes)
    assert {r.path for r in ranked} == RANKED_CHANGES
    assert [r.rank for r in ranked] == sorted(r.rank for r in ranked)
    assert [c.path for c in unranked] == ["app/legacy.py", "docs.txt", "tests/test_core.py"]
    assert diff_range(linked_repo, "HEAD", head, display="linked").changes == ()


def test_diff_range_errors(linked_repo: Path, linked_loaded: LoadedAnalysis, tmp_path: Path) -> None:
    head = linked_loaded.result.sha
    with pytest.raises(UsageError, match="'-x' is not a valid revision"):
        diff_range(linked_repo, "-x", head, display="linked")
    with pytest.raises(CliError, match="'nope' does not name a commit in linked") as unknown:
        diff_range(linked_repo, "nope", head, display="linked")
    assert (
        unknown.value.exit_code == 2 and unknown.value.hint and "fetch the full history" in unknown.value.hint
    )
    with pytest.raises(CliError, match="no longer exists") as gone:
        diff_range(tmp_path / "gone", "HEAD", head, display="gone")
    assert gone.value.exit_code == 1
    with pytest.raises(CliError, match="git failed: ") as failed:
        diff_range(linked_repo, "HEAD", "0" * 40, display="linked")
    assert failed.value.exit_code == 1


def test_diff_range_without_common_history(tmp_path: Path) -> None:
    builder = RepoBuilder(tmp_path / "split")
    when = datetime(2024, 1, 1, tzinfo=UTC)
    builder.commit({"a.py": "A = 1\n"}, "First", when)
    subprocess.run(["git", "-C", str(builder.path), "checkout", "-q", "--orphan", "other"], check=True)
    other = builder.commit({"b.py": "B = 1\n"}, "Unrelated", when)
    with pytest.raises(CliError, match=f"main and {other[:7]} have no common history") as split:
        diff_range(builder.path, "main", other, display="split")
    assert split.value.exit_code == 2


def test_diff_range_timeout(
    linked_repo: Path, linked_loaded: LoadedAnalysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tdp_core.mining.git as git

    real = git.run_git

    def slow(args: list[str], cwd: Path | None, **kwargs: Any) -> bytes:
        if args[0] == "diff":
            raise git.TransientGitError("git diff timed out after 120 s")
        return real(args, cwd, **kwargs)

    monkeypatch.setattr(git, "run_git", slow)
    with pytest.raises(CliError, match="git timed out"):
        diff_range(linked_repo, "HEAD~8", linked_loaded.result.sha, display="linked")


def test_diff_command(linked_session: Session, linked_base: str) -> None:
    dispatch(linked_session, "/diff --base HEAD~8")
    out = _out(linked_session)
    assert "Files changed in HEAD~8..." in out and f"(merge base {linked_base[:7]})" in out
    assert "Change" in out and "renamed" in out and "modified" in out
    for path in RANKED_CHANGES:
        assert path in out
    assert "4 of 7 changed files are ranked: " in out
    assert (
        "Not ranked (deleted, tests, skipped or not Python): app/legacy.py (deleted), docs.txt (added)" in out
    )
    assert "tests/test_core.py" in out
    dispatch(linked_session, "/diff HEAD~8 --top 2")
    out = _out(linked_session)
    assert "Showing 2 of 4 ranked files." in out
    dispatch(linked_session, "/diff --base HEAD")
    assert "No files changed between HEAD and" in _out(linked_session)
    dispatch(linked_session, "/diff HEAD~1")
    out = _out(linked_session)
    assert "1 of 1 changed file is ranked" in out and "Not ranked" not in out


def test_diff_command_errors(linked_session: Session) -> None:
    dispatch(linked_session, "/diff")
    out = _out(linked_session)
    assert "Error: say which commit to compare with: --base REV" in out
    assert "Usage: /diff --base REV [--top N]" in out
    dispatch(linked_session, "/diff main --base HEAD~1")
    assert "Error: give the base once" in _out(linked_session)
    dispatch(linked_session, "/diff --base nope")
    assert "'nope' does not name a commit in linked" in _out(linked_session)
    dispatch(linked_session, "/diff --base -x")
    assert "Error: '-x' is not a valid revision" in _out(linked_session)
    assert [s.text for s in complete_diff(linked_session, [], "--")] == ["--base", "--top"]
    assert [s.text for s in complete_diff(linked_session, ["--base"], "")] == []
    loaded = linked_session.current
    assert loaded is not None
    linked_session.current = dataclasses.replace(loaded, result=dataclasses.replace(loaded.result, files=()))
    dispatch(linked_session, "/diff --base HEAD~1")
    assert "This analysis has no ranked files." in _out(linked_session)


def test_diff_view_without_ranked_changes(linked_session: Session, linked_loaded: LoadedAnalysis) -> None:
    head = linked_loaded.result.sha
    many = tuple(ChangedFile(f"docs/n{i:02d}.txt", "added") for i in range(12))
    diff = DiffRange("main", head, head, head, many)
    linked_session.console.print(diff_view(linked_loaded, diff))
    out = _out(linked_session)
    assert "None of the 12 changed files are ranked." in out and "and 2 more" in out
    one = DiffRange("main", head, head, head, many[:1])
    linked_session.console.print(diff_view(linked_loaded, one))
    assert "The changed file is not ranked." in _out(linked_session)


def test_diff_json_and_markdown(linked_repo: Path, linked_loaded: LoadedAnalysis, linked_base: str) -> None:
    diff = diff_range(linked_repo, "HEAD~8", linked_loaded.result.sha, display="linked")
    doc = diff_json(linked_loaded, diff)
    assert (
        doc["schema"] == "entropiq.diff.v1" and doc["base"] == "HEAD~8" and doc["merge_base"] == linked_base
    )
    summary = doc["summary"]
    assert summary["changed"] == 7 and summary["ranked"] == 4
    assert summary["critical"] + summary["high"] + summary["medium"] + summary["low"] == 4
    helpers = next(f for f in doc["files"] if f["path"] == "app/api/helpers.py")
    assert helpers["change"] == "renamed" and helpers["old_path"] == "app/helpers.py" and "rank" in helpers
    assert doc["unranked"][0] == {"path": "app/legacy.py", "change": "deleted", "old_path": None}
    assert len(diff_json(linked_loaded, diff, top=1)["files"]) == 1
    text = diff_markdown(linked_loaded, diff)
    assert "This report covers 4 of 9 ranked files" in text
    assert text.endswith(
        f"Compared with `HEAD~8` (merge base `{linked_base[:12]}`): 7 changed files, 4 of them ranked.\n"
    )
    assert not text.endswith("\n\n")


def test_tdp_diff(saved_linked: LoadedAnalysis, linked_repo: Path, linked_base: str) -> None:
    app = build_app()
    data = runner.invoke(
        app, ["diff", str(linked_repo), "--base", "HEAD~8", "--dev-model", "--format", "json"]
    )
    assert data.exit_code == 0, data.stderr
    assert "Using the saved analysis" in data.stderr
    doc = json.loads(data.stdout)
    assert doc["summary"]["ranked"] == 4 and len(doc["unranked"]) == 3
    md = runner.invoke(
        app, ["diff", str(linked_repo), "--base", linked_base, "--dev-model", "-f", "md", "--top", "2"]
    )
    assert md.exit_code == 0, md.stderr
    assert (
        "This report covers 4 of 9 ranked files" in md.stdout and "Showing the top 2 of 4 files." in md.stdout
    )
    table = runner.invoke(app, ["diff", str(linked_repo), "--base", "HEAD~8", "--dev-model"])
    assert table.exit_code == 0 and "4 of 7 changed files are ranked" in ANSI.sub("", table.stdout)
    missing = runner.invoke(app, ["diff", str(linked_repo), "--dev-model"])
    assert missing.exit_code == 2
    unknown = runner.invoke(app, ["diff", str(linked_repo), "--base", "nope", "--dev-model"])
    assert unknown.exit_code == 2 and "does not name a commit" in unknown.stderr
    option = runner.invoke(app, ["diff", str(linked_repo), "--base=-x", "--dev-model"])
    assert option.exit_code == 2 and "is not a valid revision" in option.stderr
    assert "Using the saved analysis" not in option.stderr

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tdp_cli.commands._views import file_slocs, ranked_by_path
from tdp_cli.commands.tree import build_tree, complete_tree, tree_json, tree_view
from tdp_cli.errors import CliError
from tdp_cli.main import build_app
from tdp_cli.session import Session
from tdp_cli.shell import dispatch
from tdp_cli.store import LoadedAnalysis
from tdp_cli.theme import ASCII_GLYPHS
from tdp_core.contracts import AnalysisResult

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _out(session: Session) -> str:
    text = session.console.file.getvalue()  # type: ignore[attr-defined]
    session.console.file.seek(0)  # type: ignore[attr-defined]
    session.console.file.truncate()  # type: ignore[attr-defined]
    return text


def _top(result: AnalysisResult, prefix: str) -> str:
    return next(f.path for f in result.files if f.path.startswith(prefix))


def test_build_tree_aggregates(linked_result: AnalysisResult) -> None:
    slocs = file_slocs(linked_result)
    root = build_tree(linked_result)
    assert (root.name, root.path, root.n_files) == ("", "", 9)
    assert root.sloc == sum(slocs.values())
    assert root.top is not None and root.top.rank == 1
    assert list(root.folders) == ["app"] and [f.path for f in root.files] == ["setup.py"]
    app = root.folders["app"]
    assert (app.path, app.n_files) == ("app/", 8)
    assert app.top is not None and app.top.path == _top(linked_result, "app/")
    api = app.folders["api"]
    assert (api.name, api.path, api.n_files) == ("api", "app/api/", 4)
    assert api.sloc == sum(v for k, v in slocs.items() if k.startswith("app/api/"))
    assert api.top is not None and api.top.path == _top(linked_result, "app/api/")
    part = build_tree(linked_result, ".\\app\\api\\")
    assert (part.name, part.path, part.n_files) == ("api", "app/api/", 4)
    assert sorted(f.path for f in part.files) == [
        "app/api/__init__.py",
        "app/api/helpers.py",
        "app/api/routes.py",
        "app/api/schema.py",
    ]
    empty = build_tree(dataclasses.replace(linked_result, files=()))
    assert empty.n_files == 0 and empty.top is None


def test_build_tree_errors(linked_result: AnalysisResult) -> None:
    with pytest.raises(CliError, match=r"app/core\.py is a file") as as_file:
        build_tree(linked_result, "app/core.py")
    assert as_file.value.exit_code == 2 and as_file.value.hint == "Use /file app/core.py to see it."
    with pytest.raises(CliError, match="No ranked files under ap/") as missing:
        build_tree(linked_result, "ap")
    assert missing.value.exit_code == 2
    with pytest.raises(CliError, match="No ranked files under tests/"):
        build_tree(linked_result, "tests")


def test_tree_command(linked_session: Session) -> None:
    dispatch(linked_session, "/tree")
    out = _out(linked_session)
    assert out.splitlines()[0].startswith("linked  9 files")
    assert "app/  8 files" in out and "api/  4 files" in out
    assert "highest" in out and "lines" in out
    assert "core.py" not in out and "setup.py" not in out
    assert "Folders only; add --files to list the files with their rank." in out
    dispatch(linked_session, "/tree --files")
    out = _out(linked_session)
    ranks = ranked_by_path(linked_session.current.result)  # type: ignore[union-attr]
    assert f"#{ranks['app/core.py'].rank}" in out and "core.py  " in out
    assert "setup.py" in out and "Folders only" not in out
    dispatch(linked_session, "/tree app/api --files")
    out = _out(linked_session)
    assert out.splitlines()[0].startswith("app/api/  4 files")
    assert "routes.py" in out and "core.py" not in out
    dispatch(linked_session, "/tree --depth 1")
    out = _out(linked_session)
    assert "app/  8 files" in out and "api/" not in out


def test_tree_command_errors_and_completion(linked_session: Session) -> None:
    dispatch(linked_session, "/tree app/core.py")
    out = _out(linked_session)
    assert "Error: app/core.py is a file" in out and "Use /file app/core.py" in out
    dispatch(linked_session, "/tree --depth x")
    assert "Usage: /tree [FOLDER] [--depth N] [--files]" in _out(linked_session)

    def texts(args: list[str], word: str) -> list[str]:
        return [s.text for s in complete_tree(linked_session, args, word)]

    assert texts([], "--") == ["--depth", "--files"]
    assert texts(["--files"], "--") == ["--depth"]
    assert texts(["--depth"], "") == []
    assert texts([], "app/") == ["app/", "app/api/"]
    assert texts(["--depth", "2"], "app/a") == ["app/api/"]
    assert texts(["app/"], "") == []
    loaded = linked_session.current
    assert loaded is not None
    linked_session.current = dataclasses.replace(loaded, result=dataclasses.replace(loaded.result, files=()))
    dispatch(linked_session, "/tree")
    assert "This analysis has no ranked files." in _out(linked_session)


def test_tree_view_ascii_and_empty(linked_session: Session, linked_result: AnalysisResult) -> None:
    root = build_tree(linked_result)
    linked_session.console.print(
        tree_view(root, title="x", slocs=file_slocs(linked_result), files=True, glyphs=ASCII_GLYPHS)
    )
    out = _out(linked_session)
    assert " | " in out and "* " in out
    empty = build_tree(dataclasses.replace(linked_result, files=()))
    linked_session.console.print(tree_view(empty, title="empty", slocs={}))
    assert "empty  no ranked files" in _out(linked_session)


def test_tree_json(linked_result: AnalysisResult) -> None:
    slocs = file_slocs(linked_result)
    root = build_tree(linked_result)
    node = tree_json(root, slocs=slocs)
    assert (
        node["files"] == 9 and node["collapsed"] is False and node["top_file"] == linked_result.files[0].path
    )
    assert [c["name"] for c in node["children"]] == ["app"]
    app = node["children"][0]
    assert [c["path"] for c in app["children"]] == ["app/api/"]
    shallow = tree_json(root, slocs=slocs, depth=1)
    assert shallow["children"][0]["collapsed"] is True and shallow["children"][0]["children"] == []
    with_files = tree_json(root, slocs=slocs, files=True)
    leaves = [c for c in with_files["children"] if "rank" in c]
    assert [c["path"] for c in leaves] == ["setup.py"]
    assert set(leaves[0]) == {"name", "path", "rank", "sloc", "priority", "band"}
    empty = tree_json(build_tree(dataclasses.replace(linked_result, files=())), slocs={})
    assert empty["priority"] is None and empty["band"] is None and empty["top_file"] is None


def test_tdp_tree(saved_linked: LoadedAnalysis, linked_repo: Path) -> None:
    app = build_app()
    table = runner.invoke(app, ["tree", str(linked_repo), "--dev-model", "--files"])
    assert table.exit_code == 0, table.stderr
    assert "routes.py" in ANSI.sub("", table.stdout)
    data = runner.invoke(
        app, ["tree", str(linked_repo), "--dev-model", "-f", "json", "--path", "app", "--depth", "1"]
    )
    assert data.exit_code == 0, data.stderr
    doc = json.loads(data.stdout)
    assert doc["schema"] == "entropiq.tree.v1" and doc["repo"] == "linked"
    assert doc["root"]["path"] == "app/" and doc["root"]["children"][0]["collapsed"] is True
    missing = runner.invoke(app, ["tree", str(linked_repo), "--dev-model", "--path", "lib"])
    assert missing.exit_code == 2 and "No ranked files under lib/" in missing.stderr

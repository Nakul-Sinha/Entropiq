from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tdp_cli.commands._views import find_file, ranked_by_path
from tdp_cli.commands.graph import (
    MAX_CHILDREN,
    ImportGraph,
    central_files,
    complete_graph,
    dependency_tree,
    file_graph_json,
    graph_json,
    graph_overview,
    hidden_couplings,
    import_cycles,
    import_graph,
    reachable,
)
from tdp_cli.main import build_app
from tdp_cli.session import Session
from tdp_cli.shell import dispatch
from tdp_cli.store import LoadedAnalysis
from tdp_core.contracts import AnalysisResult, GraphData, GraphEdge, GraphNode

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _out(session: Session) -> str:
    text = session.console.file.getvalue()  # type: ignore[attr-defined]
    session.console.file.seek(0)  # type: ignore[attr-defined]
    session.console.file.truncate()  # type: ignore[attr-defined]
    return text


def _render(session: Session, renderable: object) -> str:
    session.console.print(renderable)
    return _out(session)


def test_import_graph_and_reachable(linked_result: AnalysisResult) -> None:
    graph = import_graph(linked_result)
    assert graph.n_edges == 4
    assert set(graph.imports) == set(graph.importers) == {f.path for f in linked_result.files}
    assert graph.imports["app/api/routes.py"] == ("app/views.py",)
    assert graph.imports["app/core.py"] == ("app/util.py",)
    rank = {path: item.rank for path, item in ranked_by_path(linked_result).items()}
    importers = graph.importers["app/core.py"]
    assert set(importers) == {"app/util.py", "app/views.py"}
    assert list(importers) == sorted(importers, key=rank.__getitem__)
    assert graph.importers["setup.py"] == ()
    assert reachable("app/core.py", graph.importers) == ["app/api/routes.py", "app/util.py", "app/views.py"]
    assert reachable("app/core.py", graph.imports) == ["app/util.py"]
    assert reachable("setup.py", graph.imports) == []
    assert reachable("nowhere.py", {}) == []


def test_import_graph_drops_odd_edges(linked_result: AnalysisResult) -> None:
    edges = (
        *linked_result.graph.edges,
        GraphEdge("app/core.py", "app/core.py", "import", 1.0),
        GraphEdge("app/core.py", "app/util.py", "import", 1.0),
        GraphEdge("app/core.py", "tests/test_core.py", "import", 1.0),
    )
    odd = dataclasses.replace(linked_result, graph=GraphData(linked_result.graph.nodes, edges))
    assert import_graph(odd) == import_graph(linked_result)


def test_cycles_central_and_couplings(linked_result: AnalysisResult) -> None:
    graph = import_graph(linked_result)
    assert import_cycles(graph) == [("app/core.py", "app/util.py")]
    central = central_files(linked_result, graph)
    assert [c.item.path for c in central][:2] in (
        ["app/core.py", "app/util.py"],
        ["app/util.py", "app/core.py"],
    )
    assert [c.item.path for c in central][2] == "app/views.py"
    core = next(c for c in central if c.item.path == "app/core.py")
    assert (core.dependents, core.importers, core.imports) == (3, 2, 1)
    assert core.pagerank == pytest.approx(float(linked_result.features.at["app/core.py", "pagerank"]))
    couplings = hidden_couplings(linked_result)
    assert [(e.source, e.target, e.weight) for e in couplings] == [
        ("app/api/routes.py", "app/api/schema.py", 1.0)
    ]


def test_central_files_without_nodes_or_features(linked_result: AnalysisResult) -> None:
    bare = dataclasses.replace(
        linked_result,
        graph=GraphData((), linked_result.graph.edges),
        features=linked_result.features.drop(columns=["pagerank"]),
    )
    central = central_files(bare, import_graph(bare))
    core = next(c for c in central if c.item.path == "app/core.py")
    assert core.dependents == 3 and core.pagerank == 0.0
    big = ImportGraph(
        imports={"a": ("b",), "b": ("a",), "c": ("d",), "d": ("e",), "e": ("c",), "f": ()},
        importers={},
        n_edges=5,
    )
    assert import_cycles(big) == [("c", "d", "e"), ("a", "b")]


def test_graph_overview(linked_session: Session) -> None:
    dispatch(linked_session, "/graph")
    out = _out(linked_session)
    assert "9 ranked files, 4 import edges, 1 import cycle, 1 hidden coupling" in out
    for heading in ("Most central files", "Import cycles", "Hidden couplings"):
        assert heading in out
    assert "Dependents" in out and "PageRank" in out
    assert "app/core.py, app/util.py" in out
    assert "app/api/routes.py" in out and "app/api/schema.py" in out and "100%" in out
    assert "Dependents counts every ranked file" in out
    assert "Confidence is the share" in out
    assert "Showing" not in out
    dispatch(linked_session, "/graph --top 1")
    out = _out(linked_session)
    assert "Showing 1 of 3; add --top 0 to see all." in out
    dispatch(linked_session, "/graph --top 0")
    assert "Showing" not in _out(linked_session)


def test_graph_overview_of_an_empty_graph(linked_session: Session, linked_loaded: LoadedAnalysis) -> None:
    nodes = tuple(GraphNode(n.id, n.priority, n.band, n.sloc, 0) for n in linked_loaded.result.graph.nodes)
    result = dataclasses.replace(linked_loaded.result, graph=GraphData(nodes, ()))
    out = _render(linked_session, graph_overview(dataclasses.replace(linked_loaded, result=result)))
    assert "9 ranked files, 0 import edges, 0 import cycles, 0 hidden couplings" in out
    assert "No ranked file imports another ranked file." in out
    assert "No import cycles." in out
    assert "No hidden couplings found." in out
    one = dataclasses.replace(result, files=result.files[:1])
    assert "1 ranked file, 0 import edges" in _render(
        linked_session, graph_overview(dataclasses.replace(linked_loaded, result=one))
    )
    empty = dataclasses.replace(result, files=())
    linked_session.current = dataclasses.replace(linked_loaded, result=empty)
    dispatch(linked_session, "/graph")
    assert "This analysis has no ranked files." in _out(linked_session)


def test_dependency_tree(linked_session: Session) -> None:
    dispatch(linked_session, "/graph app/core.py")
    out = _out(linked_session)
    assert "Imported by: 2 directly, 3 in total" in out
    assert "Imports: 1 directly, 1 in total" in out
    assert "app/api/routes.py" in out
    assert "(import cycle)" in out
    assert "Tests that import it: 1" in out and "tests/test_core.py" in out
    assert "Changes together with" not in out
    assert "In an import cycle with: app/util.py" in out
    dispatch(linked_session, "/graph app/core.py --depth 1")
    out = _out(linked_session)
    assert "app/api/routes.py" not in out
    assert "(+1 more)" in out
    dispatch(linked_session, "/graph views.py")
    out = _out(linked_session)
    assert "Imported by: 1 directly, 1 in total" in out
    assert "Imports: 1 directly, 2 in total" in out
    assert "In an import cycle" not in out
    dispatch(linked_session, "/graph routes.py")
    out = _out(linked_session)
    assert "Changes together with: 1" in out and "app/api/schema.py" in out and "100%" in out
    dispatch(linked_session, "/graph schema.py")
    assert "Changes together with: 1" in _out(linked_session)


def test_dependency_tree_limits(linked_session: Session, linked_loaded: LoadedAnalysis) -> None:
    result = linked_loaded.result
    hub = result.files[0]
    many = tuple(dataclasses.replace(hub, path=f"lib/m{i:02d}.py") for i in range(MAX_CHILDREN + 2))
    edges = tuple(GraphEdge(f.path, "app/core.py", "import", 1.0) for f in many)
    edges += (
        GraphEdge("lib/m00.py", "lib/m01.py", "import", 1.0),
        GraphEdge("lib/m01.py", "lib/m02.py", "import", 1.0),
    )
    wide = dataclasses.replace(
        result, files=result.files + many, graph=GraphData(result.graph.nodes, result.graph.edges + edges)
    )
    loaded = dataclasses.replace(linked_loaded, result=wide)
    item = find_file(wide, "app/core.py")
    out = _render(linked_session, dependency_tree(loaded, item, depth=3))
    assert f"Imported by: {MAX_CHILDREN + 4} directly" in out
    assert "and 4 more" in out
    assert "(shown above)" in out
    bare = dataclasses.replace(item, extra=None)
    assert "Tests that import it: 0" in _render(linked_session, dependency_tree(linked_loaded, bare))


def test_graph_errors_and_completion(linked_session: Session) -> None:
    dispatch(linked_session, "/graph nowhere.py")
    assert "No ranked file matches nowhere.py" in _out(linked_session)
    dispatch(linked_session, "/graph app/core.py --depth 0")
    out = _out(linked_session)
    assert "--depth must be between 1 and 6" in out
    assert "Usage: /graph [PATH or RANK] [--depth N] [--top N]" in out
    dispatch(linked_session, "/graph a b")
    assert "Usage: /graph" in _out(linked_session)

    def texts(args: list[str], word: str) -> list[str]:
        return [s.text for s in complete_graph(linked_session, args, word)]

    assert texts([], "--") == ["--depth", "--top"]
    assert texts(["--depth"], "") == []
    assert texts([], "app/c") == ["app/core.py"]
    assert texts(["--top", "3"], "app/u") == ["app/util.py"]
    assert texts(["app/core.py"], "app") == []


def test_graph_json(linked_loaded: LoadedAnalysis) -> None:
    doc = graph_json(linked_loaded)
    assert doc["schema"] == "entropiq.graph.v1"
    assert doc["summary"] == {"files": 9, "import_edges": 4, "cycles": 1, "hidden_couplings": 1}
    assert [c["path"] for c in doc["central"]][2] == "app/views.py"
    assert doc["cycles"] == [{"size": 2, "files": ["app/core.py", "app/util.py"]}]
    assert doc["hidden_couplings"] == [
        {"source": "app/api/routes.py", "target": "app/api/schema.py", "confidence": 1.0}
    ]
    assert len(graph_json(linked_loaded, top=1)["central"]) == 1
    item = find_file(linked_loaded.result, "app/core.py")
    one = file_graph_json(linked_loaded, item)
    assert one["schema"] == "entropiq.graph.file.v1"
    assert one["file"]["path"] == "app/core.py"
    assert set(one["dependents"]) == {"app/util.py", "app/views.py"}
    assert one["dependencies"] == ["app/util.py"]
    assert one["all_dependents"] == ["app/api/routes.py", "app/util.py", "app/views.py"]
    assert one["test_importers"] == ["tests/test_core.py"]
    assert one["cycle"] == ["app/core.py", "app/util.py"]
    assert one["hidden_couplings"] == []
    routes = file_graph_json(linked_loaded, find_file(linked_loaded.result, "routes.py"))
    assert routes["hidden_couplings"] == [{"path": "app/api/schema.py", "confidence": 1.0}]
    assert routes["cycle"] == []
    bare = file_graph_json(linked_loaded, dataclasses.replace(item, extra=None))
    assert bare["test_importers"] == []


def test_tdp_graph(saved_linked: LoadedAnalysis, linked_repo: Path) -> None:
    app = build_app()
    table = runner.invoke(app, ["graph", str(linked_repo), "--dev-model"])
    assert table.exit_code == 0, table.stderr
    assert "Most central files" in ANSI.sub("", table.stdout)
    data = runner.invoke(app, ["graph", str(linked_repo), "--dev-model", "--format", "json", "--top", "0"])
    assert data.exit_code == 0, data.stderr
    assert json.loads(data.stdout)["summary"]["cycles"] == 1
    tree = runner.invoke(app, ["graph", str(linked_repo), "--dev-model", "--file", "core.py", "--depth", "1"])
    assert tree.exit_code == 0 and "Imported by: 2 directly, 3 in total" in ANSI.sub("", tree.stdout)
    one = runner.invoke(app, ["graph", str(linked_repo), "--dev-model", "--file", "1", "-f", "json"])
    assert one.exit_code == 0 and json.loads(one.stdout)["schema"] == "entropiq.graph.file.v1"
    missing = runner.invoke(app, ["graph", str(linked_repo), "--dev-model", "--file", "nowhere.py"])
    assert missing.exit_code == 2 and "No ranked file matches" in missing.stderr
    deep = runner.invoke(app, ["graph", str(linked_repo), "--dev-model", "--depth", "9"])
    assert deep.exit_code == 2

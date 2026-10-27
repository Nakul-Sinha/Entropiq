from __future__ import annotations

import random
import sys
import time
from collections.abc import Collection, Iterable
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from tdp_core.config import DEFAULT_SETTINGS, with_overrides
from tdp_core.contracts import FEATURE_FAMILIES, FileSyntax, RawImport
from tdp_core.graph.centrality import GRAPH_FEATURE_COLUMNS, graph_features
from tdp_core.graph.imports import build_import_graph


def make_graph(
    edges: Iterable[tuple[str, str]], tests: Collection[str] = (), extra: Iterable[str] = ()
) -> nx.DiGraph[str]:
    edges = list(edges)
    graph: nx.DiGraph[str] = nx.DiGraph()
    for node in sorted({n for edge in edges for n in edge} | set(extra)):
        graph.add_node(node, is_test=node in tests)
    graph.add_edges_from(edges)
    return graph


def traced() -> bool:
    """True when coverage or a debugger slows the interpreter down."""
    return sys.gettrace() is not None or sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID) is not None


def test_columns_match_the_contract() -> None:
    impact = set(FEATURE_FAMILIES["I"])
    assert set(GRAPH_FEATURE_COLUMNS) == (
        impact - {"cochange_partners", "cochange_max_conf", "hidden_coupling"}
    ) | {"test_importers"}
    assert "test_importers" in FEATURE_FAMILIES["T"]


def test_frame_shape_and_dtypes() -> None:
    feats = graph_features(make_graph([("b.py", "a.py"), ("c.py", "a.py")]))
    assert tuple(feats.columns) == GRAPH_FEATURE_COLUMNS
    assert list(feats.index) == ["a.py", "b.py", "c.py"]
    assert feats.index.name == "path"
    for col in GRAPH_FEATURE_COLUMNS:
        expected = np.float64 if col in ("pagerank", "betweenness") else np.int64
        assert feats[col].dtype == expected, col


def test_chain_transitive_dependents_and_degrees() -> None:
    feats = graph_features(make_graph([("a.py", "b.py"), ("b.py", "c.py")]))
    assert feats["transitive_dependents"].to_dict() == {"a.py": 0, "b.py": 1, "c.py": 2}
    assert feats["fan_in"].to_dict() == {"a.py": 0, "b.py": 1, "c.py": 1}
    assert feats["fan_out"].to_dict() == {"a.py": 1, "b.py": 1, "c.py": 0}
    assert feats["scc_size"].to_dict() == {"a.py": 1, "b.py": 1, "c.py": 1}


def test_cycle_scc_size_and_dependents() -> None:
    feats = graph_features(
        make_graph([("a.py", "b.py"), ("b.py", "a.py"), ("c.py", "a.py"), ("a.py", "d.py")])
    )
    assert feats["scc_size"].to_dict() == {"a.py": 2, "b.py": 2, "c.py": 1, "d.py": 1}
    # a is reached by b (same cycle) and c; b by a and c; d by a, b and c
    assert feats["transitive_dependents"].to_dict() == {"a.py": 2, "b.py": 2, "c.py": 0, "d.py": 3}


def test_tests_excluded_from_fan_in_but_counted_in_test_importers() -> None:
    graph = make_graph(
        [
            ("a.py", "b.py"),
            ("tests/test_b.py", "b.py"),
            ("tests/test_ab.py", "b.py"),
            ("tests/test_ab.py", "a.py"),
        ],
        tests={"tests/test_b.py", "tests/test_ab.py"},
    )
    feats = graph_features(graph)
    assert list(feats.index) == ["a.py", "b.py"]
    assert feats.loc["b.py", "fan_in"] == 1
    assert feats.loc["b.py", "transitive_dependents"] == 1
    assert feats["test_importers"].to_dict() == {"a.py": 1, "b.py": 2}


def test_tests_ignored_by_centrality() -> None:
    with_tests = make_graph(
        [("a.py", "b.py"), ("b.py", "c.py"), ("t/test_x.py", "c.py"), ("c.py", "t/test_x.py")],
        tests={"t/test_x.py"},
    )
    without = make_graph([("a.py", "b.py"), ("b.py", "c.py")])
    cols = [c for c in GRAPH_FEATURE_COLUMNS if c != "test_importers"]
    pd.testing.assert_frame_equal(graph_features(with_tests)[cols], graph_features(without)[cols])


def test_pagerank_flows_to_imported_modules() -> None:
    feats = graph_features(
        make_graph([("a.py", "c.py"), ("b.py", "c.py"), ("c.py", "d.py")], extra=("e.py",))
    )
    pr = feats["pagerank"]
    assert pr.sum() == pytest.approx(1.0)
    assert pr["d.py"] > pr["c.py"] > pr["a.py"]
    assert pr["a.py"] == pytest.approx(pr["e.py"])


def test_betweenness_matches_networkx() -> None:
    graph = make_graph([("a.py", "b.py"), ("b.py", "c.py"), ("x.py", "b.py")])
    feats = graph_features(graph)
    assert feats["betweenness"]["b.py"] > 0
    assert feats["betweenness"]["a.py"] == 0
    expected = nx.betweenness_centrality(graph, normalized=True)
    assert feats["betweenness"].to_dict() == pytest.approx(expected)


def test_isolated_node_and_single_node() -> None:
    feats = graph_features(make_graph([], extra=("only.py",)))
    assert feats.loc["only.py"].to_dict() == {
        "fan_in": 0,
        "fan_out": 0,
        "transitive_dependents": 0,
        "pagerank": 1.0,
        "betweenness": 0.0,
        "scc_size": 1,
        "test_importers": 0,
    }


@pytest.mark.parametrize("tests", [set(), {"tests/test_a.py"}])
def test_empty_frames(tests: set[str]) -> None:
    graph = make_graph([], tests=tests, extra=tuple(tests))
    feats = graph_features(graph)
    assert feats.empty
    assert tuple(feats.columns) == GRAPH_FEATURE_COLUMNS
    assert feats.index.name == "path"
    assert feats["fan_in"].dtype == np.int64
    assert feats["pagerank"].dtype == np.float64


def test_missing_is_test_attribute_means_non_test() -> None:
    graph: nx.DiGraph[str] = nx.DiGraph([("a.py", "b.py")])
    assert list(graph_features(graph).index) == ["a.py", "b.py"]


@pytest.mark.parametrize("seed", range(5))
def test_transitive_dependents_match_brute_force(seed: int) -> None:
    rng = random.Random(seed)
    n = 60
    nodes = [f"m{i:02d}.py" for i in range(n)]
    tests = set(rng.sample(nodes, 8))
    edges = [(rng.choice(nodes), rng.choice(nodes)) for _ in range(90)]
    graph = make_graph([(u, v) for u, v in edges if u != v], tests=tests, extra=tuple(nodes))
    feats = graph_features(graph)

    sub = graph.subgraph(node for node in graph if node not in tests)
    for node in sub:
        assert feats.loc[node, "transitive_dependents"] == len(nx.ancestors(sub, node)), node
        scc = next(c for c in nx.strongly_connected_components(sub) if node in c)
        assert feats.loc[node, "scc_size"] == len(scc)
        assert feats.loc[node, "fan_in"] == sub.in_degree(node)
        assert feats.loc[node, "test_importers"] == sum(1 for p in graph.predecessors(node) if p in tests)


def test_betweenness_sampling_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    real = nx.betweenness_centrality

    def spy(G: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(G, **kwargs)

    monkeypatch.setattr(nx, "betweenness_centrality", spy)
    settings = with_overrides(DEFAULT_SETTINGS, seed=7)

    graph_features(make_graph([], extra=[f"f{i}.py" for i in range(2000)]), settings)
    graph_features(make_graph([], extra=[f"f{i}.py" for i in range(2001)]), settings)

    assert calls[0] == {"k": None, "normalized": True, "seed": 7}
    assert calls[1] == {"k": 500, "normalized": True, "seed": 7}


def test_sampled_betweenness_is_deterministic() -> None:
    rng = random.Random(3)
    nodes = [f"pkg/m{i:04d}.py" for i in range(2100)]
    edges = [(rng.choice(nodes), rng.choice(nodes)) for _ in range(3000)]
    graph = make_graph([(u, v) for u, v in edges if u != v], extra=tuple(nodes))
    first = graph_features(graph)
    second = graph_features(graph.copy())
    pd.testing.assert_frame_equal(first, second)
    other_seed = graph_features(graph, with_overrides(DEFAULT_SETTINGS, seed=1))
    assert not np.allclose(first["betweenness"], other_seed["betweenness"])


def test_end_to_end_from_file_syntax() -> None:
    def syntax(path: str, *imports: RawImport) -> FileSyntax:
        return FileSyntax(path, "python", 1, 1, (), 0, (), imports, (), (), False)

    files = [
        syntax("pkg/__init__.py", RawImport("core", ("Thing",), 1, 1)),
        syntax("pkg/core.py", RawImport("", ("util",), 1, 1)),
        syntax("pkg/util.py", RawImport("pkg.core", (), 0, 1)),
        syntax("app.py", RawImport("pkg", ("Thing",), 0, 1), RawImport("os", (), 0, 2)),
        syntax("tests/test_util.py", RawImport("pkg.util", ("f",), 0, 1)),
    ]
    graph = build_import_graph({f.path: f for f in files}, lambda p: p.startswith("tests/"))
    feats = graph_features(graph)
    assert list(feats.index) == ["app.py", "pkg/__init__.py", "pkg/core.py", "pkg/util.py"]
    assert feats.loc["pkg/core.py", "scc_size"] == 2
    assert feats.loc["pkg/util.py", "scc_size"] == 2
    assert feats.loc["pkg/util.py", "transitive_dependents"] == 3
    assert feats.loc["pkg/util.py", "test_importers"] == 1
    assert feats.loc["app.py", "transitive_dependents"] == 0


@pytest.mark.parametrize("shape", ["random", "dag"])
def test_performance_5000_nodes(shape: str) -> None:
    rng = random.Random(42)
    n, m = 5000, 15000
    nodes = [f"pkg{i % 50}/mod{i:04d}.py" for i in range(n)]
    tests = set(rng.sample(nodes, 500))
    edges: set[tuple[str, str]] = set()
    while len(edges) < m:
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        if shape == "dag" and i < j:
            i, j = j, i  # newer modules import older ones: no cycles
        edges.add((nodes[i], nodes[j]))
    graph = make_graph(sorted(edges), tests=tests, extra=tuple(nodes))

    start = time.perf_counter()
    feats = graph_features(graph)
    elapsed = time.perf_counter() - start

    assert len(feats) == n - len(tests)
    assert (feats["transitive_dependents"] >= feats["fan_in"]).all()
    if traced():
        pytest.skip(f"timing not checked under coverage or a debugger ({elapsed:.1f}s)")
    assert elapsed < 20.0, f"{elapsed:.1f}s"

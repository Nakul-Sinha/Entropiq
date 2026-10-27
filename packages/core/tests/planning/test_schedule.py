from __future__ import annotations

import math
import random
from collections.abc import Iterable

import networkx as nx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdp_core.planning.schedule import REASON_BEST_ROI, schedule


def graph(edges: Iterable[tuple[str, str]], nodes: Iterable[str] = ()) -> nx.DiGraph[str]:
    g: nx.DiGraph[str] = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def test_dependencies_come_first() -> None:
    # a imports b, b imports c: c is fixed first
    g = graph([("a.py", "b.py"), ("b.py", "c.py")])
    roi = {"a.py": 1.0, "b.py": 1.0, "c.py": 1.0}
    assert schedule(["a.py", "b.py", "c.py"], g, roi) == [
        ("c.py", 1, "fix before b.py, which depends on it"),
        ("b.py", 2, "fix before a.py, which depends on it"),
        ("a.py", 3, REASON_BEST_ROI),
    ]


def test_dependency_order_beats_roi() -> None:
    g = graph([("top.py", "base.py")])
    out = schedule(["top.py", "base.py"], g, {"top.py": 9.0, "base.py": 0.1})
    assert [p for p, _, _ in out] == ["base.py", "top.py"]


def test_cycle_members_share_a_group() -> None:
    g = graph([("x.py", "y.py"), ("y.py", "x.py"), ("z.py", "x.py")])
    out = schedule(["z.py", "x.py", "y.py"], g, {"x.py": 1.0, "y.py": 2.0, "z.py": 3.0})
    assert out == [
        ("y.py", 1, "fix together with x.py (dependency cycle)"),
        ("x.py", 1, "fix together with y.py (dependency cycle)"),
        ("z.py", 2, REASON_BEST_ROI),
    ]


def test_independent_files_by_roi_then_path() -> None:
    roi = {"a.py": 1.0, "b.py": 3.0, "c.py": 3.0, "d.py": 2.0}
    out = schedule(list(roi), graph([]), roi)
    assert out == [
        ("b.py", 1, REASON_BEST_ROI),
        ("c.py", 2, REASON_BEST_ROI),
        ("d.py", 3, REASON_BEST_ROI),
        ("a.py", 4, REASON_BEST_ROI),
    ]


def test_cycle_with_a_strong_member_goes_before_a_medium_file() -> None:
    g = graph([("p.py", "q.py"), ("q.py", "p.py")])
    roi = {"p.py": 5.0, "q.py": 0.1, "s.py": 3.0, "t.py": 0.5}
    out = schedule(["t.py", "s.py", "q.py", "p.py"], g, roi)
    assert [(p, grp) for p, grp, _ in out] == [("p.py", 1), ("q.py", 1), ("s.py", 2), ("t.py", 3)]


def test_named_dependent_is_the_first_in_the_plan() -> None:
    g = graph([("u1.py", "base.py"), ("u2.py", "base.py")])
    out = schedule(["u1.py", "u2.py", "base.py"], g, {"base.py": 0.5, "u1.py": 1.0, "u2.py": 5.0})
    assert out == [
        ("base.py", 1, "fix before u2.py, which depends on it"),
        ("u2.py", 2, REASON_BEST_ROI),
        ("u1.py", 3, REASON_BEST_ROI),
    ]


def test_only_direct_edges_between_chosen_files_count() -> None:
    # a reaches b only through m, which is not in the plan; a also imports itself; ghost is not in the graph
    g = graph([("a.py", "m.py"), ("m.py", "b.py"), ("a.py", "a.py")])
    out = schedule(["b.py", "ghost.py", "a.py"], g, {"a.py": 3.0, "b.py": 1.0, "ghost.py": 2.0})
    assert out == [
        ("a.py", 1, REASON_BEST_ROI),
        ("ghost.py", 2, REASON_BEST_ROI),
        ("b.py", 3, REASON_BEST_ROI),
    ]


@pytest.mark.parametrize(
    ("size", "first_reason"),
    [
        (3, "fix together with c2.py and c3.py (dependency cycle)"),
        (4, "fix together with c2.py, c3.py and c4.py (dependency cycle)"),
        (5, "fix together with c2.py, c3.py, c4.py and 1 more (dependency cycle)"),
        (7, "fix together with c2.py, c3.py, c4.py and 3 more (dependency cycle)"),
    ],
)
def test_cycle_reasons_list_at_most_three_paths(size: int, first_reason: str) -> None:
    nodes = [f"c{i}.py" for i in range(1, size + 1)]
    g = graph(zip(nodes, nodes[1:] + nodes[:1], strict=True))
    roi = {p: float(size - i) for i, p in enumerate(nodes)}
    out = schedule(nodes[::-1], g, roi)
    assert [p for p, _, _ in out] == nodes
    assert {grp for _, grp, _ in out} == {1}
    assert out[0][2] == first_reason
    if size == 5:
        assert out[2][2] == "fix together with c1.py, c2.py, c4.py and 1 more (dependency cycle)"


def test_empty_input() -> None:
    assert schedule([], graph([("a.py", "b.py")]), {}) == []


def test_input_order_does_not_matter() -> None:
    g = graph([("a.py", "b.py"), ("b.py", "a.py"), ("c.py", "a.py"), ("d.py", "c.py"), ("e.py", "f.py")])
    roi = {"a.py": 1.0, "b.py": 2.0, "c.py": 2.0, "d.py": 0.5, "e.py": 2.0, "f.py": 0.1}
    expected = schedule(sorted(roi), g, roi)
    paths = list(roi)
    rng = random.Random(3)
    for _ in range(10):
        rng.shuffle(paths)
        assert schedule(paths, g, roi) == expected


def test_invalid_input() -> None:
    g = graph([])
    with pytest.raises(ValueError, match=r"a\.py"):
        schedule(["a.py", "b.py", "a.py"], g, {"a.py": 1.0, "b.py": 1.0})
    with pytest.raises(KeyError):
        schedule(["a.py", "b.py"], g, {"a.py": 1.0})
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="roi"):
            schedule(["a.py"], g, {"a.py": bad})


@settings(max_examples=300, deadline=None, derandomize=True)
@given(
    data=st.integers(1, 9).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)), max_size=25),
            st.lists(st.floats(-5, 5, allow_nan=False, allow_subnormal=False), min_size=n, max_size=n),
            st.lists(st.booleans(), min_size=n, max_size=n),
        )
    )
)
def test_schedule_properties(data: tuple[int, list[tuple[int, int]], list[float], list[bool]]) -> None:
    n, raw_edges, values, picked = data
    paths = [f"m{i}.py" for i in range(n)]
    g = graph(((paths[u], paths[v]) for u, v in raw_edges), nodes=paths)
    chosen = [p for p, keep in zip(paths, picked, strict=True) if keep]
    roi = dict(zip(paths, values, strict=True))
    out = schedule(chosen, g, roi)

    assert sorted(p for p, _, _ in out) == sorted(chosen)
    position = {p: i for i, (p, _, _) in enumerate(out)}
    group = {p: grp for p, grp, _ in out}
    numbers = [grp for _, grp, _ in out]
    assert numbers == sorted(numbers)
    assert set(numbers) == set(range(1, len(set(numbers)) + 1))

    sub = g.subgraph(chosen)
    for component in nx.strongly_connected_components(sub):
        assert len({group[p] for p in component}) == 1
        spots = sorted(position[p] for p in component)
        assert spots == list(range(spots[0], spots[0] + len(spots)))
        if len(component) > 1:
            assert all("dependency cycle" in out[position[p]][2] for p in component)
    assert len(set(numbers)) == nx.number_strongly_connected_components(sub)
    for u, v in sub.edges():
        if group[u] != group[v]:
            assert position[v] < position[u]
            if numbers.count(group[v]) == 1:
                assert out[position[v]][2].startswith("fix before ")

    assert schedule(chosen[::-1], g, roi) == out

from __future__ import annotations

import math
import random
import sys
import time
from collections.abc import Collection, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from tdp_core.config import DEFAULT_SETTINGS, Settings, with_overrides
from tdp_core.contracts import COMMITS_SCHEMA, FEATURE_FAMILIES, FILE_CHANGES_SCHEMA, Snapshot
from tdp_core.graph.centrality import GRAPH_FEATURE_COLUMNS
from tdp_core.graph.cochange import (
    COCHANGE_FEATURE_COLUMNS,
    CochangeEdge,
    cochange_edges,
    cochange_features,
)
from tdp_core.graph.imports import build_import_graph
from tdp_core.mining import list_files, mine_history, path_lineage, read_blobs, run_git, snapshot_at
from tdp_core.mining.gitlog import MinedHistory
from tdp_core.parsing.python import is_test_file, parse_python
from tdp_core.testing import RepoBuilder

T0 = datetime(2022, 1, 3, 12, tzinfo=UTC)
SECONDS_PER_DAY = 86400

type Change = str | tuple[str, str]  # a path, or (old_path, new_path) for a rename
type Commit = tuple[float, Sequence[Change]]  # (days after T0, changes)
type Row = tuple[int, float, int, float]  # (partners, max_conf, hidden, rate)


# ------------------------------------------------------------------------------------------------ helpers
def sha(k: int) -> str:
    return f"{k:040x}"


def history(commits: Sequence[Commit], *, bulk: Collection[int] = ()) -> MinedHistory:
    """Commit ``k`` gets sha ``sha(k)`` and author time ``T0 + day`` (rounded to whole seconds)."""
    base = int(T0.timestamp())
    c: dict[str, list[object]] = {name: [] for name in COMMITS_SCHEMA.names}
    fc: dict[str, list[object]] = {name: [] for name in FILE_CHANGES_SCHEMA.names}
    for k, (day, changes) in enumerate(commits):
        c["sha"].append(sha(k))
        c["author_name"].append("Ana")
        c["author_email"].append("ana@example.com")
        c["author_time"].append(datetime.fromtimestamp(base + round(day * SECONDS_PER_DAY), UTC))
        c["parents"].append([sha(k - 1)] if k else [])
        c["subject"].append(f"commit {k}")
        c["message"].append(f"commit {k}")
        c["files_touched"].append(len(changes))
        c["is_bulk"].append(k in bulk)
        for change in changes:
            old, new = change if isinstance(change, tuple) else (None, change)
            fc["sha"].append(sha(k))
            fc["path"].append(new)
            fc["old_path"].append(old)
            fc["added"].append(1)
            fc["deleted"].append(0)
            fc["is_binary"].append(False)
    return MinedHistory(
        commits=pa.Table.from_pydict(c, schema=COMMITS_SCHEMA),
        file_changes=pa.Table.from_pydict(fc, schema=FILE_CHANGES_SCHEMA),
    )


def snap(day: float = 100) -> Snapshot:
    return Snapshot("demo", 0, T0 + timedelta(days=day), "0" * 40)


def together(n: int, *paths: str, day: float = 1.0) -> list[Commit]:
    """``n`` commits (one per day from ``day``) that each touch all ``paths``."""
    return [(day + k, paths) for k in range(n)]


def alone(n: int, path: str, day: float = 1.0) -> list[Commit]:
    return together(n, path, day=day)


def graph(edges: Iterable[tuple[str, str]], nodes: Iterable[str] = ()) -> nx.DiGraph[str]:
    edges = list(edges)
    g: nx.DiGraph[str] = nx.DiGraph()
    for node in sorted({n for edge in edges for n in edge} | set(nodes)):
        g.add_node(node, is_test=is_test_file(node))
    g.add_edges_from(edges)
    return g


def feats(
    hist: MinedHistory,
    paths: Sequence[str],
    *,
    lineage: Mapping[str, frozenset[str]] | None = None,
    import_graph: nx.DiGraph[str] | None = None,
    settings: Settings = DEFAULT_SETTINGS,
    reachable: AbstractSet[str] | None = None,
    snapshot: Snapshot | None = None,
) -> pd.DataFrame:
    return cochange_features(
        hist,
        snapshot or snap(),
        lineage or {},
        paths,
        is_test_file,
        import_graph,
        settings,
        reachable=reachable,
    )


def edges(
    hist: MinedHistory,
    paths: Sequence[str],
    *,
    lineage: Mapping[str, frozenset[str]] | None = None,
    import_graph: nx.DiGraph[str] | None = None,
    settings: Settings = DEFAULT_SETTINGS,
    reachable: AbstractSet[str] | None = None,
    snapshot: Snapshot | None = None,
) -> list[CochangeEdge]:
    return cochange_edges(
        hist,
        snapshot or snap(),
        lineage or {},
        paths,
        is_test_file,
        import_graph,
        settings,
        reachable=reachable,
    )


def rows(frame: pd.DataFrame) -> dict[str, Row]:
    return {
        str(path): (
            int(row["cochange_partners"]),
            float(row["cochange_max_conf"]),
            int(row["hidden_coupling"]),
            float(row["test_cochange_rate"]),
        )
        for path, row in frame.iterrows()
    }


def traced() -> bool:
    """True when coverage or a debugger slows the interpreter down."""
    return sys.gettrace() is not None or sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID) is not None


def check_frame(frame: pd.DataFrame) -> None:
    assert tuple(frame.columns) == COCHANGE_FEATURE_COLUMNS
    assert frame.index.name == "path"
    assert pd.api.types.is_string_dtype(frame.index.dtype)
    assert list(frame.index) == sorted(frame.index)
    for col in COCHANGE_FEATURE_COLUMNS:
        expected = np.float64 if col in ("cochange_max_conf", "test_cochange_rate") else np.int64
        assert frame[col].dtype == expected, col


# ---------------------------------------------------------------------------------------------- contract
def test_columns_match_the_contract() -> None:
    assert COCHANGE_FEATURE_COLUMNS[:3] == ("cochange_partners", "cochange_max_conf", "hidden_coupling")
    assert set(COCHANGE_FEATURE_COLUMNS[:3]) <= set(FEATURE_FAMILIES["I"])
    assert "test_cochange_rate" in FEATURE_FAMILIES["T"]
    assert set(GRAPH_FEATURE_COLUMNS) | set(COCHANGE_FEATURE_COLUMNS[:3]) == set(FEATURE_FAMILIES["I"]) | {
        "test_importers"
    }


def test_frame_shape_and_dtypes() -> None:
    frame = feats(history(together(5, "b.py", "a.py")), ["b.py", "a.py", "c.py"])
    check_frame(frame)
    assert list(frame.index) == ["a.py", "b.py", "c.py"]


@pytest.mark.parametrize("paths", [[], ["tests/test_a.py", "conftest.py"]])
def test_no_ranked_files_gives_an_empty_frame(paths: list[str]) -> None:
    hist = history(together(5, "a.py", "tests/test_a.py"))
    frame = feats(hist, paths)
    check_frame(frame)
    assert frame.empty
    assert edges(hist, paths) == []


def test_files_without_history_get_zeros() -> None:
    hist = history(together(5, "a.py", "b.py"))
    frame = feats(hist, ["x.py", "y.py"])
    check_frame(frame)
    assert rows(frame) == {"x.py": (0, 0.0, 0, 0.0), "y.py": (0, 0.0, 0, 0.0)}
    assert edges(hist, ["x.py", "y.py"]) == []


def test_empty_history() -> None:
    hist = history([])
    assert rows(feats(hist, ["a.py"])) == {"a.py": (0, 0.0, 0, 0.0)}
    assert edges(hist, ["a.py"]) == []


# ------------------------------------------------------------------------------------------------- rules
def test_two_files_always_changed_together() -> None:
    hist = history(together(5, "a.py", "b.py"))
    assert rows(feats(hist, ["a.py", "b.py"])) == {"a.py": (1, 1.0, 1, 0.0), "b.py": (1, 1.0, 1, 0.0)}
    assert edges(hist, ["a.py", "b.py"]) == [("a.py", "b.py", 5, 1.0), ("b.py", "a.py", 5, 1.0)]


def test_support_threshold_is_inclusive() -> None:
    four = history(together(4, "a.py", "b.py"))
    assert rows(feats(four, ["a.py", "b.py"]))["a.py"] == (0, 0.0, 0, 0.0)
    assert edges(four, ["a.py", "b.py"]) == []

    five = history(together(5, "a.py", "b.py"))
    assert rows(feats(five, ["a.py", "b.py"]))["a.py"] == (1, 1.0, 1, 0.0)

    lower = with_overrides(DEFAULT_SETTINGS, cochange_min_support=4)
    assert rows(feats(four, ["a.py", "b.py"], settings=lower))["a.py"] == (1, 1.0, 1, 0.0)
    assert edges(four, ["a.py", "b.py"], settings=lower) == [
        ("a.py", "b.py", 4, 1.0),
        ("b.py", "a.py", 4, 1.0),
    ]


def test_confidence_is_directional_and_inclusive() -> None:
    commits = together(6, "a.py", "b.py") + alone(6, "b.py", day=20)
    hist = history(commits)
    assert rows(feats(hist, ["a.py", "b.py"])) == {"a.py": (1, 1.0, 1, 0.0), "b.py": (1, 0.5, 1, 0.0)}
    assert edges(hist, ["a.py", "b.py"]) == [("a.py", "b.py", 6, 1.0), ("b.py", "a.py", 6, 0.5)]

    hist = history([*commits, (40, ["b.py"])])
    got = rows(feats(hist, ["a.py", "b.py"]))
    assert got["a.py"] == (1, 1.0, 1, 0.0)
    assert got["b.py"] == (0, 6 / 13, 0, 0.0)
    assert edges(hist, ["a.py", "b.py"]) == [("a.py", "b.py", 6, 1.0)]


def test_confidence_threshold_needs_no_rounding_slack() -> None:
    # 7 / 10 must reach a 0.7 threshold exactly (7 >= 0.7 * 10 fails in floating point).
    hist = history(together(7, "a.py", "b.py") + alone(3, "a.py", day=20))
    settings = with_overrides(DEFAULT_SETTINGS, cochange_min_confidence=0.7)
    assert rows(feats(hist, ["a.py", "b.py"], settings=settings))["a.py"] == (1, 0.7, 1, 0.0)


def test_hidden_coupling_follows_import_edges() -> None:
    hist = history(together(5, "a.py", "b.py"))
    paths = ["a.py", "b.py"]
    assert rows(feats(hist, paths, import_graph=None))["a.py"][2] == 1
    for g in (graph([("a.py", "b.py")]), graph([("b.py", "a.py")])):
        got = rows(feats(hist, paths, import_graph=g))
        assert got["a.py"] == (1, 1.0, 0, 0.0)
        assert got["b.py"] == (1, 1.0, 0, 0.0)
    unrelated = graph([("a.py", "c.py"), ("c.py", "b.py"), ("tests/test_a.py", "a.py")])
    assert rows(feats(hist, paths, import_graph=unrelated))["a.py"] == (1, 1.0, 1, 0.0)
    missing = graph([("x.py", "y.py")])
    assert rows(feats(hist, paths, import_graph=missing))["b.py"] == (1, 1.0, 1, 0.0)
    # Edges are returned whatever the import graph says.
    assert edges(hist, paths, import_graph=graph([("a.py", "b.py")])) == edges(hist, paths)


def test_bulk_commits_are_ignored() -> None:
    hist = history(together(5, "a.py", "b.py"), bulk={2})
    assert rows(feats(hist, ["a.py", "b.py"]))["a.py"] == (0, 0.0, 0, 0.0)
    assert edges(hist, ["a.py", "b.py"]) == []

    hist = history([(1, ["a.py"]), (2, ["a.py", "tests/test_a.py"])], bulk={1})
    assert rows(feats(hist, ["a.py"]))["a.py"] == (0, 0.0, 0, 0.0)


def test_history_stops_at_the_snapshot_time() -> None:
    before = together(4, "a.py", "b.py")
    at_t = history([*before, (100, ["a.py", "b.py"])])
    assert rows(feats(at_t, ["a.py", "b.py"]))["a.py"] == (1, 1.0, 1, 0.0)
    after_t = history([*before, (100 + 1 / SECONDS_PER_DAY, ["a.py", "b.py"])])
    assert rows(feats(after_t, ["a.py", "b.py"]))["a.py"] == (0, 0.0, 0, 0.0)
    # Sub-second snapshot times are floored like the miner's whole-second author times.
    late = Snapshot("demo", 0, T0 + timedelta(days=100, microseconds=999_999), "0" * 40)
    assert rows(feats(after_t, ["a.py", "b.py"], snapshot=late))["a.py"] == (0, 0.0, 0, 0.0)


def test_naive_snapshot_time_raises_unless_reachable_is_given() -> None:
    hist = history(together(5, "a.py", "b.py"))
    naive = Snapshot("demo", 0, datetime(2022, 4, 1), "0" * 40)
    with pytest.raises(ValueError, match="timezone-aware"):
        feats(hist, ["a.py", "b.py"], snapshot=naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        edges(hist, ["a.py", "b.py"], snapshot=naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        feats(hist, [], snapshot=naive)
    frame = feats(hist, ["a.py", "b.py"], snapshot=naive, reachable={sha(k) for k in range(5)})
    assert rows(frame)["a.py"] == (1, 1.0, 1, 0.0)


def test_reachable_replaces_the_time_rule() -> None:
    commits = [*together(5, "a.py", "b.py"), (200, ["a.py", "b.py"]), (3, ["a.py", "b.py"])]
    hist = history(commits, bulk={6})
    paths = ["a.py", "b.py"]
    assert rows(feats(hist, paths))["a.py"] == (1, 1.0, 1, 0.0)  # time rule: commits 0..4 and 6 is bulk

    # Commit 0 (before T) is unreachable, 5 (after T) is reachable, 6 is reachable but bulk.
    reachable = {sha(k) for k in range(1, 7)} | {"f" * 40}
    assert edges(hist, paths, reachable=reachable) == [("a.py", "b.py", 5, 1.0), ("b.py", "a.py", 5, 1.0)]
    reachable.discard(sha(5))
    assert rows(feats(hist, paths, reachable=reachable))["a.py"] == (0, 0.0, 0, 0.0)
    assert rows(feats(hist, paths, reachable=frozenset())) == {
        "a.py": (0, 0.0, 0, 0.0),
        "b.py": (0, 0.0, 0, 0.0),
    }


def test_renamed_files_keep_their_history() -> None:
    commits = [*together(5, "old/a.py", "b.py"), (10, [("old/a.py", "new/a.py")])]
    hist = history(commits)
    lineage = {"new/a.py": frozenset({"old/a.py", "new/a.py"}), "b.py": frozenset({"b.py"})}
    paths = ["new/a.py", "b.py"]
    assert rows(feats(hist, paths, lineage=lineage)) == {
        "b.py": (1, 1.0, 1, 0.0),
        "new/a.py": (1, 5 / 6, 1, 0.0),  # the rename row matches both names but counts once
    }
    assert edges(hist, paths, lineage=lineage) == [
        ("b.py", "new/a.py", 5, 1.0),
        ("new/a.py", "b.py", 5, 5 / 6),
    ]


def test_files_sharing_a_historical_name_are_never_partners() -> None:
    hist = history(together(5, "x.py", "z.py"))
    lineage = {"y.py": frozenset({"x.py", "y.py"}), "x.py": frozenset({"x.py"})}
    paths = ["x.py", "y.py", "z.py"]
    assert rows(feats(hist, paths, lineage=lineage)) == {
        "x.py": (1, 1.0, 1, 0.0),
        "y.py": (1, 1.0, 1, 0.0),
        "z.py": (2, 1.0, 2, 0.0),
    }
    assert edges(hist, paths, lineage=lineage) == [
        ("x.py", "z.py", 5, 1.0),
        ("y.py", "z.py", 5, 1.0),
        ("z.py", "x.py", 5, 1.0),
        ("z.py", "y.py", 5, 1.0),
    ]


def test_test_files_are_never_rows_or_partners() -> None:
    hist = history(together(5, "a.py", "tests/test_a.py"))
    paths = ["a.py", "tests/test_a.py"]
    assert rows(feats(hist, paths)) == {"a.py": (0, 0.0, 0, 1.0)}
    assert edges(hist, paths) == []


def test_test_cochange_rate_uses_historical_test_names() -> None:
    commits: list[Commit] = [
        (1, ["a.py", "tests/test_a.py"]),
        (2, ["a.py", "tests/old/test_gone.py"]),  # a test that no longer exists
        (3, ["a.py", ("tests/test_x.py", "examples/x.py")]),  # a test moved out of the tests folder
        (4, ["a.py"]),
        (5, ["a.py", "tests/test_a.py"]),  # bulk
        (6, ["tests/test_a.py"]),  # does not touch a.py
    ]
    hist = history(commits, bulk={4})
    assert rows(feats(hist, ["a.py", "tests/test_a.py"]))["a.py"] == (0, 0.0, 0, 0.75)


def test_test_cochange_rate_ignores_the_files_own_test_named_history() -> None:
    commits: list[Commit] = [
        (1, ["tests/helpers.py"]),
        (2, ["tests/helpers.py"]),
        (3, ["tests/helpers.py", "tests/test_x.py"]),
        (4, [("tests/helpers.py", "pkg/helpers.py")]),
    ]
    lineage = {"pkg/helpers.py": frozenset({"tests/helpers.py", "pkg/helpers.py"})}
    frame = feats(history(commits), ["pkg/helpers.py", "a.py"], lineage=lineage)
    assert rows(frame) == {"a.py": (0, 0.0, 0, 0.0), "pkg/helpers.py": (0, 0.0, 0, 0.25)}


def test_path_order_and_duplicates_do_not_matter() -> None:
    hist = history(together(5, "a.py", "b.py", "c.py") + alone(3, "c.py", day=30))
    paths = ["c.py", "a.py", "b.py", "tests/test_a.py"]
    snapshot = list(paths)
    frame = feats(hist, paths)
    pd.testing.assert_frame_equal(frame, feats(hist, ["b.py", "a.py", "c.py", "a.py", "c.py"]))
    assert edges(hist, paths) == edges(hist, ["b.py", "a.py", "c.py", "a.py", "c.py"])
    assert paths == snapshot  # inputs are not modified


def test_paths_missing_from_the_lineage_use_their_own_name() -> None:
    hist = history(together(5, "a.py", "b.py"))
    lineage = {"other.py": frozenset({"a.py", "other.py"})}
    assert rows(feats(hist, ["a.py", "b.py"], lineage=lineage)) == {
        "a.py": (1, 1.0, 1, 0.0),
        "b.py": (1, 1.0, 1, 0.0),
    }


def test_changes_of_unknown_commits_are_ignored() -> None:
    base = history(together(4, "a.py", "b.py"))
    orphans = pa.Table.from_pydict(
        {
            "sha": ["f" * 40, "e" * 40],
            "path": ["a.py", "b.py"],
            "old_path": [None, None],
            "added": [1, 1],
            "deleted": [0, 0],
            "is_binary": [False, False],
        },
        schema=FILE_CHANGES_SCHEMA,
    )
    hist = MinedHistory(base.commits, pa.concat_tables([base.file_changes, orphans, orphans]))
    assert rows(feats(hist, ["a.py", "b.py"]))["a.py"] == (0, 0.0, 0, 0.0)


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("cochange_min_support", 0),
        ("cochange_min_confidence", -0.1),
        ("cochange_min_confidence", 1.5),
        ("cochange_min_confidence", math.nan),
    ],
)
def test_invalid_thresholds_raise(override: str, value: float) -> None:
    hist = history(together(5, "a.py", "b.py"))
    settings = with_overrides(DEFAULT_SETTINGS, **{override: value})
    with pytest.raises(ValueError, match=override):
        feats(hist, ["a.py", "b.py"], settings=settings)
    with pytest.raises(ValueError, match=override):
        edges(hist, ["a.py", "b.py"], settings=settings)


def test_zero_confidence_makes_every_candidate_a_partner() -> None:
    hist = history(together(5, "a.py", "b.py") + alone(95, "b.py", day=1))
    settings = with_overrides(DEFAULT_SETTINGS, cochange_min_confidence=0.0)
    assert rows(feats(hist, ["a.py", "b.py"], settings=settings))["b.py"] == (1, 0.05, 1, 0.0)
    assert rows(feats(hist, ["a.py", "b.py"]))["b.py"] == (0, 0.05, 0, 0.0)


def test_edges_agree_with_the_features() -> None:
    hist, case = random_case(7)
    settings = with_overrides(DEFAULT_SETTINGS, cochange_min_support=2, cochange_min_confidence=0.3)
    frame = cochange_features(
        hist, snap(), case.lineage, case.paths, is_test_file, case.graph, settings, reachable=case.reachable
    )
    found = cochange_edges(
        hist, snap(), case.lineage, case.paths, is_test_file, case.graph, settings, reachable=case.reachable
    )
    got = rows(frame)
    assert found, "the case should have strong pairs"
    assert found == sorted(found)
    assert len({(a, b) for a, b, _, _ in found}) == len(found)
    for a, b, support, conf in found:
        assert type(a) is str and type(b) is str and type(support) is int and type(conf) is float
        assert a != b and a in got and b in got
        assert support >= 2 and 0.3 <= conf <= 1.0
        assert conf <= got[a][1]  # max_conf covers every candidate
    for path, (partners, _, _, _) in got.items():
        assert partners == sum(1 for a, _, _, _ in found if a == path), path


# ----------------------------------------------------------------------------- brute-force cross-check
POOL = (
    "a.py",
    "b.py",
    "c.py",
    "pkg/d.py",
    "pkg/e.py",
    "pkg/f.py",
    "old/g.py",
    "README.md",
    "tests/test_a.py",
    "tests/test_b.py",
    "tests/helpers.py",
    "conftest.py",
)


@dataclass(frozen=True)
class Case:
    paths: list[str]
    lineage: dict[str, frozenset[str]]
    graph: nx.DiGraph[str] | None
    reachable: set[str] | None
    settings: Settings


def random_case(seed: int) -> tuple[MinedHistory, Case]:
    rng = random.Random(seed)
    commits: list[Commit] = []
    for k in range(rng.randint(40, 60)):
        changes: list[Change] = []
        for name in rng.sample(POOL, rng.randint(1, 4)):
            if rng.random() < 0.15:
                changes.append((rng.choice([p for p in POOL if p != name]), name))
            else:
                changes.append(name)
        day = 100.0 if k % 17 == 0 else rng.uniform(0, 130)  # T is day 100
        commits.append((day, changes))
    bulk = {k for k in range(len(commits)) if rng.random() < 0.1}
    hist = history(commits, bulk=bulk)

    paths = rng.sample(POOL, rng.randint(2, len(POOL)))
    lineage = {
        p: frozenset({p, *rng.sample(POOL, rng.randint(0, 2))})
        for p in POOL
        if p in paths and rng.random() < 0.6
    }
    g: nx.DiGraph[str] | None = None
    if rng.random() < 0.7:
        pairs = [rng.sample(POOL, 2) for _ in range(rng.randint(0, 10))]
        g = graph([(a, b) for a, b in pairs], nodes=POOL)
    reachable: set[str] | None = None
    if rng.random() < 0.4:
        reachable = {sha(k) for k in range(len(commits)) if rng.random() < 0.6} | {"f" * 40}
    settings = with_overrides(
        DEFAULT_SETTINGS,
        cochange_min_support=rng.choice([1, 2, 3]),
        cochange_min_confidence=rng.choice([0.0, 0.3, 0.5, 1.0]),
    )
    return hist, Case(paths, lineage, g, reachable, settings)


def reference(hist: MinedHistory, case: Case) -> tuple[dict[str, Row], list[CochangeEdge]]:
    """Straightforward set-based version of the rules in the module docstring."""
    settings = case.settings
    cols = sorted({p for p in case.paths if not is_test_file(p)})
    names = {p: set(case.lineage.get(p, frozenset())) | {p} for p in cols}
    selected = set()
    for commit in hist.commits.to_pylist():
        if commit["is_bulk"]:
            continue
        if case.reachable is None:
            if commit["author_time"] <= snap().time:
                selected.add(commit["sha"])
        elif commit["sha"] in case.reachable:
            selected.add(commit["sha"])
    touched_names: dict[str, set[str]] = {}
    for change in hist.file_changes.to_pylist():
        if change["sha"] in selected:
            bucket = touched_names.setdefault(change["sha"], set())
            bucket.add(change["path"])
            if change["old_path"] is not None:
                bucket.add(change["old_path"])
    touches = {p: {s for s, ns in touched_names.items() if ns & names[p]} for p in cols}

    result: dict[str, Row] = {}
    found: list[CochangeEdge] = []
    for a in cols:
        n_a = len(touches[a])
        with_test = sum(1 for s in touches[a] if {x for x in touched_names[s] if is_test_file(x)} - names[a])
        partners = hidden = 0
        max_conf = 0.0
        for b in cols:
            if b == a or names[a] & names[b]:
                continue
            support = len(touches[a] & touches[b])
            if support < settings.cochange_min_support:
                continue
            conf = support / n_a
            max_conf = max(max_conf, conf)
            if conf >= settings.cochange_min_confidence:
                partners += 1
                g = case.graph
                if g is None or not (g.has_edge(a, b) or g.has_edge(b, a)):
                    hidden += 1
                found.append((a, b, support, conf))
        result[a] = (partners, max_conf, hidden, with_test / n_a if n_a else 0.0)
    return result, found


@pytest.mark.parametrize("seed", range(30))
def test_matches_a_brute_force_reference(seed: int) -> None:
    hist, case = random_case(seed)
    frame = cochange_features(
        hist,
        snap(),
        case.lineage,
        case.paths,
        is_test_file,
        case.graph,
        case.settings,
        reachable=case.reachable,
    )
    found = cochange_edges(
        hist,
        snap(),
        case.lineage,
        case.paths,
        is_test_file,
        case.graph,
        case.settings,
        reachable=case.reachable,
    )
    check_frame(frame)
    want_rows, want_edges = reference(hist, case)
    got = rows(frame)
    assert got.keys() == want_rows.keys()
    for path, (partners, max_conf, hidden, rate) in want_rows.items():
        assert got[path][0] == partners, path
        assert got[path][1] == pytest.approx(max_conf, rel=1e-12, abs=0), path
        assert got[path][2] == hidden, path
        assert got[path][3] == pytest.approx(rate, rel=1e-12, abs=0), path
    assert [(a, b, s) for a, b, s, _ in found] == [(a, b, s) for a, b, s, _ in want_edges]
    assert [c for _, _, _, c in found] == pytest.approx([c for _, _, _, c in want_edges], rel=1e-12)


# --------------------------------------------------------------------------------------- real git history
def test_end_to_end_on_a_scripted_repository(repo_builder: RepoBuilder, t0: datetime) -> None:
    day = timedelta(days=1)
    rb = repo_builder
    rb.commit(
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "import pkg.b\nA = 0\n",
            "pkg/b.py": "B = 0\n",
            "pkg/c.py": "C = 0\n",
            "tests/test_a.py": "import pkg.a\n",
        },
        "start",
        t0,
    )
    rb.commit({"pkg/a.py": "import pkg.b\nA = 1\n", "pkg/c.py": "C = 1\n"}, "one", t0 + day)
    rb.commit(
        {
            "pkg/a.py": "import pkg.b\nA = 2\n",
            "pkg/c.py": "C = 2\n",
            "tests/test_a.py": "import pkg.a\nX = 1\n",
        },
        "two",
        t0 + 2 * day,
    )
    rb.commit({"pkg/a.py": "import pkg.b\nA = 3\n", "pkg/b.py": "B = 1\n"}, "three", t0 + 3 * day)
    rb.commit({"pkg/a.py": "import pkg.b\nA = 4\n", "pkg/b.py": "B = 2\n"}, "four", t0 + 4 * day)
    rb.rename("pkg/c.py", "pkg/d.py", "rename", t0 + 5 * day)
    bulk: dict[str, str | bytes | None] = {f"data/f{i:02d}.txt": f"{i}\n" for i in range(49)}
    bulk |= {"pkg/a.py": "import pkg.b\nA = 5\n", "pkg/b.py": "B = 3\n"}
    rb.commit(bulk, "bulk", t0 + 6 * day)
    rb.commit({"pkg/a.py": "import pkg.b\nA = 6\n", "pkg/d.py": "C = 3\n"}, "late", t0 + 8 * day)

    t = t0 + 7 * day
    hist = mine_history(rb.path)
    snapshot = snapshot_at("demo", hist.commits, t)
    lineage = path_lineage(hist.file_changes, hist.commits, t)
    blobs = {p: b for p, b in list_files(rb.path, snapshot.sha).items() if p.endswith(".py")}
    sources = read_blobs(rb.path, blobs.values())
    syntax = {p: parse_python(p, sources[b]) for p, b in blobs.items()}
    g = build_import_graph(syntax, is_test_file)
    paths = list(syntax)
    settings = with_overrides(DEFAULT_SETTINGS, cochange_min_support=2)

    frame = cochange_features(hist, snapshot, lineage, paths, is_test_file, g, settings)
    check_frame(frame)
    got = rows(frame)
    assert got.keys() == {"pkg/__init__.py", "pkg/a.py", "pkg/b.py", "pkg/d.py"}
    assert got["pkg/__init__.py"] == (0, 0.0, 0, 1.0)
    assert got["pkg/a.py"] == (2, 0.6, 1, 0.4)
    assert got["pkg/b.py"][:3] == (1, 1.0, 0)
    assert got["pkg/b.py"][3] == pytest.approx(1 / 3)
    assert got["pkg/d.py"] == (1, 0.75, 1, 0.5)
    expected_edges = [
        ("pkg/a.py", "pkg/b.py", 3, 0.6),
        ("pkg/a.py", "pkg/d.py", 3, 0.6),
        ("pkg/b.py", "pkg/a.py", 3, 1.0),
        ("pkg/d.py", "pkg/a.py", 3, 0.75),
    ]
    assert cochange_edges(hist, snapshot, lineage, paths, is_test_file, g, settings) == expected_edges

    no_graph = cochange_features(hist, snapshot, lineage, paths, is_test_file, None, settings)
    assert no_graph["hidden_coupling"].tolist() == [0, 2, 1, 1]

    reachable = set(run_git(["rev-list", "--no-merges", snapshot.sha], rb.path).decode().split())
    assert len(reachable) == 7
    pd.testing.assert_frame_equal(
        cochange_features(hist, snapshot, lineage, paths, is_test_file, g, settings, reachable=reachable),
        frame,
    )
    assert (
        cochange_edges(hist, snapshot, lineage, paths, is_test_file, g, settings, reachable=reachable)
        == expected_edges
    )


# ------------------------------------------------------------------------------------------------- speed
def test_speed_1000_files_20000_commits() -> None:
    rng = random.Random(42)
    packages = [
        [f"pkg{p:02d}/mod{i:02d}.py" for i in range(45)] + [f"tests/test_{p:02d}_{i}.py" for i in range(5)]
        for p in range(20)
    ]
    files = [f for package in packages for f in package]
    commits: list[Commit] = []
    for _ in range(20_000):
        package = rng.choice(packages)
        changes: list[Change] = []
        if rng.random() < 0.1:  # the first two modules of each package are coupled
            changes.extend(package[:2])
        else:
            pool = package if rng.random() < 0.8 else files  # mostly changes inside one package
            for name in rng.sample(pool, rng.randint(1, 5)):
                changes.append((name.replace(".py", "_old.py"), name) if rng.random() < 0.02 else name)
        commits.append((rng.uniform(0, 99), changes))
    hist = history(commits, bulk={k for k in range(0, 20_000, 97)})
    lineage = {f: frozenset({f, f.replace(".py", "_old.py")}) for f in files}

    start = time.perf_counter()
    frame = cochange_features(hist, snap(), lineage, files, is_test_file, None)
    mid = time.perf_counter()
    found = cochange_edges(hist, snap(), lineage, files, is_test_file, None)
    end = time.perf_counter()

    assert len(frame) == 900
    assert {(a, b) for a, b, _, _ in found} >= {(p[0], p[1]) for p in packages} | {
        (p[1], p[0]) for p in packages
    }
    assert int(frame["cochange_partners"].sum()) == len(found)
    assert (frame["test_cochange_rate"] > 0).any()
    if traced():
        pytest.skip(f"timing not checked under coverage or a debugger ({mid - start:.1f}s, {end - mid:.1f}s)")
    assert mid - start < 10.0, f"features {mid - start:.1f}s"
    assert end - mid < 10.0, f"edges {end - mid:.1f}s"

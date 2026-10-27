from __future__ import annotations

from collections.abc import Iterable, Mapping

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from tdp_core.contracts import FEATURE_FAMILIES, FileSyntax
from tdp_core.features.tests_signals import TEST_SIGNAL_COLUMNS, test_features, top_level_package
from tdp_core.graph.imports import build_import_graph
from tdp_core.parsing.python import is_test_file, parse_python


def syn(path: str, sloc: int) -> FileSyntax:
    """A FileSyntax with only the path and SLOC filled in."""
    return FileSyntax(path, "python", sloc, 1, (), 0, (), (), (), (), False)


def graph(tests: Iterable[str], others: Iterable[str], edges: Iterable[tuple[str, str]]) -> nx.DiGraph[str]:
    """An import graph shaped like build_import_graph's output (edge a -> b: a imports b)."""
    g: nx.DiGraph[str] = nx.DiGraph()
    for t in tests:
        g.add_node(t, is_test=True)
    for o in others:
        g.add_node(o, is_test=False)
    g.add_edges_from(edges)
    return g


def has_test(path: str, test_paths: list[str]) -> int:
    """has_test_file of one path, with every file known only by name."""
    frame = test_features([path], test_paths, {}, graph(test_paths, [path], []))
    return int(frame["has_test_file"].to_numpy()[0])


NON_TEST_SLOC: Mapping[str, int] = {
    "pkg/__init__.py": 0,
    "pkg/a.py": 30,
    "pkg/b.py": 70,
    "other/x.py": 40,
    "src/lib1/m.py": 25,
    "src/lib1/sub/n.py": 25,
    "src/solo.py": 10,
    "setup.py": 5,
    "lib/vend/v.py": 20,
    "empty/__init__.py": 0,
}
TEST_SLOC: Mapping[str, int] = {
    "tests/test_a.py": 50,
    "tests/test_both.py": 10,
    "tests/helpers.py": 8,
    "tests/test_h.py": 12,
    "pkg/tests/test_inner.py": 6,
    "pkg/tests/conftest.py": 4,
    "tests/test_src.py": 15,
    "tests/test_solo.py": 9,
    "tests/test_empty.py": 7,
}
EDGES: tuple[tuple[str, str], ...] = (
    ("tests/test_a.py", "pkg/a.py"),
    ("tests/test_a.py", "pkg/b.py"),
    ("tests/test_both.py", "pkg/a.py"),
    ("tests/test_both.py", "other/x.py"),
    ("tests/test_h.py", "tests/helpers.py"),
    ("pkg/tests/test_inner.py", "pkg/b.py"),
    ("pkg/tests/test_inner.py", "pkg/tests/conftest.py"),
    ("tests/test_src.py", "src/lib1/sub/n.py"),
    ("tests/test_solo.py", "src/solo.py"),
    ("tests/test_empty.py", "empty/__init__.py"),
    ("other/x.py", "pkg/a.py"),
)
EXPECTED_RATIO: Mapping[str, float] = {
    "pkg/__init__.py": 0.66,
    "pkg/a.py": 0.66,
    "pkg/b.py": 0.66,
    "other/x.py": 0.25,
    "src/lib1/m.py": 0.3,
    "src/lib1/sub/n.py": 0.3,
    "src/solo.py": 0.9,
    "setup.py": 0.0,
    "lib/vend/v.py": 0.0,
    "empty/__init__.py": 7.0,
}


def hand_built() -> tuple[dict[str, FileSyntax], nx.DiGraph[str]]:
    syntax = {p: syn(p, n) for p, n in {**NON_TEST_SLOC, **TEST_SLOC}.items()}
    return syntax, graph(TEST_SLOC, NON_TEST_SLOC, EDGES)


def test_columns_are_two_of_family_t() -> None:
    assert TEST_SIGNAL_COLUMNS == ("has_test_file", "pkg_test_ratio")
    assert set(TEST_SIGNAL_COLUMNS) <= set(FEATURE_FAMILIES["T"])
    # The other two family T columns come from the import graph and the co-change features.
    assert "test_importers" not in TEST_SIGNAL_COLUMNS
    assert "test_cochange_rate" not in TEST_SIGNAL_COLUMNS


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/a/b.py", "pkg"),
        ("src/pkg/a.py", "src/pkg"),
        ("src/pkg/__init__.py", "src/pkg"),
        ("lib/pkg/x.py", "lib/pkg"),
        ("src/mod.py", "src/mod.py"),
        ("setup.py", "setup.py"),
        ("srcx/a.py", "srcx"),
    ],
)
def test_top_level_package(path: str, expected: str) -> None:
    assert top_level_package(path) == expected


def test_has_test_file_prefix_and_suffix_names() -> None:
    assert has_test("pkg/utils.py", ["tests/test_utils.py"]) == 1
    assert has_test("pkg/core.py", ["tests/core_test.py"]) == 1
    assert has_test("pkg/other.py", ["tests/test_utils.py", "tests/core_test.py"]) == 0


def test_has_test_file_matches_basenames_in_any_directory() -> None:
    assert has_test("a/b/c/deep.py", ["x/y/test_deep.py"]) == 1


def test_has_test_file_is_zero_for_init_files() -> None:
    assert has_test("pkg/__init__.py", ["tests/test___init__.py", "tests/test_init.py"]) == 0


@pytest.mark.parametrize(
    "test_path", ["tests/test_compat.py", "tests/test__compat.py", "tests/compat_test.py"]
)
def test_has_test_file_private_module_matches_without_underscore(test_path: str) -> None:
    assert has_test("pkg/_compat.py", [test_path]) == 1


def test_has_test_file_dunder_module_needs_its_full_stem() -> None:
    assert has_test("pkg/__main__.py", ["tests/test_main.py"]) == 0
    assert has_test("pkg/__main__.py", ["tests/test___main__.py"]) == 1


def test_has_test_file_only_counts_paths_in_test_paths() -> None:
    syntax = {p: syn(p, 1) for p in ("pkg/utils.py", "scripts/test_utils.py")}
    g = graph([], syntax, [])
    frame = test_features(["pkg/utils.py"], [], syntax, g)
    assert frame.loc["pkg/utils.py", "has_test_file"] == 0


def test_has_test_file_is_case_sensitive() -> None:
    assert has_test("pkg/utils.py", ["tests/Test_Utils.py"]) == 0
    assert has_test("pkg/utils.py", ["tests/test_Utils.py"]) == 0


def test_pkg_test_ratio_on_hand_built_graph() -> None:
    syntax, g = hand_built()
    frame = test_features(list(NON_TEST_SLOC), list(TEST_SLOC), syntax, g)
    assert list(frame.index) == sorted(NON_TEST_SLOC)
    for path, expected in EXPECTED_RATIO.items():
        assert frame.loc[path, "pkg_test_ratio"] == pytest.approx(expected), path
    expected_has = {
        "pkg/a.py": 1,
        "pkg/b.py": 0,
        "src/solo.py": 1,
        "setup.py": 0,
        "pkg/__init__.py": 0,
        "empty/__init__.py": 0,
    }
    for path, flag in expected_has.items():
        assert frame.loc[path, "has_test_file"] == flag, path


def test_a_row_does_not_depend_on_the_other_requested_rows() -> None:
    syntax, g = hand_built()
    frame = test_features(["pkg/a.py"], list(TEST_SLOC), syntax, g)
    assert list(frame.index) == ["pkg/a.py"]
    assert frame.loc["pkg/a.py", "pkg_test_ratio"] == pytest.approx(0.66)


def test_tolerates_missing_syntax_and_graph_nodes() -> None:
    syntax, g = hand_built()
    # A test node without syntax adds 0 SLOC to pkg, and a test path absent from the graph adds nothing.
    g.add_node("tests/test_ghost.py", is_test=True)
    g.add_edge("tests/test_ghost.py", "pkg/a.py")
    syntax["tests/test_offgraph.py"] = syn("tests/test_offgraph.py", 1000)
    tests = [*TEST_SLOC, "tests/test_ghost.py", "tests/test_offgraph.py", "tests/test_g.py"]
    frame = test_features(["pkg/a.py", "ghost/g.py", "src/solo.py"], tests, syntax, g)
    assert frame.loc["pkg/a.py", "pkg_test_ratio"] == pytest.approx(0.66)
    assert frame.loc["src/solo.py", "pkg_test_ratio"] == pytest.approx(0.9)
    # ghost/g.py is in neither the syntax map nor the graph: ratio 0, name-based has_test_file.
    assert frame.loc["ghost/g.py", "pkg_test_ratio"] == 0.0
    assert frame.loc["ghost/g.py", "has_test_file"] == 1


def test_test_paths_decide_membership_not_the_node_attribute() -> None:
    syntax, g = hand_built()
    g.add_node("tools/check.py", is_test=True)
    g.add_edge("tools/check.py", "pkg/a.py")
    syntax["tools/check.py"] = syn("tools/check.py", 500)
    frame = test_features(["pkg/a.py", "tools/check.py"], list(TEST_SLOC), syntax, g)
    assert frame.loc["pkg/a.py", "pkg_test_ratio"] == pytest.approx(0.66)
    # As a non-test file, tools/check.py is part of its own package's source SLOC.
    assert frame.loc["tools/check.py", "pkg_test_ratio"] == 0.0


def test_test_files_as_rows_follow_the_same_rules() -> None:
    syntax, g = hand_built()
    frame = test_features(["pkg/tests/test_inner.py", "tests/test_a.py"], list(TEST_SLOC), syntax, g)
    # pkg/tests/test_inner.py groups with pkg; tests/ has no non-test source, so its ratio divides by 1.
    assert frame.loc["pkg/tests/test_inner.py", "pkg_test_ratio"] == pytest.approx(0.66)
    assert frame.loc["tests/test_a.py", "pkg_test_ratio"] == 0.0


def test_output_format() -> None:
    syntax, g = hand_built()
    frame = test_features(["setup.py", "pkg/a.py", "setup.py", "pkg/a.py"], list(TEST_SLOC), syntax, g)
    expected_index = pd.Index(["pkg/a.py", "setup.py"], dtype="str", name="path")
    pd.testing.assert_index_equal(frame.index, expected_index)
    assert tuple(frame.columns) == TEST_SIGNAL_COLUMNS
    assert frame["has_test_file"].dtype == np.int64
    assert frame["pkg_test_ratio"].dtype == np.float64
    assert not frame.isna().to_numpy().any()


def test_empty_paths_give_an_empty_frame() -> None:
    syntax, g = hand_built()
    frame = test_features([], list(TEST_SLOC), syntax, g)
    assert frame.empty
    assert tuple(frame.columns) == TEST_SIGNAL_COLUMNS
    assert frame["has_test_file"].dtype == np.int64
    assert frame["pkg_test_ratio"].dtype == np.float64
    pd.testing.assert_index_equal(frame.index, pd.Index([], dtype="str", name="path"))


def test_with_real_parser_and_import_graph() -> None:
    sources = {
        "src/pkg/__init__.py": b"from pkg.core import run\n",
        "src/pkg/core.py": b"def run():\n    return 1\n",
        "src/pkg/_util.py": b"X = 1\nY = 2\n",
        "tests/test_core.py": b"from pkg.core import run\n\n\ndef test_run():\n    assert run() == 1\n",
        "tests/test_util.py": b"from pkg import _util\n\n\ndef test_x():\n    assert _util.X == 1\n",
        "setup.py": b"import setuptools\n\nsetuptools.setup()\n",
    }
    syntax = {p: parse_python(p, s) for p, s in sources.items()}
    tests = [p for p in syntax if is_test_file(p)]
    g = build_import_graph(syntax, is_test_file)
    assert sorted(tests) == ["tests/test_core.py", "tests/test_util.py"]
    assert {p: s.sloc for p, s in syntax.items()} == {
        "src/pkg/__init__.py": 1,
        "src/pkg/core.py": 2,
        "src/pkg/_util.py": 2,
        "tests/test_core.py": 3,
        "tests/test_util.py": 3,
        "setup.py": 2,
    }
    assert sorted(g.edges) == [
        ("src/pkg/__init__.py", "src/pkg/core.py"),
        ("tests/test_core.py", "src/pkg/core.py"),
        ("tests/test_util.py", "src/pkg/_util.py"),
    ]
    rows = [p for p in syntax if p not in tests]
    frame = test_features(rows, tests, syntax, g)
    assert frame["has_test_file"].to_dict() == {
        "setup.py": 0,
        "src/pkg/__init__.py": 0,
        "src/pkg/_util.py": 1,
        "src/pkg/core.py": 1,
    }
    for path in ("src/pkg/__init__.py", "src/pkg/core.py", "src/pkg/_util.py"):
        assert frame.loc[path, "pkg_test_ratio"] == pytest.approx((3 + 3) / (1 + 2 + 2)), path
    assert frame.loc["setup.py", "pkg_test_ratio"] == 0.0


def test_test_features_is_not_collected_by_pytest() -> None:
    assert getattr(test_features, "__test__", True) is False

"""Test signal features (family T): ``has_test_file`` and ``pkg_test_ratio``.

Family T has four columns. This module computes two of them: ``test_importers`` comes from
``graph_features`` (the import graph) and ``test_cochange_rate`` from the co-change features.

Definitions shared by both columns:

- **Test files** are exactly the paths in ``test_paths``. The ``is_test`` node attribute of the import graph
  is never read, so the caller decides membership in one place.
- **has_test_file** is 1 when some test file's basename is ``test_<stem>.py`` or ``<stem>_test.py`` for the
  file's stem, anywhere in the repository, and 0 otherwise. Matching is exact and case sensitive. Every
  ``__init__.py`` gets 0. A private module also matches tests named after its stem without the leading
  underscores (``_client.py`` matches ``test_client.py`` as well as ``test__client.py``); dunder modules
  such as ``__main__.py`` only match their full stem.
- **Package key** (``top_level_package``): the first path component, or the first two when the first is a
  source root (``src`` or ``lib``, the same roots the import graph strips from module names). Files at the
  repository root and files directly under a source root are their own package.
- **pkg_test_ratio** of a file is ``tested_sloc / max(source_sloc, 1)`` for the file's package, where
  ``source_sloc`` is the SLOC of every non-test file of that package in ``syntax_by_path`` (not only the
  requested rows, so a row never depends on which other rows were requested) and ``tested_sloc`` is the SLOC
  of the test files that import at least one non-test module of the package. Each test counts once per
  package it imports, fully for every such package. Test to test edges and non-test importers never count.
  The value is 0 or more and can exceed 1 (more test code than source code).

Tolerated inputs (none raises): a row or test missing from ``syntax_by_path`` counts as 0 SLOC, a test path
that is not a node of the graph contributes nothing, and ``paths`` may contain test files (they follow the
same rules). The cost is O(files + edges).
"""

from __future__ import annotations

import logging
import posixpath
from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from tdp_core.contracts import FileSyntax
from tdp_core.graph.imports import SOURCE_ROOTS

logger = logging.getLogger(__name__)

__all__ = ["TEST_SIGNAL_COLUMNS", "test_features", "top_level_package"]

TEST_SIGNAL_COLUMNS: tuple[str, ...] = ("has_test_file", "pkg_test_ratio")


def top_level_package(path: str) -> str:
    """Grouping key of the package a repo-relative POSIX path belongs to.

    The key is the first path component, or ``<root>/<name>`` when the first component is a source root
    (``src`` or ``lib``): ``pkg/a/b.py`` gives ``pkg``, ``src/pkg/a.py`` gives ``src/pkg``, ``src/mod.py``
    gives ``src/mod.py`` and ``setup.py`` gives ``setup.py``. The key only groups files; it is not an
    import name.
    """
    parts = path.split("/")
    if len(parts) > 1 and parts[0] in SOURCE_ROOTS:
        return parts[0] + "/" + parts[1]
    return parts[0]


def _test_stems(path: str) -> tuple[str, ...]:
    """Stems a test file may be named after: its stem, and the stem without leading underscores."""
    stem = posixpath.splitext(posixpath.basename(path))[0]
    if stem == "__init__":
        return ()
    if stem.startswith("__") and stem.endswith("__"):
        return (stem,)
    bare = stem.lstrip("_")
    if bare and bare != stem:
        return (stem, bare)
    return (stem,)


def test_features(
    paths: Sequence[str],
    test_paths: Sequence[str],
    syntax_by_path: Mapping[str, FileSyntax],
    import_graph: nx.DiGraph[str],
) -> pd.DataFrame:
    """``has_test_file`` (0/1, int64) and ``pkg_test_ratio`` (test SLOC per source SLOC, float64) per path.

    ``paths`` are the rows (repo-relative POSIX, duplicates collapse), ``test_paths`` the test files,
    ``syntax_by_path`` the parsed files (test files included, so their SLOC is known) and ``import_graph``
    the graph from ``build_import_graph`` (edge ``a -> b`` means ``a`` imports ``b``). The module docstring
    gives the exact rules.

    Returns a frame indexed by the sorted unique ``paths`` (index name ``"path"``, dtype ``str``) with
    columns exactly ``TEST_SIGNAL_COLUMNS`` and no NaN. Empty ``paths`` give an empty frame with the same
    columns and dtypes.
    """
    rows = sorted(set(paths))
    tests = frozenset(test_paths)
    test_names = frozenset(t.rsplit("/", 1)[-1] for t in tests)

    source_sloc: dict[str, int] = {}
    for p, syntax in syntax_by_path.items():
        if p not in tests:
            key = top_level_package(p)
            source_sloc[key] = source_sloc.get(key, 0) + syntax.sloc

    tested_sloc: dict[str, int] = {}
    for t in sorted(tests):
        if t not in import_graph:
            continue
        pkgs = {top_level_package(m) for m in import_graph.successors(t) if m not in tests}
        syntax_t = syntax_by_path.get(t)
        sloc_t = syntax_t.sloc if syntax_t is not None else 0
        for key in pkgs:
            tested_sloc[key] = tested_sloc.get(key, 0) + sloc_t

    has: list[int] = []
    ratio: list[float] = []
    for p in rows:
        stems = _test_stems(p)
        has.append(int(any(f"test_{s}.py" in test_names or f"{s}_test.py" in test_names for s in stems)))
        key = top_level_package(p)
        ratio.append(tested_sloc.get(key, 0) / max(source_sloc.get(key, 0), 1))

    logger.debug(
        "test features: %d rows, %d test files, %d packages with tests",
        len(rows),
        len(tests),
        len(tested_sloc),
    )
    return pd.DataFrame(
        {
            "has_test_file": np.asarray(has, dtype=np.int64),
            "pkg_test_ratio": np.asarray(ratio, dtype=np.float64),
        },
        index=pd.Index(rows, dtype="str", name="path"),
    )


# pytest collects any module-level callable named test_* in a test module, imported ones included.
test_features.__test__ = False  # type: ignore[attr-defined]

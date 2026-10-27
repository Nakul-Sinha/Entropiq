"""Python import resolution and the repository import graph.

Module names come from repo-relative POSIX paths: ``a/b/c.py`` is ``a.b.c`` and ``a/b/__init__.py`` is
``a.b``. Files under a top-level ``src/`` or ``lib/`` directory are registered under both names (with and
without that first component), which covers the common source layouts. Only paths whose components are
valid Python identifiers can be imported, so other ``.py`` files (``scripts/my-tool.py``, ``.github/x.py``)
are not registered. Imports that do not resolve to a file in the repository (standard library, third-party
packages) are dropped. Nothing outside the repository is ever looked at.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping

import networkx as nx

from tdp_core.contracts import FileSyntax, RawImport

logger = logging.getLogger(__name__)

SOURCE_ROOTS = frozenset({"src", "lib"})


class ModuleIndex:
    """Maps dotted Python module names to repo-relative file paths. Create it with :meth:`build`."""

    __slots__ = ("_files",)

    def __init__(self, files: Mapping[str, str]) -> None:
        self._files = dict(files)

    @classmethod
    def build(cls, paths: Iterable[str]) -> ModuleIndex:
        """Index every ``.py`` path in ``paths`` (repo-relative POSIX strings; other files are ignored).

        A file gets the name of its path (``a/b/c.py`` -> ``a.b.c``, ``a/b/__init__.py`` -> ``a.b``) and, if
        its first component is ``src`` or ``lib``, also the name without it. When two files claim the same
        name the shorter path wins (ties: the lexicographically smaller path), so ``a/b.py`` beats
        ``a/b/__init__.py`` and ``pkg/m.py`` beats ``src/pkg/m.py``. A root ``__init__.py`` has no name.
        """
        files: dict[str, str] = {}
        for path in sorted(set(paths)):
            for name in _module_names(path):
                current = files.get(name)
                if current is None or (len(path), path) < (len(current), current):
                    files[name] = path
        return cls(files)

    def lookup(self, dotted: str) -> str | None:
        """File path of the module ``dotted`` (exact name, for example ``"a.b"``), or None if unknown."""
        return self._files.get(dotted)

    def __len__(self) -> int:
        """Number of registered module names (a ``src/`` file can count twice)."""
        return len(self._files)


def resolve(imp: RawImport, importer: str, index: ModuleIndex) -> list[str]:
    """Repository files that ``imp``, found in the file ``importer``, depends on.

    Absolute imports (``level == 0``): ``import a.b.c`` gives the file of ``a.b.c`` if it exists, else of the
    longest existing prefix (``a.b``, then ``a``). ``from a.b import x, y`` gives, for each name, the file of
    the submodule ``a.b.x`` if it exists, otherwise the file of ``a.b`` (with the same prefix fallback).

    Relative imports (``level >= 1``): the base package is the importer's directory (for ``__init__.py`` that
    is its own package), moved up ``level - 1`` directories; ``module`` is appended and the ``from`` rule
    above applies. Going above the repository root returns ``[]``; the root itself is a valid base.

    Unknown modules (standard library, third-party) give ``[]``. The result has no duplicates, keeps the
    order of ``imp.names``, and may contain ``importer`` itself (``from . import name_in_init``); the graph
    builder drops such self references.
    """
    if imp.level <= 0:
        base = imp.module
    else:
        package = importer.split("/")[:-1]
        up = imp.level - 1
        if up > len(package):
            return []
        parts = package[: len(package) - up]
        if imp.module:
            parts += imp.module.split(".")
        base = ".".join(parts)

    if not imp.names:
        hit = _longest_prefix(index, base)
        return [hit] if hit is not None else []

    out: list[str] = []
    for name in imp.names:
        hit = index.lookup(f"{base}.{name}" if base else name)
        if hit is None:
            hit = _longest_prefix(index, base)
        if hit is not None and hit not in out:
            out.append(hit)
    return out


def build_import_graph(
    syntax_by_path: Mapping[str, FileSyntax], is_test: Callable[[str], bool]
) -> nx.DiGraph[str]:
    """Directed import graph over the files in ``syntax_by_path`` (keys are repo-relative paths).

    Every key becomes a node with the boolean attribute ``is_test`` (from ``is_test(path)``), including files
    that import nothing and are imported by nothing. An edge ``a -> b`` means ``a`` imports ``b``; imports
    are resolved with :func:`resolve` against a :class:`ModuleIndex` of the keys. Self references are
    dropped and repeated imports of the same file collapse into one edge (edges carry no attributes).
    Nodes are added in sorted path order.
    """
    paths = sorted(syntax_by_path)
    index = ModuleIndex.build(paths)
    graph: nx.DiGraph[str] = nx.DiGraph()
    for path in paths:
        graph.add_node(path, is_test=bool(is_test(path)))
    for path in paths:
        for imp in syntax_by_path[path].imports:
            for target in resolve(imp, path, index):
                if target != path:
                    graph.add_edge(path, target)
    logger.debug(
        "import graph: %d files, %d edges, %d module names",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(index),
    )
    return graph


def _module_names(path: str) -> list[str]:
    """Dotted names a file can be imported as (empty for non-Python or non-importable paths)."""
    if not path.endswith(".py"):
        return []
    parts = path[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    if not parts or not all(part.isidentifier() for part in parts):
        return []
    names = [".".join(parts)]
    if parts[0] in SOURCE_ROOTS and len(parts) > 1:
        names.append(".".join(parts[1:]))
    return names


def _longest_prefix(index: ModuleIndex, dotted: str) -> str | None:
    """File of ``dotted`` or of its longest existing dotted prefix, or None."""
    parts = dotted.split(".") if dotted else []
    while parts:
        hit = index.lookup(".".join(parts))
        if hit is not None:
            return hit
        parts.pop()
    return None

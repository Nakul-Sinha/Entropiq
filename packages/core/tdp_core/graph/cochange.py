"""Co-change (logical) coupling features: family I, plus ``test_cochange_rate`` of family T.

Files that keep changing in the same commits are logically coupled even when neither imports the other
(Gall, Hajek and Jazayeri, "Detection of logical coupling based on product release history", ICSM 1998).
The history up to the snapshot becomes a binary commits x files matrix ``M`` (one row per history commit
that touches a ranked file, one column per ranked file), and every pair count comes from one sparse product
``S = M.T @ M``: ``S[a, a]`` is the number of history commits touching ``a`` and ``S[a, b]`` the number
touching both ``a`` and ``b``.

Terms shared by :func:`cochange_features` and :func:`cochange_edges`:

- Ranked files: the members of ``paths`` for which ``is_test`` is false. Only they get rows and only they
  can be partners; test files and files outside ``paths`` (deleted files, docs, config) never are.
- History: the commits of ``hist.commits`` with ``is_bulk`` false and ``author_time <= snapshot.time``
  (inclusive, whole seconds). When ``reachable`` is given it replaces the time rule: history is the commits
  with ``is_bulk`` false whose sha is in ``reachable``, and ``snapshot.time`` is not consulted.
- Touch: a commit touches a ranked file when any of its ``file_changes`` rows has ``path`` or ``old_path``
  among the file's names (``lineage.get(path, frozenset()) | {path}``), so renamed files keep their history.
  A commit counts once per file however many of its rows match.
- ``support(a, b)``: number of history commits touching both ``a`` and ``b`` (symmetric, a commit count).
- ``conf(a -> b) = support(a, b) / commits(a)``, in 0..1, where ``commits(a)`` is the number of history
  commits touching ``a``. It is directional: the share of a's commits that also touch b.
- Candidate partner of ``a``: another ranked file ``b`` that shares no name with ``a`` and has
  ``support(a, b) >= settings.cochange_min_support``.
- Strong partner of ``a``: a candidate with ``conf(a -> b) >= settings.cochange_min_confidence``.

Lineage overlap: lineage is name based, so when a file is renamed and a new file later reuses the old name,
every commit to that name counts for both files and the pair looks perfectly coupled through commits that
edited only one of them. Two ranked files whose names overlap are therefore never candidates of each other
(their own commit counts still include the shared commits).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import Snapshot
from tdp_core.mining.gitlog import MinedHistory

logger = logging.getLogger(__name__)

COCHANGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "cochange_partners",
    "cochange_max_conf",
    "hidden_coupling",
    "test_cochange_rate",
)

# (a, b, support, conf a -> b): shared history commits and the share of a's history commits that touch b.
type CochangeEdge = tuple[str, str, int, float]


@dataclass(frozen=True, slots=True)
class _Pairs:
    """Per-file counts and the candidate pairs (sorted by source, then target) for one set of inputs."""

    cols: list[str]  # ranked files, sorted; positions are the matrix columns
    commits: npt.NDArray[np.int64]  # history commits touching each file
    test_commits: npt.NDArray[np.int64]  # ... that also touch another (test-named) file
    src: npt.NDArray[np.int64]  # candidate pair (src -> dst) column positions
    dst: npt.NDArray[np.int64]
    support: npt.NDArray[np.int64]
    conf: npt.NDArray[np.float64]  # support / commits[src]
    strong: npt.NDArray[np.bool_]  # conf >= cochange_min_confidence


def cochange_features(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    is_test: Callable[[str], bool],
    import_graph: nx.DiGraph[str] | None,
    settings: Settings = DEFAULT_SETTINGS,
    *,
    reachable: AbstractSet[str] | None = None,
) -> pd.DataFrame:
    """Co-change features of the ranked (non-test) files of ``paths`` at ``snapshot``.

    See the module docstring for history, touch, support, confidence and partners. Arguments:

    - ``hist``: mined history. Only the commit columns ``sha``, ``author_time``, ``is_bulk`` and the change
      columns ``sha``, ``path``, ``old_path`` are read; ``is_bulk`` is trusted as mined. Change rows whose sha
      is not a history commit (bulk, after T, unreachable or unknown) are ignored.
    - ``snapshot``: ``snapshot.time`` (timezone-aware) bounds the history when ``reachable`` is None.
    - ``lineage``: current path -> every historical name, as returned by ``path_lineage`` for the same
      history. Paths missing from it use their own name only.
    - ``paths``: the files to rank; order and duplicates do not matter and test files are dropped.
    - ``is_test``: test-file predicate, applied to ``paths`` and to every historical name in the history.
    - ``import_graph``: import graph (edge ``a -> b`` when a imports b, as built by ``build_import_graph``) or
      None.
    - ``settings``: ``cochange_min_support`` (commits, at least 1) and ``cochange_min_confidence`` (0..1).
    - ``reachable``: shas of the non-merge commits reachable from the snapshot commit. When given it replaces
      the author-time rule; bulk commits stay excluded and shas that are not in ``hist.commits`` are ignored.

    Returns a DataFrame indexed by the sorted ranked paths (index name ``"path"``, dtype ``str``) with the
    columns of :data:`COCHANGE_FEATURE_COLUMNS`:

    - ``cochange_partners`` (int64): number of strong partners.
    - ``cochange_max_conf`` (float64, 0..1): highest ``conf(a -> b)`` over the candidate partners, whatever
      the confidence threshold; 0.0 without candidates, so it can be below the threshold while partners is 0.
    - ``hidden_coupling`` (int64): strong partners with no import edge in either direction (files missing
      from the graph have no edges); equals ``cochange_partners`` when ``import_graph`` is None.
    - ``test_cochange_rate`` (float64, 0..1): share of the file's history commits that also touch a test file,
      meaning a historical name (``path`` or ``old_path`` of the commit's rows, so deleted and renamed tests
      count) for which ``is_test`` is true and which is not one of the file's own names; 0.0 without commits.

    Files without history get zeros. No ranked paths gives an empty frame with the same columns and dtypes.
    The inputs are not modified.

    Raises ``ValueError`` when ``cochange_min_support < 1``, ``cochange_min_confidence`` is outside 0..1, or
    ``snapshot.time`` is naive while ``reachable`` is None.
    """
    pairs = _pairs(hist, snapshot, lineage, paths, is_test, settings, reachable)
    n = len(pairs.cols)
    src, dst, strong = pairs.src, pairs.dst, pairs.strong

    max_conf = np.zeros(n, dtype=np.float64)
    np.maximum.at(max_conf, src, pairs.conf)
    partners = np.bincount(src[strong], minlength=n).astype(np.int64)
    if import_graph is None:
        hidden = partners.copy()
    else:
        hidden = np.zeros(n, dtype=np.int64)
        for a, b in zip(src[strong].tolist(), dst[strong].tolist(), strict=True):
            path_a, path_b = pairs.cols[a], pairs.cols[b]
            if not (import_graph.has_edge(path_a, path_b) or import_graph.has_edge(path_b, path_a)):
                hidden[a] += 1

    rate = np.zeros(n, dtype=np.float64)
    active = pairs.commits > 0
    rate[active] = pairs.test_commits[active] / pairs.commits[active]

    columns = {
        "cochange_partners": partners,
        "cochange_max_conf": max_conf,
        "hidden_coupling": hidden,
        "test_cochange_rate": rate,
    }
    return pd.DataFrame(
        {col: columns[col] for col in COCHANGE_FEATURE_COLUMNS},
        index=pd.Index(pairs.cols, dtype="str", name="path"),
    )


def cochange_edges(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    is_test: Callable[[str], bool],
    import_graph: nx.DiGraph[str] | None,
    settings: Settings = DEFAULT_SETTINGS,
    *,
    reachable: AbstractSet[str] | None = None,
) -> list[CochangeEdge]:
    """Every strong co-change pair among the ranked files, as ``(a, b, support, conf a -> b)`` tuples.

    Takes the same arguments as :func:`cochange_features` and applies the same rules (see the module
    docstring). ``import_graph`` is accepted so both functions share one call signature but is not used:
    pairs are returned with or without an import edge, and callers that want hidden coupling drop the pairs
    with an edge in either direction.

    ``support`` (``int``) is the number of shared history commits and ``conf`` (``float``, 0..1) the share of
    a's history commits that also touch b. ``(a, b)`` and ``(b, a)`` both appear when both directions are
    strong; their confidences differ when the two files have different commit counts. The list is sorted by
    ``(a, b)``, holds ``cochange_partners`` entries with source ``a`` for every ranked file ``a``, and is
    empty when there are no ranked files.

    Raises the same ``ValueError`` as :func:`cochange_features`.
    """
    pairs = _pairs(hist, snapshot, lineage, paths, is_test, settings, reachable)
    strong = pairs.strong
    return [
        (pairs.cols[a], pairs.cols[b], int(s), float(c))
        for a, b, s, c in zip(
            pairs.src[strong].tolist(),
            pairs.dst[strong].tolist(),
            pairs.support[strong].tolist(),
            pairs.conf[strong].tolist(),
            strict=True,
        )
    ]


# ---------------------------------------------------------------------------------------------- internals
def _pairs(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    is_test: Callable[[str], bool],
    settings: Settings,
    reachable: AbstractSet[str] | None,
) -> _Pairs:
    """Commit counts per ranked file and the candidate pairs, shared by both public functions."""
    min_support = settings.cochange_min_support
    min_conf = settings.cochange_min_confidence
    if min_support < 1:
        raise ValueError(f"cochange_min_support must be at least 1, got {min_support}")
    if not 0.0 <= min_conf <= 1.0:
        raise ValueError(f"cochange_min_confidence must be in 0..1, got {min_conf}")
    cutoff = None if reachable is not None else _unix_seconds(snapshot.time)

    test_memo: dict[str, bool] = {}

    def test_name(name: str) -> bool:
        flag = test_memo.get(name)
        if flag is None:
            flag = test_memo[name] = bool(is_test(name))
        return flag

    cols = sorted({p for p in paths if not test_name(p)})
    n = len(cols)
    if n == 0:
        return _empty_pairs()
    selected = _history_shas(hist.commits, cutoff, reachable)

    owners: dict[str, list[int]] = {}  # historical name -> ranked files that carry it
    own_tests: dict[int, frozenset[str]] = {}  # ranked file -> its own test-named historical names
    for col, path in enumerate(cols):
        names = set(lineage.get(path, frozenset())) | {path}
        for name in names:
            owners.setdefault(name, []).append(col)
        tests = frozenset(name for name in names if test_name(name))
        if tests:
            own_tests[col] = tests
    conflict_keys = np.array(
        sorted({a * n + b for js in owners.values() if len(js) > 1 for a in js for b in js if a != b}),
        dtype=np.int64,
    )

    touched: dict[str, set[int]] = {}  # history sha -> ranked files it touches
    tests_by_sha: dict[str, set[str]] = {}  # history sha -> test-named names it touches
    fc = hist.file_changes
    for sha, path, old_path in zip(
        fc.column("sha").to_pylist(),
        fc.column("path").to_pylist(),
        fc.column("old_path").to_pylist(),
        strict=True,
    ):
        if sha not in selected:
            continue
        for name in (path, old_path):
            if name is None:
                continue
            if test_name(name):
                tests_by_sha.setdefault(sha, set()).add(name)
            owned = owners.get(name)
            if owned:
                touched.setdefault(sha, set()).update(owned)

    rows = sorted(touched)
    row_list: list[int] = []
    col_list: list[int] = []
    for k, sha in enumerate(rows):
        files = touched[sha]
        row_list.extend([k] * len(files))
        col_list.extend(files)
    r = np.array(row_list, dtype=np.int64)
    c = np.array(col_list, dtype=np.int64)
    # (row, col) pairs are unique, so M is binary (SciPy would sum duplicate coordinates).
    matrix = sp.csr_array((np.ones(len(c), dtype=np.int64), (r, c)), shape=(len(rows), n))
    product = (matrix.T @ matrix).tocoo()

    commits = np.bincount(c, minlength=n).astype(np.int64)
    has_test = np.array([sha in tests_by_sha for sha in rows], dtype=np.bool_)
    test_commits = np.bincount(c[has_test[r]], minlength=n).astype(np.int64)
    for col, mine in own_tests.items():  # a file's own test-named history is not another test file
        test_commits[col] = sum(
            1 for k in r[c == col].tolist() if not tests_by_sha.get(rows[k], set()) <= mine
        )

    src = np.asarray(product.row, dtype=np.int64)
    dst = np.asarray(product.col, dtype=np.int64)
    support = np.asarray(product.data, dtype=np.int64)
    keep = (src != dst) & (support >= min_support) & ~np.isin(src * n + dst, conflict_keys)
    src, dst, support = src[keep], dst[keep], support[keep]
    order = np.lexsort((dst, src))
    src, dst, support = src[order], dst[order], support[order]
    conf = support / commits[src]  # commits[src] >= support >= 1
    # Compare the quotient itself (never min_conf * commits) so 7 / 10 >= 0.7 holds exactly.
    strong = conf >= min_conf

    logger.debug(
        "co-change: %d history commits, %d touch %d ranked files, %d candidate pairs, %d strong",
        len(selected),
        len(rows),
        n,
        len(support),
        int(strong.sum()),
    )
    return _Pairs(cols, commits, test_commits, src, dst, support, conf, strong)


def _history_shas(commits: pa.Table, cutoff: int | None, reachable: AbstractSet[str] | None) -> set[str]:
    """Shas of the non-bulk commits in ``reachable``, or authored at or before ``cutoff`` without it."""
    shas: list[str] = commits.column("sha").to_pylist()
    bulk: list[bool | None] = commits.column("is_bulk").to_pylist()
    if reachable is not None:
        return {sha for sha, is_bulk in zip(shas, bulk, strict=True) if not is_bulk and sha in reachable}
    assert cutoff is not None  # set by the caller whenever reachable is None
    times: list[int] = commits.column("author_time").cast(pa.int64()).to_pylist()
    return {
        sha
        for sha, is_bulk, t in zip(shas, bulk, times, strict=True)
        if not is_bulk and t is not None and t <= cutoff
    }


def _unix_seconds(t: datetime) -> int:
    """Whole seconds since the epoch, floored so ``author_time <= t`` stays exact. Rejects naive datetimes."""
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"snapshot.time must be timezone-aware: {t!r}")
    return math.floor(t.timestamp())


def _empty_pairs() -> _Pairs:
    ints = np.zeros(0, dtype=np.int64)
    floats = np.zeros(0, dtype=np.float64)
    return _Pairs([], ints, ints, ints, ints, ints, floats, np.zeros(0, dtype=np.bool_))

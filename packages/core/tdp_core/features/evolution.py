"""Evolution (family E) and ownership (family O) features, plus monthly commit activity per file.

Every function here shares these definitions:

- **T** is ``snapshot.time`` (timezone-aware; a naive time raises ``ValueError``). ``snapshot.sha``,
  ``snapshot.repo`` and ``snapshot.index`` are not used.
- **History up to T**: with ``reachable=None``, the commits of ``hist.commits`` whose ``author_time <= T``.
  With a ``reachable`` set of shas (the non-merge commits reachable from the repository state at T, see the
  leakage-safe snapshot rule), exactly the commits whose sha is in that set, with no time filter. Shas in
  ``reachable`` that are missing from ``hist`` are ignored.
- **Counted commits** are the history commits with ``is_bulk == False``. The flag comes from the mined
  ``is_bulk`` column; ``settings.bulk_commit_files`` is applied by ``mine_history`` and not again here.
- **Touching**: the names of a file ``p`` are ``lineage.get(p, frozenset()) | {p}``. A commit touches ``p``
  when any of its ``file_changes`` rows has ``path`` or ``old_path`` among those names. It counts once per
  file, and its churn for that file is the sum of ``added + deleted`` over the matching rows (a rename row
  whose two names both match counts once).
- **Age** of a commit, in days: ``max(T - author_time, 0) / 86400``. Clipping at zero only matters in
  ``reachable`` mode, where a reachable commit can carry an author time after T.

Feature frames are indexed by the sorted, de-duplicated ``paths`` (index name ``"path"``), have no NaN, and
use int64 for counts and flags and float64 for everything else (the assembly step casts to float32).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import FEATURE_FAMILIES, Snapshot
from tdp_core.mining.gitlog import MinedHistory

logger = logging.getLogger(__name__)

__all__ = [
    "EVOLUTION_COLUMNS",
    "OWNERSHIP_COLUMNS",
    "TOUCH_COLUMNS",
    "commit_activity",
    "evolution_features",
    "file_touches",
    "ownership_features",
]

# rel_churn_decayed and cc_trend need static metrics, so feature assembly adds them.
EVOLUTION_COLUMNS: tuple[str, ...] = tuple(
    c for c in FEATURE_FAMILIES["E"] if c not in ("rel_churn_decayed", "cc_trend")
)
OWNERSHIP_COLUMNS: tuple[str, ...] = FEATURE_FAMILIES["O"]
TOUCH_COLUMNS: tuple[str, ...] = (
    "path",
    "sha",
    "author_email",
    "author_time",
    "age_days",
    "churn",
    "is_bulk",
)

_INT_COLUMNS = frozenset(
    {
        "commits_total",
        "commits_90d",
        "commits_180d",
        "churn_total",
        "is_new",
        "n_authors",
        "minor_authors",
        "owner_active",
    }
)
_DAY_S = 86_400.0

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class _History:
    """The commits of the history up to T, sorted by author time, ties by sha."""

    t: float  # T in unix seconds
    sha: pa.Array  # string
    author_time: IntArray  # unix seconds, raw
    is_bulk: BoolArray
    email_code: IntArray  # index into ``emails``
    emails: npt.NDArray[np.object_]  # unique emails, sorted ascending

    @property
    def n(self) -> int:
        return len(self.author_time)

    @property
    def age_days(self) -> FloatArray:
        return np.asarray(np.maximum(self.t - self.author_time, 0.0) / _DAY_S, dtype=np.float64)

    def span_days(self) -> float:
        """Days from the earliest history commit (bulk included) to T, clipped at 0; 0.0 for no history."""
        if self.n == 0:
            return 0.0
        return max(self.t - float(self.author_time.min()), 0.0) / _DAY_S


@dataclass(frozen=True)
class _Touches:
    """One entry per (file, touching commit), sorted by file, then author time, then sha."""

    history: _History
    paths: list[str]  # sorted unique requested paths
    target: IntArray  # index into ``paths``
    commit: IntArray  # index into the history commits
    churn: IntArray  # added + deleted over the matching change rows

    @property
    def age_days(self) -> FloatArray:
        return self.history.age_days[self.commit]

    @property
    def is_bulk(self) -> BoolArray:
        return self.history.is_bulk[self.commit]


def file_touches(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    *,
    reachable: AbstractSet[str] | None = None,
    include_bulk: bool = False,
) -> pd.DataFrame:
    """The touching commits of each file in the history up to T (see the module docstring).

    Returns one row per (current path, commit) pair with the columns of :data:`TOUCH_COLUMNS`: ``path``
    (str, the requested path, not the historical name), ``sha`` (str), ``author_email`` (str),
    ``author_time`` (int64, raw unix seconds, so it can be later than T in ``reachable`` mode), ``age_days``
    (float64, days before T clipped at 0), ``churn`` (int64, lines added plus deleted under any of the file's
    names in that commit) and ``is_bulk`` (bool). Bulk commits are left out unless ``include_bulk`` is true.
    Rows are sorted by path, then author time, then sha, with a ``RangeIndex``. Paths are de-duplicated.

    Raises ``ValueError`` if ``snapshot.time`` is naive.
    """
    touches = _touch_table(hist, snapshot, lineage, paths, reachable)
    keep = np.ones(len(touches.target), dtype=np.bool_) if include_bulk else ~touches.is_bulk
    h = touches.history
    commit = touches.commit[keep]
    path_values = np.asarray(touches.paths, dtype=object)[touches.target[keep]]
    sha_values = np.asarray(h.sha.to_numpy(zero_copy_only=False), dtype=object)[commit]
    frame = pd.DataFrame(
        {
            "path": pd.Series(path_values, dtype="str"),
            "sha": pd.Series(sha_values, dtype="str"),
            "author_email": pd.Series(h.emails[h.email_code[commit]], dtype="str"),
            "author_time": pd.Series(h.author_time[commit], dtype=np.int64),
            "age_days": pd.Series(touches.age_days[keep], dtype=np.float64),
            "churn": pd.Series(touches.churn[keep], dtype=np.int64),
            "is_bulk": pd.Series(touches.is_bulk[keep], dtype=np.bool_),
        }
    )
    return frame


def evolution_features(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    settings: Settings = DEFAULT_SETTINGS,
    *,
    reachable: AbstractSet[str] | None = None,
) -> pd.DataFrame:
    """Evolution features (family E without ``rel_churn_decayed`` and ``cc_trend``) for ``paths`` at T.

    History, counted commits, touching and ages are defined in the module docstring. Columns, in
    :data:`EVOLUTION_COLUMNS` order, over each file's counted touching commits unless stated otherwise:

    - ``commits_total`` (int64): number of counted touching commits.
    - ``commits_90d`` / ``commits_180d`` (int64): those with age ``<= settings.recent_days[0]`` /
      ``[1]`` days (inclusive; the names refer to the default windows of 90 and 180 days).
    - ``commits_decayed`` (float64): ``sum(2 ** (-age / settings.half_life_days))``, so a commit exactly one
      half-life old weighs 0.5 (equal to ``exp(-ln 2 / half_life * age)``).
    - ``churn_total`` (int64): lines added plus deleted; ``churn_decayed`` (float64): churn times the weight.
    - ``age_days`` (float64): age of the oldest touching commit, **bulk commits included**, because a bulk
      commit that touches the file proves the file existed then. A file with no touching commit at all gets
      the observed history span (days from the earliest history commit to T, clipped at 0), a lower bound
      on its age because it exists at T but was created before the mined history; 0.0 without history.
    - ``staleness_days`` (float64): age of the newest counted touching commit, or ``age_days`` when there
      is none.
    - ``burstiness`` (float64): coefficient of variation (population std / mean) of the gaps in days between
      consecutive counted touching commits (author times clipped at T, ties ordered by sha); 0.0 with fewer
      than 3 commits or when the mean gap is 0.
    - ``is_new`` (int64): 1 if ``age_days < settings.new_file_days`` (strict), else 0.

    ``paths`` are sorted and de-duplicated into the index (name ``"path"``); empty ``paths`` give an empty
    frame with the same columns and dtypes. Raises ``ValueError`` if ``snapshot.time`` is naive or
    ``settings.half_life_days <= 0``.
    """
    if settings.half_life_days <= 0:
        raise ValueError(f"half_life_days must be > 0, got {settings.half_life_days}")
    touches = _touch_table(hist, snapshot, lineage, paths, reachable)
    n = len(touches.paths)
    age_all = touches.age_days
    counted = ~touches.is_bulk
    target = touches.target[counted]
    age = age_all[counted]
    churn = touches.churn[counted]
    weight = np.exp2(-age / settings.half_life_days)

    churn_total = np.zeros(n, dtype=np.int64)
    np.add.at(churn_total, target, churn)

    oldest = np.full(n, np.nan)
    np.fmax.at(oldest, touches.target, age_all)
    age_days = np.where(np.isnan(oldest), touches.history.span_days(), oldest)
    newest = np.full(n, np.nan)
    np.fmin.at(newest, target, age)
    staleness_days = np.where(np.isnan(newest), age_days, newest)

    clipped = np.minimum(touches.history.author_time[touches.commit[counted]], touches.history.t)
    columns: dict[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]] = {
        "commits_total": _count(target, n),
        "commits_90d": _count(target[age <= settings.recent_days[0]], n),
        "commits_180d": _count(target[age <= settings.recent_days[1]], n),
        "commits_decayed": np.bincount(target, weights=weight, minlength=n).astype(np.float64),
        "churn_total": churn_total,
        "churn_decayed": np.bincount(target, weights=churn * weight, minlength=n).astype(np.float64),
        "age_days": age_days.astype(np.float64),
        "staleness_days": staleness_days.astype(np.float64),
        "burstiness": _burstiness(target, clipped, n),
        "is_new": (age_days < settings.new_file_days).astype(np.int64),
    }
    return _frame(touches.paths, EVOLUTION_COLUMNS, columns)


def ownership_features(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    settings: Settings = DEFAULT_SETTINGS,
    *,
    reachable: AbstractSet[str] | None = None,
) -> pd.DataFrame:
    """Ownership features (family O) for ``paths`` at T, over each file's counted touching commits.

    History, counted commits and touching are defined in the module docstring; authors are identified by
    (lowercased, mailmapped) email. With ``p_a`` the share of the file's counted touching commits made by
    author ``a``, the columns in :data:`OWNERSHIP_COLUMNS` order are:

    - ``n_authors`` (int64): number of distinct authors.
    - ``owner_share`` (float64, 0 to 1): ``max p_a``.
    - ``minor_authors`` (int64): authors with ``p_a < settings.minor_author_share`` (strict).
    - ``author_entropy`` (float64, bits): ``-sum(p_a * log2 p_a)``; 0.0 for one author, 1.0 for two equal.
    - ``owner_active`` (int64): 1 if the owner (most counted touching commits; ties go to the author whose
      latest touching commit to this file is most recent, then to the smaller email) made any counted
      commit anywhere in the repository with age ``<= settings.owner_active_days`` (inclusive), else 0.

    A file without counted touching commits gets 0 in every column. ``paths`` are sorted and de-duplicated
    into the index (name ``"path"``); empty ``paths`` give an empty frame with the same columns and dtypes.
    Raises ``ValueError`` if ``snapshot.time`` is naive.
    """
    touches = _touch_table(hist, snapshot, lineage, paths, reachable)
    h = touches.history
    n = len(touches.paths)
    counted = ~touches.is_bulk
    target = touches.target[counted]
    age = touches.age_days[counted]
    n_emails = max(len(h.emails), 1)
    email = h.email_code[touches.commit[counted]]

    group_key, inverse, n_a = np.unique(target * n_emails + email, return_inverse=True, return_counts=True)
    g_target = group_key // n_emails
    g_email = group_key % n_emails
    g_newest = np.full(len(group_key), np.inf)
    np.minimum.at(g_newest, inverse, age)

    total = _count(target, n)
    share = n_a / np.maximum(total[g_target], 1)
    owner_share = np.zeros(n, dtype=np.float64)
    np.maximum.at(owner_share, g_target, share)
    entropy = np.bincount(g_target, weights=-share * np.log2(share), minlength=n).astype(np.float64) + 0.0

    owner_active = np.zeros(n, dtype=np.int64)
    if len(group_key):
        order = np.lexsort((g_email, g_newest, -n_a, g_target))
        sorted_target = g_target[order]
        first = order[np.concatenate(([True], sorted_target[1:] != sorted_target[:-1]))]
        last_commit_age = np.full(n_emails, np.inf)
        commits_counted = ~h.is_bulk
        np.minimum.at(last_commit_age, h.email_code[commits_counted], h.age_days[commits_counted])
        active = last_commit_age[g_email[first]] <= settings.owner_active_days
        owner_active[g_target[first]] = active.astype(np.int64)

    columns: dict[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]] = {
        "n_authors": _count(g_target, n),
        "owner_share": owner_share,
        "minor_authors": _count(g_target[share < settings.minor_author_share], n),
        "author_entropy": entropy,
        "owner_active": owner_active,
    }
    return _frame(touches.paths, OWNERSHIP_COLUMNS, columns)


def commit_activity(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    months: int = 24,
    *,
    reachable: AbstractSet[str] | None = None,
) -> dict[str, tuple[tuple[str, int], ...]]:
    """Counted touching commits per calendar month (UTC) for the ``months`` months ending with T's month.

    Uses the same commits as ``commits_total`` (history up to T, bulk excluded, lineage followed). A commit
    authored after T (``reachable`` mode only) counts in T's month. Returns ``{path: (("YYYY-MM", count),
    ...)}`` with exactly ``months`` pairs per path, oldest month first, zero-filled, counts as Python ints;
    keys are the sorted, de-duplicated ``paths``. Empty ``paths`` give ``{}``.

    Raises ``ValueError`` if ``snapshot.time`` is naive or ``months < 1``.
    """
    if months < 1:
        raise ValueError(f"months must be >= 1, got {months}")
    touches = _touch_table(hist, snapshot, lineage, paths, reachable)
    h = touches.history
    n = len(touches.paths)
    counted = ~touches.is_bulk
    target = touches.target[counted]

    t_floor = math.floor(h.t)
    commit_month = _month_ids(np.minimum(h.author_time, t_floor))
    last_month = int(_month_ids(np.asarray([t_floor], dtype=np.int64))[0])
    first_month = last_month - months + 1
    month = commit_month[touches.commit[counted]]
    inside = (month >= first_month) & (month <= last_month)
    slot = target[inside] * months + (month[inside] - first_month)
    counts = np.bincount(slot, minlength=n * months).reshape(n, months)

    labels = [str(np.datetime64(first_month + k, "M")) for k in range(months)]
    return {
        path: tuple(zip(labels, (int(c) for c in row), strict=True))
        for path, row in zip(touches.paths, counts, strict=True)
    }


# ---------------------------------------------------------------------------------------------- internals
def _snapshot_seconds(snapshot: Snapshot) -> float:
    if snapshot.time.utcoffset() is None:
        raise ValueError(f"snapshot time must be timezone-aware: {snapshot.time!r}")
    return snapshot.time.timestamp()


def _history(hist: MinedHistory, snapshot: Snapshot, reachable: AbstractSet[str] | None) -> _History:
    t = _snapshot_seconds(snapshot)
    table = hist.commits.select(["sha", "author_email", "author_time", "is_bulk"])
    times = np.asarray(pc.cast(table.column("author_time"), pa.int64()).to_numpy(), dtype=np.int64)
    if reachable is None:
        keep = times <= t
    else:
        value_set = pa.array(list(reachable), type=pa.string())
        keep = np.asarray(pc.is_in(table.column("sha"), value_set=value_set).to_numpy(), dtype=np.bool_)
    table = table.filter(pa.array(keep)).sort_by([("author_time", "ascending"), ("sha", "ascending")])

    emails_raw = pc.fill_null(table.column("author_email"), "").to_numpy(zero_copy_only=False)
    codes, uniques = pd.factorize(np.asarray(emails_raw, dtype=object), sort=True)
    return _History(
        t=t,
        sha=table.column("sha").combine_chunks(),
        author_time=np.asarray(pc.cast(table.column("author_time"), pa.int64()).to_numpy(), dtype=np.int64),
        is_bulk=np.asarray(pc.fill_null(table.column("is_bulk"), False).to_numpy(), dtype=np.bool_),
        email_code=np.asarray(codes, dtype=np.int64),
        emails=np.asarray(uniques, dtype=object),
    )


def _touch_table(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    reachable: AbstractSet[str] | None,
) -> _Touches:
    """Match change rows to the requested files with integer codes (no Python loop over commits)."""
    history = _history(hist, snapshot, reachable)
    targets = sorted(set(paths))
    n = len(targets)
    empty = np.zeros(0, dtype=np.int64)
    if n == 0 or history.n == 0:
        return _Touches(history, targets, empty, empty, empty)

    # (name, target) pairs, one per historical name of each requested file.
    name_code: dict[str, int] = {}
    pair_name: list[int] = []
    pair_target: list[int] = []
    for i, path in enumerate(targets):
        for name in sorted(lineage.get(path, frozenset()) | {path}):
            pair_name.append(name_code.setdefault(name, len(name_code)))
            pair_target.append(i)
    pair_order = np.argsort(np.asarray(pair_name, dtype=np.int64), kind="stable")
    pair_name_sorted = np.asarray(pair_name, dtype=np.int64)[pair_order]
    pair_target_sorted = np.asarray(pair_target, dtype=np.int64)[pair_order]

    changes = hist.file_changes
    names = pa.array(list(name_code), type=pa.string())
    row_commit = _index_in(changes.column("sha"), history.sha)
    row_path = _index_in(changes.column("path"), names)
    row_old = _index_in(changes.column("old_path"), names)
    row_lines = _int_column(changes, "added") + _int_column(changes, "deleted")

    # Long form: one (row, name) entry per matching path and old_path of rows in the history.
    in_history = row_commit >= 0
    by_path = np.flatnonzero(in_history & (row_path >= 0))
    by_old = np.flatnonzero(in_history & (row_old >= 0))
    long_row = np.concatenate((by_path, by_old)).astype(np.int64)
    long_name = np.concatenate((row_path[by_path], row_old[by_old])).astype(np.int64)

    # Expand each entry to every file that has this name (a name can belong to several files).
    start = np.searchsorted(pair_name_sorted, long_name, side="left")
    stop = np.searchsorted(pair_name_sorted, long_name, side="right")
    width = (stop - start).astype(np.int64)
    entry_row = np.repeat(long_row, width)
    offset = np.arange(int(width.sum()), dtype=np.int64) - np.repeat(np.cumsum(width) - width, width)
    entry_target = pair_target_sorted[np.repeat(start, width) + offset]

    # A change row counts once per file even when both its names match.
    row_target = np.unique(entry_row * n + entry_target)
    rows = row_target // n
    files = row_target % n

    # One touch per (file, commit); commits are already in (author_time, sha) order.
    n_commits = history.n
    touch_key, inverse = np.unique(files * n_commits + row_commit[rows], return_inverse=True)
    churn = np.zeros(len(touch_key), dtype=np.int64)
    np.add.at(churn, inverse, row_lines[rows])
    logger.debug(
        "touch table: %d files, %d history commits, %d change rows, %d touches",
        n,
        n_commits,
        len(row_lines),
        len(touch_key),
    )
    return _Touches(
        history=history,
        paths=targets,
        target=(touch_key // n_commits).astype(np.int64),
        commit=(touch_key % n_commits).astype(np.int64),
        churn=churn,
    )


def _index_in(values: pa.ChunkedArray, value_set: pa.Array) -> IntArray:
    """Position of each value in ``value_set``, or -1 when absent or null."""
    found = pc.index_in(values, value_set=value_set)
    return np.asarray(pc.fill_null(found, -1).to_numpy(), dtype=np.int64)


def _int_column(table: pa.Table, name: str) -> IntArray:
    """An integer column as int64, nulls as 0."""
    return np.asarray(pc.fill_null(pc.cast(table.column(name), pa.int64()), 0).to_numpy(), dtype=np.int64)


def _count(codes: IntArray, n: int) -> IntArray:
    return np.bincount(codes, minlength=n).astype(np.int64)


def _burstiness(target: IntArray, clipped_time: IntArray | FloatArray, n: int) -> FloatArray:
    """Population coefficient of variation of the gaps between consecutive commits of each file.

    ``target`` must be grouped by file and ordered in time within each file.
    """
    same = target[1:] == target[:-1]
    gaps = (np.diff(clipped_time.astype(np.float64)) / _DAY_S)[same]
    gap_file = target[1:][same]
    n_gaps = np.bincount(gap_file, minlength=n)
    mean = np.bincount(gap_file, weights=gaps, minlength=n) / np.maximum(n_gaps, 1)
    var = np.bincount(gap_file, weights=(gaps - mean[gap_file]) ** 2, minlength=n) / np.maximum(n_gaps, 1)
    ok = (n_gaps >= 2) & (mean > 0)
    out = np.zeros(n, dtype=np.float64)
    out[ok] = np.sqrt(var[ok]) / mean[ok]
    return out


def _month_ids(unix_seconds: IntArray) -> IntArray:
    """Calendar month (UTC) of each time, as months since 1970-01."""
    months = unix_seconds.astype("datetime64[s]").astype("datetime64[M]")
    return np.asarray(months.astype(np.int64), dtype=np.int64)


def _frame(
    paths: list[str],
    order: tuple[str, ...],
    columns: Mapping[str, npt.NDArray[np.int64] | npt.NDArray[np.float64]],
) -> pd.DataFrame:
    index = pd.Index(paths, dtype="str", name="path")
    data = {
        col: np.asarray(columns[col], dtype=np.int64 if col in _INT_COLUMNS else np.float64) for col in order
    }
    return pd.DataFrame(data, index=index)

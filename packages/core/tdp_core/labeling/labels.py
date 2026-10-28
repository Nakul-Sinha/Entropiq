"""Training labels (latent defects, future churn, pain, grade) and the defect-history features (family D).

Shared definitions:

- **T** is ``snapshot.time`` (timezone-aware; a naive time raises ``ValueError``), compared in whole seconds.
  In time mode the label window ends at ``end = T + settings.horizon_days`` days.
- **Modes.** With both reachable sets None the functions use the **time mode** (the plan's original rules).
  With ``reachable_T`` (non-merge commits reachable from the state at T) and ``reachable_end`` (the same at
  T + H, see :func:`reachable_window`) they use the **reachable mode** of the leakage-safe snapshot rule.
  Giving exactly one of them raises ``ValueError``, and ``reachable_T`` must be a subset of
  ``reachable_end``.
- **Prior history P**: commits whose sha is in ``reachable_T`` (no time filter), or ``author_time <= T`` in
  time mode.
- **Window W**: ``reachable_end - reachable_T`` with no time filter (commits authored before T but merged
  after T belong here), or ``T < author_time <= end`` in time mode.
- **Commit identity**: commits with the same ``(author_email, author_time, subject)`` are copies of one
  change (cherry-picks and rebases keep these). A sha missing from ``hist`` is its own identity.
  ``prior_ids`` are the identities of the ``hist`` commits in P.
- **Counted window commits W\\***: non-bulk ``hist`` commits of W whose identity is not in ``prior_ids``,
  one per identity (the smallest sha). A window copy of a prior commit is not new work.
- An inducing commit is **before T** when its sha is in ``reachable_T`` (time mode: ``inducing_time <= T``)
  or its identity is in ``prior_ids`` (a copy of a prior commit).
- A fix is **in the window** when its sha is in W (time mode: ``T < fix_time <= end``) and its identity is
  not in ``prior_ids``.
- **Window names** of a ranked file f: ``start(f)`` is f plus the names in ``lineage_at_T[f]`` that no file
  alive at T uses (alive = the keys of ``lineage_at_T`` plus ``paths``), so a name another live file holds
  now never counts for f, while a stale branch that still edits a pre-T name does. ``names_W(f)`` is the
  closure of ``start(f)`` under the rename edges ``old_path -> path`` of every change row of every W commit
  (bulk commits included, so a move of the whole tree is followed).
- **Prior names** (features only, the touching rule of the evolution features):
  ``names_P(f) = lineage_at_T.get(f, frozenset()) | {f}``.

Labels of a ranked file f:

- ``latent_defects``: distinct identities of window fixes with at least one ``szz`` row whose ``path`` is in
  ``names_W(f)`` and whose inducing commit is before T.
- ``future_churn``: lines added plus deleted by the W\\* commits on ``names_W(f)``, each commit counted once
  per file (matching ``path`` or ``old_path``, as in the evolution features).
- ``pain = latent_defects + gamma * log1p(future_churn / max(median future_churn, 1))``, the median taken over
  the files of this query, and ``grade`` from :func:`grades_from_pain`.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import FEATURE_FAMILIES, INDEX_COLUMNS, LABEL_COLUMNS, Snapshot
from tdp_core.features.evolution import file_touches
from tdp_core.mining.gitlog import MinedHistory, _unix_seconds
from tdp_core.mining.reach import reachable_shas, snapshot_sha

logger = logging.getLogger(__name__)

__all__ = [
    "DEFECT_HISTORY_COLUMNS",
    "GRADE_CUTS",
    "LABEL_FRAME_COLUMNS",
    "defect_history_features",
    "grades_from_pain",
    "index_frame",
    "make_labels",
    "pain_from_counts",
    "reachable_window",
]

LABEL_FRAME_COLUMNS: tuple[str, ...] = INDEX_COLUMNS + LABEL_COLUMNS
DEFECT_HISTORY_COLUMNS: tuple[str, ...] = FEATURE_FAMILIES["D"]
GRADE_CUTS: tuple[float, float, float] = (0.50, 0.80, 0.95)

_SZZ_COLUMNS = ("fix_sha", "fix_time", "path", "inducing_sha", "inducing_time")
_DAY_S = 86_400
_INT16 = np.iinfo(np.int16)


class _SzzRow(NamedTuple):
    """One usable szz row (no null field); times in unix seconds."""

    fix_sha: str
    fix_time: int
    path: str
    inducing_sha: str
    inducing_time: int


@dataclass(frozen=True)
class _Commits:
    """The commits of ``hist`` as Python lists, with each sha's identity (the smallest sha of its group)."""

    sha: list[str]
    time: list[int]
    bulk: list[bool]
    identity: dict[str, str]

    def ident(self, sha: str) -> str:
        return self.identity.get(sha, sha)


@dataclass(frozen=True)
class _Periods:
    """Membership tests for the prior history P and the window W (see the module docstring)."""

    t: int
    end: int
    reachable_t: AbstractSet[str] | None
    reachable_end: AbstractSet[str] | None

    def prior(self, sha: str, when: int) -> bool:
        if self.reachable_t is None:
            return when <= self.t
        return sha in self.reachable_t

    def window(self, sha: str, when: int) -> bool:
        if self.reachable_t is None or self.reachable_end is None:
            return self.t < when <= self.end
        return sha in self.reachable_end and sha not in self.reachable_t


# ------------------------------------------------------------------------------------------------ helpers
def _snapshot_seconds(snapshot: Snapshot) -> int:
    if snapshot.time.tzinfo is None or snapshot.time.utcoffset() is None:
        raise ValueError(f"snapshot time must be timezone-aware: {snapshot.time!r}")
    return _unix_seconds(snapshot.time)


def _sorted_paths(paths: Iterable[str]) -> list[str]:
    return sorted({str(p) for p in paths})


def index_frame(snapshot: Snapshot, paths: Sequence[str]) -> pd.DataFrame:
    """The :data:`~tdp_core.contracts.INDEX_COLUMNS` of a query: one row per sorted unique path.

    Columns and dtypes: ``repo`` (``"str"``, ``snapshot.repo``), ``snapshot_idx`` (int16, ``snapshot.index``),
    ``snapshot_time`` (``datetime64[us, UTC]``, ``snapshot.time`` converted to UTC) and ``path`` (``"str"``),
    with a ``RangeIndex``. Feature and label frames built with this helper merge on ``INDEX_COLUMNS`` with
    identical dtypes.

    Raises ``ValueError`` for a naive ``snapshot.time`` or an index outside the int16 range.
    """
    _snapshot_seconds(snapshot)
    if not _INT16.min <= snapshot.index <= _INT16.max:
        raise ValueError(f"snapshot index {snapshot.index} does not fit in int16")
    targets = _sorted_paths(paths)
    n = len(targets)
    when = pd.Timestamp(snapshot.time).tz_convert("UTC")
    return pd.DataFrame(
        {
            "repo": pd.Series([snapshot.repo] * n, dtype="str"),
            "snapshot_idx": pd.Series(np.full(n, snapshot.index, dtype=np.int16)),
            "snapshot_time": pd.Series([when] * n, dtype="datetime64[us, UTC]"),
            "path": pd.Series(targets, dtype="str"),
        }
    )


def reachable_window(
    repo_dir: Path, snapshot: Snapshot, settings: Settings = DEFAULT_SETTINGS
) -> tuple[frozenset[str], frozenset[str]]:
    """``(reachable_T, reachable_end)`` for a snapshot made with ``make_snapshots(..., repo_dir=...)``.

    ``reachable_T`` holds the non-merge commits reachable from ``snapshot.sha`` (the state at T) and
    ``reachable_end`` those reachable from the state at ``T + settings.horizon_days`` days, both on the
    default branch (the branch ``HEAD`` of ``repo_dir`` points at). The sets are nested.

    Raises ``ValueError`` for a naive ``snapshot.time`` or when no state exists at T + H (a snapshot older
    than the repository), and ``GitError`` when git fails.
    """
    _snapshot_seconds(snapshot)
    end_sha = snapshot_sha(repo_dir, snapshot.time + timedelta(days=settings.horizon_days))
    if end_sha is None:
        raise ValueError(f"no state of {repo_dir} exists at {snapshot.time.isoformat()} plus the horizon")
    return reachable_shas(repo_dir, snapshot.sha), reachable_shas(repo_dir, end_sha)


def pain_from_counts(
    latent_defects: npt.ArrayLike, future_churn: npt.ArrayLike, gamma: float
) -> npt.NDArray[np.float32]:
    """``latent + gamma * log1p(churn / max(median(churn), 1))`` per file, as float32.

    Both inputs are 1-D, of equal length and non-negative (counts of one query); the median is over
    ``future_churn`` (0 for an empty query). ``gamma`` must be finite and ``>= 0``. Raises ``ValueError``
    otherwise.
    """
    latent = np.asarray(latent_defects, dtype=np.float64)
    churn = np.asarray(future_churn, dtype=np.float64)
    if latent.ndim != 1 or churn.ndim != 1 or latent.shape != churn.shape:
        raise ValueError(f"latent_defects {latent.shape} and future_churn {churn.shape} must be equal 1-D")
    if not (np.all(np.isfinite(latent)) and np.all(np.isfinite(churn))):
        raise ValueError("latent_defects and future_churn must be finite")
    if np.any(latent < 0) or np.any(churn < 0):
        raise ValueError("latent_defects and future_churn must be non-negative")
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError(f"gamma must be finite and >= 0, got {gamma}")
    median = float(np.median(churn)) if churn.size else 0.0
    pain = latent + gamma * np.log1p(churn / max(median, 1.0))
    return np.asarray(pain, dtype=np.float32)


def grades_from_pain(pain: npt.ArrayLike) -> npt.NDArray[np.int8]:
    """Grades 0 to 4 from the pains of one query.

    Grade 0 where pain is 0. The non-zero pains are ranked (``rank(pct=True, method="average")``) and the
    percentile ``p`` gives grade 1 for ``p <= 0.50``, 2 for ``p <= 0.80``, 3 for ``p <= 0.95`` and 4 above
    (:data:`GRADE_CUTS`). The input must be 1-D, finite and non-negative (``ValueError`` otherwise).
    """
    values = np.asarray(pain, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"pain must be 1-D, got shape {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("pain must be finite and non-negative")
    grades = np.zeros(values.shape, dtype=np.int8)
    nonzero = values > 0
    if nonzero.any():
        pct = pd.Series(values[nonzero]).rank(pct=True, method="average").to_numpy(dtype=np.float64)
        low, mid, high = GRADE_CUTS
        grades[nonzero] = np.where(pct <= low, 1, np.where(pct <= mid, 2, np.where(pct <= high, 3, 4)))
    return grades


def _periods(
    t: int,
    horizon_days: int,
    reachable_t: AbstractSet[str] | None,
    reachable_end: AbstractSet[str] | None,
) -> _Periods:
    if (reachable_t is None) != (reachable_end is None):
        raise ValueError("pass both reachable_T and reachable_end, or neither")
    if reachable_t is not None and reachable_end is not None and not reachable_t <= reachable_end:
        raise ValueError("reachable_T must be a subset of reachable_end")
    return _Periods(t, t + horizon_days * _DAY_S, reachable_t, reachable_end)


def _require(table: pa.Table, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in table.column_names]
    if missing:
        raise ValueError(f"{what} table lacks columns {missing}")


def _seconds(column: pa.ChunkedArray) -> list[int | None]:
    """A timestamp column of any unit as whole unix seconds (nulls kept)."""
    as_seconds = pc.cast(column, pa.timestamp("s", tz="UTC"), safe=False)
    values: list[int | None] = pc.cast(as_seconds, pa.int64()).to_pylist()
    return values


def _szz_rows(szz: pa.Table) -> list[_SzzRow]:
    _require(szz, _SZZ_COLUMNS, "szz")
    rows: list[_SzzRow] = []
    for fix_sha, fix_time, path, inducing_sha, inducing_time in zip(
        szz.column("fix_sha").to_pylist(),
        _seconds(szz.column("fix_time")),
        szz.column("path").to_pylist(),
        szz.column("inducing_sha").to_pylist(),
        _seconds(szz.column("inducing_time")),
        strict=True,
    ):
        if fix_time is None or inducing_time is None or None in (fix_sha, path, inducing_sha):
            continue
        rows.append(_SzzRow(fix_sha, fix_time, path, inducing_sha, inducing_time))
    return rows


def _commits(hist: MinedHistory) -> _Commits:
    table = hist.commits
    shas: list[str] = table.column("sha").to_pylist()
    times = [int(t) for t in pc.fill_null(pc.cast(table.column("author_time"), pa.int64()), 0).to_pylist()]
    bulk = [bool(b) for b in pc.fill_null(table.column("is_bulk"), False).to_pylist()]
    emails = table.column("author_email").to_pylist()
    subjects = table.column("subject").to_pylist()
    smallest: dict[tuple[object, int, object], str] = {}
    keys = list(zip(emails, times, subjects, strict=True))
    for sha, key in zip(shas, keys, strict=True):
        current = smallest.get(key)
        if current is None or sha < current:
            smallest[key] = sha
    identity = {sha: smallest[key] for sha, key in zip(shas, keys, strict=True)}
    return _Commits(sha=shas, time=times, bulk=bulk, identity=identity)


def _owners(names: Mapping[str, AbstractSet[str]], targets: Sequence[str]) -> dict[str, list[int]]:
    """Reverse index: name -> indexes of the ranked files that hold it."""
    owners: dict[str, list[int]] = defaultdict(list)
    for i, path in enumerate(targets):
        for name in names[path]:
            owners[name].append(i)
    return owners


def _window_renames(file_changes: pa.Table, window: AbstractSet[str]) -> list[tuple[str, str]]:
    """``(old_path, path)`` of every rename row of a window commit."""
    if not window or file_changes.num_rows == 0:
        return []
    in_window = pc.is_in(file_changes.column("sha"), value_set=pa.array(sorted(window), type=pa.string()))
    mask = pc.and_(pc.fill_null(in_window, False), pc.is_valid(file_changes.column("old_path")))
    renames = file_changes.filter(mask)
    return list(zip(renames.column("old_path").to_pylist(), renames.column("path").to_pylist(), strict=True))


def _window_names(
    targets: Sequence[str],
    lineage_at_t: Mapping[str, AbstractSet[str]],
    renames: Iterable[tuple[str, str]],
) -> dict[str, frozenset[str]]:
    """``names_W(f)`` for every ranked file (see the module docstring)."""
    alive = set(lineage_at_t) | set(targets)
    forward: dict[str, set[str]] = defaultdict(set)
    for old, new in renames:
        if old is not None and new is not None and old != new:
            forward[old].add(new)
    out: dict[str, frozenset[str]] = {}
    for path in targets:
        start = {path} | {name for name in lineage_at_t.get(path, frozenset()) if name not in alive}
        seen = set(start)
        stack = list(start)
        while stack:
            for nxt in forward.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        out[path] = frozenset(seen)
    return out


def _churn_by_path(touches: pd.DataFrame, targets: Sequence[str]) -> npt.NDArray[np.int64]:
    if touches.empty:
        return np.zeros(len(targets), dtype=np.int64)
    churn = touches.groupby("path", sort=False)["churn"].sum()
    return np.asarray(churn.reindex(list(targets), fill_value=0).to_numpy(), dtype=np.int64)


def _label_frame(
    snapshot: Snapshot,
    targets: Sequence[str],
    latent: npt.NDArray[np.int64],
    churn: npt.NDArray[np.int64],
    gamma: float,
) -> pd.DataFrame:
    frame = index_frame(snapshot, targets)
    pain = pain_from_counts(latent, churn, gamma)
    frame["latent_defects"] = latent.astype(np.int32)
    frame["future_churn"] = churn.astype(np.int32)
    frame["pain"] = pain
    frame["grade"] = grades_from_pain(pain)
    frame["has_latent_defect"] = latent > 0
    return frame[list(LABEL_FRAME_COLUMNS)]


# ------------------------------------------------------------------------------------------------ labels
def make_labels(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage_at_T: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    szz: pa.Table,
    settings: Settings = DEFAULT_SETTINGS,
    *,
    reachable_T: AbstractSet[str] | None = None,
    reachable_end: AbstractSet[str] | None = None,
) -> pd.DataFrame:
    """The training labels of one query (see the module docstring for every definition).

    Returns a flat frame with columns exactly :data:`LABEL_FRAME_COLUMNS` and a ``RangeIndex``, one row per
    sorted unique path: the index columns of :func:`index_frame`, then ``latent_defects`` (int32),
    ``future_churn`` (int32), ``pain`` (float32, with ``settings.gamma``), ``grade`` (int8, ranked on the
    float32 pain, so it can be recomputed from the stored column) and ``has_latent_defect`` (bool). Files
    without window activity get zeros and grade 0; empty ``paths`` give an empty frame with the same dtypes.

    ``lineage_at_T`` should be the full ``path_lineage`` output at T (every path alive at T), since its keys
    say which names are alive. ``szz`` needs the columns ``fix_sha``, ``fix_time``, ``path``,
    ``inducing_sha`` and ``inducing_time`` (timestamps of any unit; extra columns are ignored, rows with a
    null field are skipped). Pass ``reachable_T`` and ``reachable_end`` from :func:`reachable_window` for
    the leakage-safe rule, or neither for the time rule.

    Raises ``ValueError`` for a naive ``snapshot.time``, only one reachable set, sets that are not nested,
    or missing ``szz`` columns.
    """
    t = _snapshot_seconds(snapshot)
    periods = _periods(t, settings.horizon_days, reachable_T, reachable_end)
    szz_rows = _szz_rows(szz)
    targets = _sorted_paths(paths)
    empty = np.zeros(len(targets), dtype=np.int64)
    if not targets:
        return _label_frame(snapshot, targets, empty, empty, settings.gamma)

    commits = _commits(hist)
    prior_ids: set[str] = set()
    window_hist: list[int] = []
    for i, (sha, when) in enumerate(zip(commits.sha, commits.time, strict=True)):
        if periods.prior(sha, when):
            prior_ids.add(commits.ident(sha))
        elif periods.window(sha, when):
            window_hist.append(i)

    # W*: one non-bulk window commit per identity that is new after T.
    counted: dict[str, str] = {}
    for i in window_hist:
        sha = commits.sha[i]
        ident = commits.ident(sha)
        if commits.bulk[i] or ident in prior_ids:
            continue
        if ident not in counted or sha < counted[ident]:
            counted[ident] = sha
    window_star = frozenset(counted.values())

    if reachable_T is not None and reachable_end is not None:
        window_all: AbstractSet[str] = frozenset(reachable_end - reachable_T)
    else:
        window_all = frozenset(commits.sha[i] for i in window_hist)
    names = _window_names(targets, lineage_at_T, _window_renames(hist.file_changes, window_all))
    owners = _owners(names, targets)

    fixes: list[set[str]] = [set() for _ in targets]
    for row in szz_rows:
        hit = owners.get(row.path)
        if not hit or not periods.window(row.fix_sha, row.fix_time):
            continue
        fix_id = commits.ident(row.fix_sha)
        if fix_id in prior_ids:
            continue
        inducing_before = periods.prior(row.inducing_sha, row.inducing_time) or (
            commits.ident(row.inducing_sha) in prior_ids
        )
        if not inducing_before:
            continue
        for i in hit:
            fixes[i].add(fix_id)
    latent = np.asarray([len(s) for s in fixes], dtype=np.int64)

    touches = file_touches(hist, snapshot, names, targets, reachable=window_star)
    churn = _churn_by_path(touches, targets)
    logger.debug(
        "labels %s/%d: %d files, %d window commits (%d counted), %d szz rows, %d with latent defects",
        snapshot.repo,
        snapshot.index,
        len(targets),
        len(window_all),
        len(window_star),
        len(szz_rows),
        int(np.count_nonzero(latent)),
    )
    return _label_frame(snapshot, targets, latent, churn, settings.gamma)


# ------------------------------------------------------------------------------------------------ features
def defect_history_features(
    hist: MinedHistory,
    snapshot: Snapshot,
    lineage_at_T: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    bugfix: pa.Table,
    szz: pa.Table,
    *,
    reachable_T: AbstractSet[str] | None = None,
    reachable_end: AbstractSet[str] | None = None,
    max_fixes: int | None = None,
) -> pd.DataFrame:
    """The defect-history features ``prior_bugfix_commits`` and ``prior_latent_bugs`` as of T.

    The prior fixes S are the non-bulk commits of the prior history P (``reachable_T``, or
    ``author_time <= T`` when it is None) that ``bugfix`` marks with ``is_bugfix``. Copies of one change
    (same identity, see the module docstring) count once.

    - ``prior_bugfix_commits``: fixes in S touching the file under any prior name ``names_P(f)``
      (a change row matches on ``path`` or ``old_path``).
    - ``prior_latent_bugs``: fixes in S' with at least one ``szz`` row whose ``path`` is in ``names_P(f)``.
      S' is S, or with ``max_fixes`` the newest ``max_fixes`` fixes of S (author time descending, ties by
      sha), which mirrors ``szz_table(max_fixes=...)`` at inference. ``szz`` rows of fixes that are not in
      S are ignored, so a broader szz table can be combined with a keyword-only bug-fix table.

    Nothing after T is read: the result is the same for any history that agrees on P. ``reachable_end`` is
    accepted so callers can pass the same reachable sets as to :func:`make_labels`; it is only checked
    (it needs ``reachable_T`` and must contain it) and never read, because a feature never sees the label
    window. ``reachable_T`` alone is enough. Returns a frame indexed by the sorted unique paths
    (``pd.Index``, dtype ``"str"``, name ``"path"``) with columns :data:`DEFECT_HISTORY_COLUMNS` as int64;
    empty ``paths`` give an empty frame with the same columns and dtypes.

    Raises ``ValueError`` for a naive ``snapshot.time``, a negative ``max_fixes``, ``reachable_end`` without
    ``reachable_T`` or not containing it, a ``bugfix`` table without ``sha`` and ``is_bugfix``, or an
    ``szz`` table without the columns that :func:`make_labels` needs.
    """
    t = _snapshot_seconds(snapshot)
    if max_fixes is not None and max_fixes < 0:
        raise ValueError(f"max_fixes must be >= 0, got {max_fixes}")
    if reachable_end is not None:
        if reachable_T is None:
            raise ValueError("pass reachable_T together with reachable_end")
        if not reachable_T <= reachable_end:
            raise ValueError("reachable_T must be a subset of reachable_end")
    _require(bugfix, ("sha", "is_bugfix"), "bugfix")
    szz_rows = _szz_rows(szz)
    targets = _sorted_paths(paths)
    index = pd.Index(targets, dtype="str", name="path")
    if not targets:
        return pd.DataFrame({c: np.zeros(0, dtype=np.int64) for c in DEFECT_HISTORY_COLUMNS}, index=index)

    bugfix_shas = bugfix.column("sha").to_pylist()
    bugfix_flags = bugfix.column("is_bugfix").to_pylist()
    flagged = {sha for sha, flag in zip(bugfix_shas, bugfix_flags, strict=True) if flag and sha is not None}
    commits = _commits(hist)
    periods = _Periods(t, t, reachable_T, None)
    prior_fixes = sorted(
        (when, sha)
        for sha, when, bulk in zip(commits.sha, commits.time, commits.bulk, strict=True)
        if sha in flagged and not bulk and periods.prior(sha, when)
    )
    fix_set = {sha for _, sha in prior_fixes}

    touches = file_touches(hist, snapshot, lineage_at_T, targets, reachable=reachable_T)
    touched = touches[touches["sha"].isin(fix_set)] if fix_set else touches.iloc[0:0]
    bugfix_ids: list[set[str]] = [set() for _ in targets]
    position = {path: i for i, path in enumerate(targets)}
    for path, sha in zip(touched["path"].tolist(), touched["sha"].tolist(), strict=True):
        bugfix_ids[position[path]].add(commits.ident(sha))

    if max_fixes is not None:
        newest = sorted(prior_fixes, key=lambda item: (-item[0], item[1]))[:max_fixes]
        traced = {sha for _, sha in newest}
    else:
        traced = fix_set
    names = {path: lineage_at_T.get(path, frozenset()) | {path} for path in targets}
    owners = _owners(names, targets)
    latent_ids: list[set[str]] = [set() for _ in targets]
    for row in szz_rows:
        if row.fix_sha not in traced:
            continue
        for i in owners.get(row.path, ()):
            latent_ids[i].add(commits.ident(row.fix_sha))

    return pd.DataFrame(
        {
            "prior_bugfix_commits": np.asarray([len(s) for s in bugfix_ids], dtype=np.int64),
            "prior_latent_bugs": np.asarray([len(s) for s in latent_ids], dtype=np.int64),
        },
        index=index,
    )[list(DEFECT_HISTORY_COLUMNS)]

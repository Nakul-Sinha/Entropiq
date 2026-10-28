"""Feature assembly: the feature table of one repository snapshot (``build_features``).

Definitions used throughout:

- **T** is ``snapshot.time`` (timezone-aware). The **state at T** is the tree of ``snapshot.sha``, and the
  **history up to T** is the set of non-merge commits reachable from ``snapshot.sha`` (plan section 15.5),
  computed with :func:`~tdp_core.mining.reach.reachable_shas` unless the caller passes it as ``reachable``.
  Commits in ``hist`` outside that set are never read, so a history mined at HEAD and one mined at
  ``snapshot.sha`` give identical features. Ages and decay weights use author times, clipped at zero.
- **Lineage** is :func:`~tdp_core.mining.lineage.path_lineage` over the reachable commits, with ``until`` =
  their newest author time (``until = T`` would drop reachable commits authored after T).
- **Files**: every regular file at T that has a language plugin. Files that ``skip_reason`` rejects
  (too large, vendored, generated, minified) and files whose parse times out or fails are not parsed and are
  listed in ``BuildResult.skipped``. Parsed files are split with ``tdp_core.parsing.is_test_file``: test
  files feed the import graph and the test signals, and every other parsed file is one **row**.
- **cc_trend** is the least-squares slope of ``cc_sum`` (module plus function cyclomatic complexity) over the
  file's versions at T, T - 90 d, T - 180 d, T - 270 d and T - 360 d, in complexity per year (365.25 days),
  with nominal x offsets. The state at T - 90 k days is the newest first-parent ancestor of ``snapshot.sha``
  with committer date at or before that time. A version is found under the current path, else under the
  first other lineage name (sorted) with the same language; missing, skipped, timed-out or failed versions
  are dropped, and fewer than two versions give 0.
- **SATD ages** come from one ``git blame --line-porcelain -w`` per file with SATD candidates, run at
  ``snapshot.sha`` on the candidate lines only: age in days = (T - author time) / 86,400.
- **Percentiles**: ``pct_<f>`` is ``rank(pct=True, method="average")`` of the stored float32 value of ``f``
  within the snapshot, for every non-binary feature.

Units: the static maxima (``cc_max``, ``cog_max``, ``nesting_max``, ``longest_function`` in lines,
``params_max``) are over functions only and 0 for a file without functions; ``rel_churn_decayed`` is decayed
churned lines per SLOC; the other families keep the units of their modules.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import (
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    FEATURE_INPUT_COLUMNS,
    INDEX_COLUMNS,
    CloneStats,
    FileSyntax,
    Snapshot,
)
from tdp_core.features.cache import ParseTask, StaticCache, parse_blobs
from tdp_core.features.evolution import evolution_features, ownership_features
from tdp_core.features.satd import SATD_COLUMNS, SatdClassifier, satd_candidates, satd_features
from tdp_core.features.tests_signals import test_features
from tdp_core.graph.centrality import graph_features
from tdp_core.graph.cochange import CochangeEdge, cochange_edges, cochange_features
from tdp_core.graph.imports import build_import_graph
from tdp_core.labeling.labels import DEFECT_HISTORY_COLUMNS, defect_history_features, index_frame
from tdp_core.mining.git import GitError, check_rev, run_git
from tdp_core.mining.gitlog import MinedHistory, _unix_seconds
from tdp_core.mining.lineage import path_lineage
from tdp_core.mining.reach import reachable_shas
from tdp_core.mining.tree import list_files, read_blobs
from tdp_core.parsing import get_plugin, is_test_file, skip_reason
from tdp_core.parsing.clones import clone_stats

__all__ = [
    "BUILD_STAGES",
    "CC_TREND_STEP_DAYS",
    "CC_TREND_VERSIONS",
    "DAYS_PER_YEAR",
    "STATIC_COLUMNS",
    "BuildResult",
    "build_features",
    "cc_sum",
    "static_features",
    "trend_slope",
]

logger = logging.getLogger(__name__)

STATIC_COLUMNS: tuple[str, ...] = FEATURE_FAMILIES["S"]
BUILD_STAGES: tuple[str, ...] = ("parse", "graphs", "features")
CC_TREND_STEP_DAYS: Final = 90
CC_TREND_VERSIONS: Final = 5  # the state at T plus four earlier states
DAYS_PER_YEAR: Final = 365.25

_STATIC_FLOAT = frozenset({"cc_mean", "dup_ratio"})
_INT16 = np.iinfo(np.int16)
_DAY_S = 86_400.0
_READ_CHUNK = 1000  # blobs per read_blobs call
_MAX_RANGES_PER_BLAME = 200
_BLAME_TIMEOUT_S = 300.0
_BLAME_ARGS = (
    "-c",
    "blame.ignoreRevsFile=",
    "-c",
    "diff.algorithm=myers",
    "--literal-pathspecs",
    "blame",
    "--line-porcelain",
    "-w",
)
_BLAME_HEADER = re.compile(rb"^[0-9a-f]{40}(?:[0-9a-f]{24})? \d+ (\d+)(?: \d+)?$")

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BuildResult:
    """Everything ``build_features`` computed for one snapshot.

    - ``features``: columns ``INDEX_COLUMNS + FEATURE_INPUT_COLUMNS`` (93), a ``RangeIndex`` and one row per
      parsed non-test file, sorted by path. Index dtypes ``str``, int16, ``datetime64[us, UTC]``, ``str``
      (as ``labels.index_frame``, so labels merge on ``INDEX_COLUMNS``); every feature column float32.
    - ``syntax``: every parsed file at T (test files included), sorted by path, ``syntax[p].path == p``.
    - ``blob_shas``: the blob sha at ``snapshot.sha`` of every key of ``syntax``.
    - ``import_graph``: ``build_import_graph(syntax, is_test_file)`` (nodes carry ``is_test``).
    - ``cochange_edges``: the strong co-change pairs among the rows (``graph.cochange.cochange_edges``).
    - ``test_paths``: the parsed test files, sorted.
    - ``skipped``: every file at T with a language plugin that was not parsed, mapped to its reason
      (a ``skip_reason`` value, ``"timeout"`` or ``"parse_failed"``), sorted, test files included.
    - ``lineage``: the full ``path_lineage`` output at T (every live path of any type).
    - ``clones``: ``clone_stats`` over the rows (the source of ``dup_ratio``), for cost estimates.
    - ``timings_s``: seconds spent in each stage of :data:`BUILD_STAGES`.
    """

    features: pd.DataFrame
    syntax: dict[str, FileSyntax]
    blob_shas: dict[str, str]
    import_graph: nx.DiGraph[str]
    cochange_edges: list[CochangeEdge]
    test_paths: tuple[str, ...]
    skipped: dict[str, str]
    lineage: dict[str, frozenset[str]]
    clones: dict[str, CloneStats] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------- helpers
def cc_sum(syntax: FileSyntax) -> int:
    """Total cyclomatic complexity of a file: ``module_cc`` plus every function's cyclomatic complexity."""
    return syntax.module_cc + sum(f.cyclomatic for f in syntax.functions)


def trend_slope(points: Sequence[tuple[float, float]]) -> float:
    """Least-squares slope of y on x for ``(x, y)`` points (same as ``np.polyfit(x, y, 1)[0]``).

    Returns 0.0 for fewer than two points or when every x is the same.
    """
    if len(points) < 2:
        return 0.0
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    dx = x - x.mean()
    denom = float(np.dot(dx, dx))
    if denom == 0.0:
        return 0.0
    return float(np.dot(dx, y - y.mean()) / denom)


def static_features(
    paths: Sequence[str],
    syntax_by_path: Mapping[str, FileSyntax],
    clones: Mapping[str, CloneStats],
) -> pd.DataFrame:
    """The static family (``STATIC_COLUMNS``) of ``paths`` from their parses and clone statistics.

    - ``sloc``, ``n_classes``: from ``FileSyntax``; ``n_functions``: number of functions and methods.
    - ``cc_sum``: :func:`cc_sum`; ``cc_mean``: mean cyclomatic complexity over functions (0.0 without).
    - ``cc_max``, ``cog_max``, ``nesting_max``, ``params_max``: maxima over functions; ``longest_function``:
      the largest ``end_line - start_line + 1`` in lines. All 0 for a file without functions (module code
      only counts in ``cc_sum``).
    - ``dup_ratio``: ``clones[p].dup_ratio`` (0..1).

    Returns a frame indexed by the sorted unique ``paths`` (``pd.Index``, dtype ``"str"``, name ``"path"``)
    with int64 columns except ``cc_mean`` and ``dup_ratio`` (float64). Empty ``paths`` give an empty frame
    with the same columns and dtypes. Raises ``KeyError`` for a path missing from either mapping.
    """
    targets = sorted(set(paths))
    n = len(targets)
    ints = {c: np.zeros(n, dtype=np.int64) for c in STATIC_COLUMNS if c not in _STATIC_FLOAT}
    floats = {c: np.zeros(n, dtype=np.float64) for c in _STATIC_FLOAT}
    for i, path in enumerate(targets):
        syntax = syntax_by_path[path]
        floats["dup_ratio"][i] = clones[path].dup_ratio
        functions = syntax.functions
        cyclomatic = [f.cyclomatic for f in functions]
        floats["cc_mean"][i] = sum(cyclomatic) / len(cyclomatic) if cyclomatic else 0.0
        ints["sloc"][i] = syntax.sloc
        ints["cc_sum"][i] = cc_sum(syntax)
        ints["cc_max"][i] = max(cyclomatic, default=0)
        ints["cog_max"][i] = max((f.cognitive for f in functions), default=0)
        ints["nesting_max"][i] = max((f.max_nesting for f in functions), default=0)
        ints["n_functions"][i] = len(functions)
        ints["n_classes"][i] = syntax.classes
        ints["longest_function"][i] = max((f.end_line - f.start_line + 1 for f in functions), default=0)
        ints["params_max"][i] = max((f.params for f in functions), default=0)
    values: dict[str, npt.NDArray[np.int64] | FloatArray] = {**ints, **floats}
    return pd.DataFrame(
        {c: values[c] for c in STATIC_COLUMNS}, index=pd.Index(targets, dtype="str", name="path")
    )


def _chunks[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _line_runs(lines: Sequence[int]) -> list[tuple[int, int]]:
    """Sorted unique 1-based ``lines`` merged into inclusive ``(first, last)`` runs of consecutive lines."""
    runs: list[tuple[int, int]] = []
    for line in sorted(set(lines)):
        if runs and line == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], line)
        else:
            runs.append((line, line))
    return runs


def _parse_blame_times(raw: bytes) -> list[tuple[int, int]]:
    """``(final_line, author_time)`` per line of ``git blame --line-porcelain`` output.

    Each entry is a header ``<sha> <orig_line> <final_line> [<count>]``, key lines and a content line that
    starts with a TAB and ends the entry. Raises ``GitError`` for malformed output (content without a header
    or ``author-time``, an unexpected line where a header belongs, or a truncated last entry).
    """
    out: list[tuple[int, int]] = []
    final_line: int | None = None
    author_time: int | None = None
    for line in raw.split(b"\n"):
        if final_line is None:
            if not line:
                continue
            match = _BLAME_HEADER.match(line)
            if match is None:
                raise GitError(f"unexpected git blame output: {line[:120]!r}")
            final_line = int(match[1])
            author_time = None
        elif line.startswith(b"\t"):
            if author_time is None:
                raise GitError(f"git blame entry for line {final_line} has no author-time")
            out.append((final_line, author_time))
            final_line = None
        elif line.startswith(b"author-time "):
            try:
                author_time = int(line[len(b"author-time ") :])
            except ValueError as exc:
                raise GitError(f"bad author-time in git blame output: {line[:120]!r}") from exc
    if final_line is not None:
        raise GitError(f"truncated git blame output after line {final_line}")
    return out


def _satd_line_ages(
    repo_dir: Path, sha: str, path: str, lines: Sequence[int], t_unix: int
) -> dict[int, float] | None:
    """Age in days at ``t_unix`` of each of ``lines`` of ``path`` at ``sha``, from git blame; None on failure.

    One ``git blame --line-porcelain -w`` call per 200 runs of consecutive lines (one call per file in
    practice), with the ignore-revs file and the diff algorithm pinned. Blame at ``sha`` only sees history
    reachable from it, follows whole-file renames and ignores whitespace-only edits. Ages can be negative
    for lines authored after T (``satd_features`` clips them). A name that is not valid UTF-8 (it cannot be
    passed back to git) and any git failure give None, so the file's SATD age is unknown (0).
    """
    if "\ufffd" in path:
        logger.debug("SATD: not blaming %r (the name is not valid UTF-8)", path)
        return None
    runs = _line_runs(lines)
    ages: dict[int, float] = {}
    try:
        for chunk in _chunks(runs, _MAX_RANGES_PER_BLAME):
            args = list(_BLAME_ARGS)
            for first, last in chunk:
                args += ["-L", f"{first},{last}"]
            args += [sha, "--", path]
            for final_line, author_time in _parse_blame_times(
                run_git(args, repo_dir, timeout=_BLAME_TIMEOUT_S)
            ):
                ages[final_line] = (t_unix - author_time) / _DAY_S
    except GitError as exc:
        logger.warning("SATD: git blame of %s at %s failed, its SATD age is unknown: %s", path, sha, exc)
        return None
    return ages


def _state_before(repo_dir: Path, sha: str, t_unix: int) -> str | None:
    """The newest first-parent ancestor of ``sha`` (itself included) with committer date ``<= t_unix``.

    The same rule as ``mining.reach.snapshot_sha``, walking from a commit instead of a branch. None when
    every such commit is newer (or ``t_unix`` is before 1970).
    """
    if t_unix < 0:
        return None
    check_rev(sha)
    out = run_git(
        ["rev-list", "-1", "--first-parent", f"--before=@{t_unix} +0000", "--end-of-options", sha, "--"],
        repo_dir,
    )
    return out.decode("ascii", errors="replace").strip() or None


def _stage_done(
    name: str, start: float, timings: dict[str, float], on_stage: Callable[[str], None] | None
) -> None:
    timings[name] = time.perf_counter() - start
    if on_stage is not None:
        on_stage(name)


# ------------------------------------------------------------------------------------------- stages
@dataclass(frozen=True, slots=True)
class _Parsed:
    syntax: dict[str, FileSyntax]  # parsed files at T, sorted by path
    blob_shas: dict[str, str]
    skipped: dict[str, str]


def _parse_snapshot(
    repo_dir: Path, sha: str, cache: StaticCache | None, settings: Settings, workers: int
) -> _Parsed:
    files = list_files(repo_dir, sha)
    candidates = sorted(p for p in files if get_plugin(p) is not None)
    skipped: dict[str, str] = {}
    tasks: list[ParseTask] = []
    for chunk in _chunks(candidates, _READ_CHUNK):
        sources = read_blobs(repo_dir, [files[p] for p in chunk])
        for path in chunk:
            source = sources[files[path]]
            reason = skip_reason(path, source, settings)
            if reason is not None:
                skipped[path] = reason
            else:
                tasks.append(ParseTask(files[path], path, source))
    results = parse_blobs(tasks, cache, timeout_s=settings.parse_timeout_s, jobs=workers)
    syntax: dict[str, FileSyntax] = {}
    blob_shas: dict[str, str] = {}
    for task in tasks:
        result = results[task.key]
        if isinstance(result, FileSyntax):
            syntax[task.path] = result if result.path == task.path else replace(result, path=task.path)
            blob_shas[task.path] = task.key
        else:
            skipped[task.path] = result
    return _Parsed(syntax, blob_shas, dict(sorted(skipped.items())))


def _cc_trends(
    repo_dir: Path,
    snapshot: Snapshot,
    rows: Sequence[str],
    parsed: _Parsed,
    lineage: Mapping[str, frozenset[str]],
    cache: StaticCache | None,
    settings: Settings,
    workers: int,
) -> dict[str, float]:
    """``cc_trend`` of every row (see the module docstring)."""
    trends = dict.fromkeys(rows, 0.0)
    if not rows:
        return trends
    t_unix = _unix_seconds(snapshot.time)
    states: list[tuple[int, str]] = []
    for k in range(1, CC_TREND_VERSIONS):
        state = _state_before(repo_dir, snapshot.sha, t_unix - k * CC_TREND_STEP_DAYS * 86_400)
        if state is not None:
            states.append((k, state))
    trees = {sha: list_files(repo_dir, sha) for sha in dict.fromkeys(sha for _, sha in states)}

    # Version k of row p: (k, name, blob) under p or the first other lineage name with p's language.
    versions: dict[str, list[tuple[int, str, str]]] = {}
    for path in rows:
        plugin = get_plugin(path)
        names = sorted(lineage.get(path, frozenset()) | {path})
        found: list[tuple[int, str, str]] = []
        for k, sha in states:
            tree = trees[sha]
            name = path if path in tree else None
            if name is None:
                name = next((n for n in names if n in tree and get_plugin(n) is plugin), None)
            if name is not None:
                found.append((k, name, tree[name]))
        versions[path] = found

    # A (name, blob) pair that was parsed at T passed skip_reason with the same arguments. Other pairs are
    # checked under their old name; blobs parsed at T are reused, the rest go through the cache.
    known = {blob: parsed.syntax[p] for p, blob in parsed.blob_shas.items()}
    used = {(name, blob) for found in versions.values() for _, name, blob in found}
    unchecked = sorted(used - set(parsed.blob_shas.items()))
    accepted: set[tuple[str, str]] = set()
    tasks: list[ParseTask] = []
    for chunk in _chunks(unchecked, _READ_CHUNK):
        sources = read_blobs(repo_dir, [blob for _, blob in chunk])
        for name, blob in chunk:
            source = sources[blob]
            if skip_reason(name, source, settings) is not None:
                continue
            accepted.add((name, blob))
            if blob not in known:
                tasks.append(ParseTask(blob, name, source))
    results = parse_blobs(tasks, cache, timeout_s=settings.parse_timeout_s, jobs=workers)

    for path in rows:
        points = [(0.0, float(cc_sum(parsed.syntax[path])))]
        for k, name, blob in versions[path]:
            if parsed.blob_shas.get(name) != blob and (name, blob) not in accepted:
                continue
            version = known.get(blob) or results.get(blob)
            if isinstance(version, FileSyntax):
                points.append((-k * CC_TREND_STEP_DAYS / DAYS_PER_YEAR, float(cc_sum(version))))
        trends[path] = trend_slope(points)
    return trends


def _satd(
    repo_dir: Path,
    snapshot: Snapshot,
    rows: Sequence[str],
    syntax: Mapping[str, FileSyntax],
    satd_model: SatdClassifier | None,
    workers: int,
) -> dict[str, dict[str, float]]:
    """The SATD features of every row, with ages from git blame of the candidate lines."""
    t_unix = _unix_seconds(snapshot.time)
    blame_lines = {
        path: [c.line for c in candidates]
        for path in rows
        if (candidates := satd_candidates(syntax[path].comments))
    }
    ages: dict[str, dict[int, float] | None]
    if workers > 1 and len(blame_lines) > 1:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(blame_lines)), thread_name_prefix="tdp-blame"
        ) as ex:
            futures = {
                path: ex.submit(_satd_line_ages, repo_dir, snapshot.sha, path, lines, t_unix)
                for path, lines in blame_lines.items()
            }
            ages = {path: future.result() for path, future in futures.items()}
    else:
        ages = {
            path: _satd_line_ages(repo_dir, snapshot.sha, path, lines, t_unix)
            for path, lines in blame_lines.items()
        }
    return {path: satd_features(syntax[path], satd_model, ages.get(path)) for path in rows}


def _assemble(
    snapshot: Snapshot,
    rows: Sequence[str],
    frames: Sequence[pd.DataFrame],
    extra: Mapping[str, FloatArray],
) -> pd.DataFrame:
    """Join the component frames and extra columns into the final float32 feature table with percentiles."""
    values: dict[str, FloatArray] = {}
    for frame in frames:
        if list(frame.index) != list(rows):
            raise RuntimeError(
                f"a component frame with columns {list(frame.columns)} is not indexed by the rows"
            )
        for column in frame.columns:
            values[str(column)] = frame[column].to_numpy(dtype=np.float64)
    values.update(extra)
    if len(values) != len(FEATURE_COLUMNS) or set(values) != set(FEATURE_COLUMNS):
        missing = sorted(set(FEATURE_COLUMNS) - set(values))
        unknown = sorted(set(values) - set(FEATURE_COLUMNS))
        raise RuntimeError(f"feature columns do not match the contract: missing {missing}, unknown {unknown}")
    base = pd.DataFrame({c: values[c].astype(np.float32) for c in FEATURE_COLUMNS})
    bad = [c for c in FEATURE_COLUMNS if not np.isfinite(base[c].to_numpy()).all()]
    if bad:
        raise RuntimeError(f"non-finite feature values in {bad}")
    pct = pd.DataFrame(
        {
            f"pct_{c}": base[c].rank(pct=True, method="average").astype(np.float32)
            for c in FEATURE_COLUMNS
            if c not in BINARY_FEATURES
        }
    )
    features = pd.concat([index_frame(snapshot, rows), base, pct], axis=1)
    return features[list(INDEX_COLUMNS + FEATURE_INPUT_COLUMNS)]


# ------------------------------------------------------------------------------------------- entry point
def build_features(
    repo_dir: Path,
    hist: MinedHistory,
    snapshot: Snapshot,
    *,
    bugfix: pa.Table | None,
    szz: pa.Table | None,
    satd_model: SatdClassifier | None,
    cache: StaticCache | None,
    settings: Settings = DEFAULT_SETTINGS,
    jobs: int = 0,
    reachable: AbstractSet[str] | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> BuildResult:
    """Build the feature table of ``snapshot`` (definitions in the module docstring).

    Arguments:

    - ``repo_dir``: a work tree or bare mirror holding ``snapshot.sha``; ``hist``: history mined from it (at
      ``snapshot.sha`` or any descendant; commits outside the history up to T are ignored).
    - ``bugfix`` and ``szz``: a ``classify_bugfix`` table and an ``szz_table`` for the defect-history
      features, passed together; ``prior_latent_bugs`` counts the newest ``settings.inference_szz_max_fixes``
      prior fixes, as at inference. Both None give zeros. An empty ``SZZ_SCHEMA`` table gives
      ``prior_latent_bugs = 0`` with real ``prior_bugfix_commits``.
    - ``satd_model``: a SATD classifier (``load_default_classifier()``) or None to count every candidate.
    - ``cache``: a static cache for parses (``DiskStaticCache``) or None. A cache that raises is dropped
      with a warning, so the result equals a build without a cache.
    - ``jobs``: parse workers and git blame threads; 0 means ``os.cpu_count()``.
    - ``reachable``: the history up to T when the caller already has it (for example ``reachable_T`` from
      ``labels.reachable_window``); None computes ``reachable_shas(repo_dir, snapshot.sha)``.
    - ``on_stage``: called with each name of :data:`BUILD_STAGES` when that stage finishes.

    Every file with a language plugin at T is parsed (through ``cache``, with the per-file limit
    ``settings.parse_timeout_s``) or listed in ``skipped``. Returns a :class:`BuildResult`; a snapshot without
    rows gives a table with zero rows and the same columns and dtypes.

    Raises ``ValueError`` for a naive ``snapshot.time``, a ``snapshot.index`` outside int16, ``jobs < 0``,
    exactly one of ``bugfix`` and ``szz``, or a non-positive ``settings.parse_timeout_s``; ``GitError`` when
    git cannot read ``snapshot.sha`` in ``repo_dir``.
    """
    _unix_seconds(snapshot.time)  # ValueError for a naive time
    if not _INT16.min <= snapshot.index <= _INT16.max:
        raise ValueError(f"snapshot index {snapshot.index} does not fit in int16")
    if jobs < 0:
        raise ValueError(f"jobs must be >= 0, got {jobs}")
    if (bugfix is None) != (szz is None):
        raise ValueError(
            "pass bugfix and szz together; an empty SZZ_SCHEMA table gives prior_latent_bugs = 0"
        )
    if not settings.parse_timeout_s > 0:
        raise ValueError(f"settings.parse_timeout_s must be positive, got {settings.parse_timeout_s!r}")
    workers = jobs or os.cpu_count() or 1
    timings: dict[str, float] = {}

    # ---- parse
    start = time.perf_counter()
    parsed = _parse_snapshot(repo_dir, snapshot.sha, cache, settings, workers)
    syntax = parsed.syntax
    test_paths = tuple(p for p in syntax if is_test_file(p))
    rows = [p for p in syntax if not is_test_file(p)]
    _stage_done("parse", start, timings, on_stage)

    # ---- graphs
    start = time.perf_counter()
    reach = frozenset(reachable) if reachable is not None else reachable_shas(repo_dir, snapshot.sha)
    commits_t = hist.commits.filter(
        pc.is_in(hist.commits.column("sha"), value_set=pa.array(sorted(reach), type=pa.string()))
    )
    until = pc.max(commits_t.column("author_time")).as_py() if commits_t.num_rows else None
    lineage = path_lineage(hist.file_changes, commits_t, until or snapshot.time)
    graph = build_import_graph(syntax, is_test_file)
    graph_feats = graph_features(graph, settings)
    co_args = (hist, snapshot, lineage, rows, is_test_file, graph, settings)
    cochange = cochange_features(*co_args, reachable=reach)
    edges = cochange_edges(*co_args, reachable=reach)
    _stage_done("graphs", start, timings, on_stage)

    # ---- features
    start = time.perf_counter()
    clones = clone_stats({p: syntax[p] for p in rows}, settings)
    static = static_features(rows, syntax, clones)
    evolution = evolution_features(hist, snapshot, lineage, rows, settings, reachable=reach)
    ownership = ownership_features(hist, snapshot, lineage, rows, settings, reachable=reach)
    tests = test_features(rows, test_paths, syntax, graph)
    satd = _satd(repo_dir, snapshot, rows, syntax, satd_model, workers)
    if bugfix is not None and szz is not None:
        defect = defect_history_features(
            hist,
            snapshot,
            lineage,
            rows,
            bugfix,
            szz,
            reachable_T=reach,
            max_fixes=settings.inference_szz_max_fixes,
        )
    else:
        defect = pd.DataFrame(
            {c: np.zeros(len(rows), dtype=np.int64) for c in DEFECT_HISTORY_COLUMNS},
            index=pd.Index(rows, dtype="str", name="path"),
        )
    trends = _cc_trends(repo_dir, snapshot, rows, parsed, lineage, cache, settings, workers)
    extra: dict[str, FloatArray] = {
        "rel_churn_decayed": evolution["churn_decayed"].to_numpy(dtype=np.float64)
        / np.maximum(static["sloc"].to_numpy(dtype=np.float64), 1.0),
        "cc_trend": np.array([trends[p] for p in rows], dtype=np.float64),
    }
    for column in SATD_COLUMNS:
        extra[column] = np.array([satd[p][column] for p in rows], dtype=np.float64)
    frames = (static, evolution, ownership, defect, graph_feats, cochange, tests)
    features = _assemble(snapshot, rows, frames, extra)
    _stage_done("features", start, timings, on_stage)

    logger.debug(
        "%s snapshot %d (%s): %d rows, %d tests, %d skipped, %d co-change edges, timings %s",
        snapshot.repo,
        snapshot.index,
        snapshot.sha,
        len(rows),
        len(test_paths),
        len(parsed.skipped),
        len(edges),
        {k: round(v, 3) for k, v in timings.items()},
    )
    return BuildResult(
        features=features,
        syntax=syntax,
        blob_shas=parsed.blob_shas,
        import_graph=graph,
        cochange_edges=edges,
        test_paths=test_paths,
        skipped=parsed.skipped,
        lineage=lineage,
        clones=clones,
        timings_s=timings,
    )

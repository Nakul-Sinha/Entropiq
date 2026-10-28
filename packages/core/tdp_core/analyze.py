"""Analyse one commit of a repository end to end: mine, parse, build features, score, explain and plan.

:func:`analyze` runs these stages in order and reports each one through ``progress`` when it **finishes**,
with the percentage of the whole analysis reached at that point (:data:`ANALYSIS_STAGES`):

1. ``mine`` (25): resolve the revision to its commit, mine the non-merge history reachable from it
   (``settings.history_window_days`` back from its committer time, at most ``settings.max_commits``
   commits), detect bug-fix commits by keyword (no issue tracker) and run SZZ for the newest
   ``settings.inference_szz_max_fixes`` fixes.
2. ``parse`` (50), ``graphs`` (60), ``features`` (70): :func:`~tdp_core.features.build.build_features` at
   the analysed commit, with every mined commit as the history up to T (plan section 15.5).
3. ``score`` (80): ranker margins, priority percentiles, bands and calibrated risk with its range.
4. ``explain`` (90): SHAP reasons, mitigations and debt types relative to the repository mean.
5. ``plan`` (95): cost estimates, the fix plan within the budget, monthly activity, the graph and the
   per-file extras.

The snapshot time T is the analysed commit's committer time, so the result depends only on the commit, the
bundle and the settings (``analyzed_at`` and timings aside), and is the same for any ``jobs`` value and with
or without caches. The analysis is read-only: every git command goes through
:func:`~tdp_core.mining.git.run_git` and none of them clones, fetches or writes to ``repo_dir``. The only
writes go to the optional static ``cache`` and the optional ``szz_cache_dir``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import networkx as nx
import pandas as pd

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import (
    EXTRACTOR_VERSION,
    FEATURE_INPUT_COLUMNS,
    AnalysisResult,
    CostEstimate,
    FileExtra,
    FileScore,
    GraphData,
    GraphEdge,
    GraphNode,
    Snapshot,
    band_for,
)
from tdp_core.features.build import BuildResult, build_features
from tdp_core.features.cache import StaticCache
from tdp_core.features.evolution import commit_activity
from tdp_core.features.satd import SatdClassifier
from tdp_core.features.satd_model import load_default_classifier
from tdp_core.graph.cochange import CochangeEdge
from tdp_core.labeling.bugfix import classify_bugfix
from tdp_core.labeling.szz import szz_table
from tdp_core.mining.git import GitError, TransientGitError, check_rev, run_git
from tdp_core.mining.gitlog import mine_history
from tdp_core.model.explain import Explainer, Explanation
from tdp_core.model.priority import priority_from_scores
from tdp_core.planning.cost import estimate_cost
from tdp_core.planning.plan import make_plan
from tdp_core.serialize import to_jsonable

if TYPE_CHECKING:  # importing the bundle module loads xgboost, which only the ranker itself needs
    from tdp_core.model.bundle import Bundle

__all__ = ["ACTIVITY_MONTHS", "ANALYSIS_STAGES", "DEFAULT_JOBS", "ProgressFn", "analyze"]

logger = logging.getLogger(__name__)

type ProgressFn = Callable[[str, int, str], None]
"""Progress callback ``(stage, pct, message)``: ``pct`` is 0..100 of the analysis, ``message`` one line."""

ANALYSIS_STAGES: Final[tuple[tuple[str, int], ...]] = (
    ("mine", 25),
    ("parse", 50),
    ("graphs", 60),
    ("features", 70),
    ("score", 80),
    ("explain", 90),
    ("plan", 95),
)
DEFAULT_JOBS: Final = 4  # parse workers, blame and SZZ threads (the resource cap of plan section 15.3)
ACTIVITY_MONTHS: Final = 24  # months of FileExtra.commit_activity

_STAGE_PCT: Final[Mapping[str, int]] = dict(ANALYSIS_STAGES)
_BUILD_MESSAGES: Final[Mapping[str, str]] = {
    "parse": "Parsed the source files",
    "graphs": "Built the import and co-change graphs",
    "features": "Computed the file features",
}
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_DAY_S: Final = 86_400
# git misreads ``--since=@<ts>`` for small timestamps (before 1973-03-03), so earlier cutoffs mine everything.
_MIN_SINCE_S: Final = 100_000_000


def analyze(
    repo_dir: Path,
    sha: str,
    bundle: Bundle,
    *,
    settings: Settings = DEFAULT_SETTINGS,
    cache: StaticCache | None = None,
    satd_model: SatdClassifier | None = None,
    progress: ProgressFn | None = None,
    budget_hours: float | None = None,
    jobs: int = DEFAULT_JOBS,
    szz_cache_dir: Path | None = None,
) -> AnalysisResult:
    """Analyse commit ``sha`` of the repository at ``repo_dir`` with ``bundle`` (see the module docstring).

    Arguments:

    - ``repo_dir``: an existing work tree or bare mirror (never cloned or fetched here).
    - ``sha``: any revision naming a commit (full or short sha, ``HEAD``, a branch or a tag); the result
      holds the full sha. Only the committed state is analysed, never the work tree.
    - ``bundle``: a loaded model bundle (``load_bundle``); its ranker, calibrator, reason templates and cost
      model are used.
    - ``settings``: history window in days, commit cap, SZZ cap, parse limit in seconds, default budget in
      hours and the minimum plan priority, among others.
    - ``cache``: a static parse cache (``DiskStaticCache``, ``PostgresStaticCache``) or None.
    - ``satd_model``: the SATD classifier; None uses ``load_default_classifier()``, as training does.
    - ``progress``: called as ``progress(stage, pct, message)`` after each stage, in :data:`ANALYSIS_STAGES`
      order with those percentages (0..100). An exception it raises stops the analysis and propagates.
    - ``budget_hours``: the plan budget in hours; None means ``settings.default_budget_hours``. Zero is a
      valid budget and gives an empty plan.
    - ``jobs``: parse workers, blame threads and SZZ threads; 0 means ``os.cpu_count()``.
    - ``szz_cache_dir``: a folder for per-fix SZZ results (one per repository), or None.

    Returns an :class:`~tdp_core.contracts.AnalysisResult`: ``files`` sorted by priority descending, then
    path; ``features`` indexed by path (sorted) with the columns ``FEATURE_INPUT_COLUMNS`` as float32;
    ``extras`` keyed by path; ``graph`` with one node per ranked file, the import edges between ranked files
    and one hidden-coupling edge per strongly co-changing pair without an import edge; the plan; and
    ``stats`` (JSON types only) with counts, the history window, the snapshot time, ``timings_s`` per stage
    and ``total_s`` in seconds. A commit without rankable files (only tests or no Python code) gives empty
    files, features with zero rows, empty extras and graph, and an empty plan.

    Raises ``ValueError`` for a budget that is not finite and >= 0, ``jobs < 0``, an invalid revision
    string, a negative history window, commit cap or SZZ cap, or a parse limit that is not positive, and
    ``FileNotFoundError`` when ``repo_dir`` is not a directory; these checks run before any git command.
    Raises ``GitError`` when ``sha`` names no commit or ``repo_dir`` is not a repository, and
    ``TransientGitError`` when git times out. A bundle with invalid reason templates or cost model values
    raises ``ValueError`` later, when the first file is explained or costed (a commit without rankable
    files never reaches those checks).
    """
    started = time.perf_counter()
    budget = settings.default_budget_hours if budget_hours is None else float(budget_hours)
    if not (math.isfinite(budget) and budget >= 0):
        raise ValueError(f"budget_hours must be finite and >= 0, got {budget_hours!r}")
    if jobs < 0:
        raise ValueError(f"jobs must be >= 0, got {jobs}")
    _check_settings(settings)
    check_rev(sha)
    repo = Path(repo_dir)
    if not repo.is_dir():
        raise FileNotFoundError(f"repository directory not found: {repo}")
    workers = jobs or os.cpu_count() or 1

    def report(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, _STAGE_PCT[stage], message)

    timings: dict[str, float] = {}

    # ---- mine
    mark = time.perf_counter()
    full_sha, when = _resolve_commit(repo, sha)
    shallow = run_git(["rev-parse", "--is-shallow-repository"], repo).strip() == b"true"
    if shallow:
        logger.warning(
            "%s is a shallow clone: history before the shallow boundary is missing, so history-based "
            "features undercount",
            repo,
        )
    since = _history_since(when, settings.history_window_days)
    sub = time.perf_counter()
    hist = mine_history(repo, full_sha, since=since, max_commits=settings.max_commits, settings=settings)
    mine_timings = {"history": time.perf_counter() - sub}
    sub = time.perf_counter()
    bugfix = classify_bugfix(hist.commits, hist.file_changes)
    mine_timings["bugfix"] = time.perf_counter() - sub
    sub = time.perf_counter()
    szz = szz_table(
        repo,
        hist,
        bugfix,
        max_fixes=settings.inference_szz_max_fixes,
        jobs=workers,
        cache_dir=szz_cache_dir,
    )
    mine_timings["szz"] = time.perf_counter() - sub
    n_commits = hist.commits.num_rows
    n_bugfix = sum(1 for flag in bugfix.column("is_bugfix").to_pylist() if flag)
    n_szz_fixes = len(set(szz.column("fix_sha").to_pylist()))
    timings["mine"] = time.perf_counter() - mark
    found = _count(n_bugfix, "bug fix", "bug fixes")
    report("mine", f"Mined {_count(n_commits, 'commit')} and found {found}")

    # ---- parse, graphs, features
    snapshot = Snapshot(repo=_repo_slug(repo), index=0, time=when, sha=full_sha)
    reach = frozenset(hist.commits.column("sha").to_pylist())
    model = satd_model if satd_model is not None else load_default_classifier()
    build = build_features(
        repo,
        hist,
        snapshot,
        bugfix=bugfix,
        szz=szz,
        satd_model=model,
        cache=cache,
        settings=settings,
        jobs=workers,
        reachable=reach,
        on_stage=lambda stage: report(stage, _BUILD_MESSAGES[stage]),
    )
    timings.update(build.timings_s)

    # ---- score
    mark = time.perf_counter()
    X = build.features.set_index("path")[list(FEATURE_INPUT_COLUMNS)]
    scores = bundle.ranker.score(X)
    priorities = priority_from_scores(scores)
    risk, risk_low, risk_high = bundle.calibrator.predict(scores)
    timings["score"] = time.perf_counter() - mark
    report("score", f"Scored {_count(len(X), 'file')}")

    # ---- explain
    mark = time.perf_counter()
    explanations: list[Explanation] = Explainer(bundle).explain(X) if len(X) else []
    files: list[FileScore] = []
    for i, (path, e) in enumerate(zip(X.index, explanations, strict=True)):
        priority = int(priorities[i])
        files.append(
            FileScore(
                path=str(path),
                score=float(scores[i]),
                priority=priority,
                band=band_for(priority),
                risk=float(risk[i]),
                risk_low=float(risk_low[i]),
                risk_high=float(risk_high[i]),
                reasons=e.reasons,
                mitigations=e.mitigations,
                debt_types=e.debt_types,
                shap_by_family=e.shap_by_family,
            )
        )
    files.sort(key=lambda f: (-f.priority, f.path))
    timings["explain"] = time.perf_counter() - mark
    report("explain", f"Explained {_count(len(files), 'file')}")

    # ---- plan
    mark = time.perf_counter()
    costs: dict[str, CostEstimate] = {
        str(path): estimate_cost(row, build.syntax[str(path)], build.clones[str(path)], bundle.cost_model)
        for path, row in X.iterrows()
    }
    plan = make_plan(files, X, costs, build.import_graph, budget, settings.min_plan_priority)
    paths = [str(p) for p in X.index]
    activity = commit_activity(hist, snapshot, build.lineage, paths, ACTIVITY_MONTHS, reachable=reach)
    graph = _graph_data(files, X, build.import_graph, build.cochange_edges)
    extras = _extras(paths, build, activity, costs)
    timings["plan"] = time.perf_counter() - mark
    hours = "hour" if budget == 1 else "hours"
    report("plan", f"Planned {_count(len(plan.steps), 'fix', 'fixes')} within {budget:g} {hours}")

    total = time.perf_counter() - started
    stats: dict[str, Any] = {
        "n_files": len(files),
        "n_test_files": len(build.test_paths),
        "n_skipped": len(build.skipped),
        "skipped": dict(build.skipped),
        "n_commits": n_commits,
        "history_since": to_jsonable(since) if since is not None else None,
        "history_truncated": n_commits >= settings.max_commits,
        "shallow": shallow,
        "snapshot_time": to_jsonable(when),
        "n_bugfix_commits": n_bugfix,
        "n_szz_fixes": n_szz_fixes,
        "timings_s": {stage: timings[stage] for stage, _ in ANALYSIS_STAGES},
        "mine_timings_s": mine_timings,
        "total_s": total,
    }
    logger.info(
        "analysed %s at %s: %d files, %d commits, %d plan steps (%.1f of %g hours) in %.1f s",
        snapshot.repo,
        full_sha,
        len(files),
        n_commits,
        len(plan.steps),
        plan.total_hours,
        budget,
        total,
    )
    return AnalysisResult(
        sha=full_sha,
        analyzed_at=datetime.now(UTC).replace(microsecond=0),
        model_version=bundle.version,
        extractor_version=EXTRACTOR_VERSION,
        files=tuple(files),
        features=X,
        extras=extras,
        graph=graph,
        plan=plan,
        stats=stats,
    )


# ------------------------------------------------------------------------------------------- helpers
def _check_settings(settings: Settings) -> None:
    """Reject settings that would only fail after git has run (negative windows and caps)."""
    window = settings.history_window_days
    if window is not None and window < 0:
        raise ValueError(f"settings.history_window_days must be >= 0 or None, got {window}")
    if settings.max_commits < 0:
        raise ValueError(f"settings.max_commits must be >= 0, got {settings.max_commits}")
    if settings.inference_szz_max_fixes < 0:
        raise ValueError(
            f"settings.inference_szz_max_fixes must be >= 0, got {settings.inference_szz_max_fixes}"
        )
    if not settings.parse_timeout_s > 0:
        raise ValueError(f"settings.parse_timeout_s must be positive, got {settings.parse_timeout_s!r}")


def _resolve_commit(repo: Path, rev: str) -> tuple[str, datetime]:
    """The full sha of the commit ``rev`` names (tags are peeled) and its committer time (aware UTC).

    Raises ``GitError`` when ``rev`` is unknown or names a tree or blob, and lets other git failures
    (not a repository, timeouts) propagate unchanged.
    """
    check_rev(rev)
    args = ["rev-parse", "--verify", "--quiet", "--end-of-options", f"{rev}^{{commit}}"]
    try:
        out = run_git(args, repo)
    except GitError as exc:
        if isinstance(exc, TransientGitError) or exc.returncode != 1:
            raise
        raise GitError(
            f"{rev!r} does not name a commit in {repo}",
            cmd=exc.cmd,
            returncode=exc.returncode,
            stderr=exc.stderr,
        ) from exc
    full_sha = out.decode("ascii").strip()
    stamp = run_git(
        ["-c", "log.showSignature=false", "log", "-1", "--format=%ct", "--end-of-options", full_sha, "--"],
        repo,
    )
    return full_sha, _EPOCH + timedelta(seconds=int(stamp.strip()))


def _history_since(when: datetime, window_days: int | None) -> datetime | None:
    """The ``--since`` cutoff ``window_days`` before ``when``; None for no window or a cutoff before 1973."""
    if window_days is None:
        return None
    cutoff = math.floor(when.timestamp()) - window_days * _DAY_S
    if cutoff < _MIN_SINCE_S:
        return None
    return _EPOCH + timedelta(seconds=cutoff)


def _repo_slug(repo: Path) -> str:
    """A readable repository name for ``Snapshot.repo``: the folder name without ``.git``."""
    resolved = repo.resolve()
    name = resolved.parent.name if resolved.name == ".git" else resolved.name
    name = name.removesuffix(".git")
    return name or "repo"


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _graph_data(
    files: Sequence[FileScore],
    X: pd.DataFrame,
    import_graph: nx.DiGraph[str],
    cochange_edges: Sequence[CochangeEdge],
) -> GraphData:
    """Nodes for the ranked files (in ``files`` order); import edges between them, then hidden coupling."""
    ranked = set(X.index)
    sloc = X["sloc"].to_dict()
    dependents = X["transitive_dependents"].to_dict()
    nodes = tuple(
        GraphNode(
            id=f.path,
            priority=f.priority,
            band=f.band,
            sloc=int(sloc[f.path]),
            dependents=int(dependents[f.path]),
        )
        for f in files
    )
    imports = sorted((u, v) for u, v in import_graph.edges if u != v and u in ranked and v in ranked)
    edges = [GraphEdge(u, v, "import", 1.0) for u, v in imports]
    coupled = [e for e in cochange_edges if e[0] in ranked and e[1] in ranked]
    edges += _hidden_coupling_edges(coupled, import_graph)
    return GraphData(nodes=nodes, edges=tuple(edges))


def _hidden_coupling_edges(edges: Sequence[CochangeEdge], import_graph: nx.DiGraph[str]) -> list[GraphEdge]:
    """One ``hidden_coupling`` edge per co-changing pair without an import edge in either direction.

    ``edges`` may hold both directions of a pair; the edge kept points along the stronger direction (higher
    confidence, ties to the direction whose source sorts first) and its weight is that confidence. The
    result is sorted by (source, target).
    """
    best: dict[tuple[str, str], tuple[str, str, float]] = {}
    for a, b, _support, conf in edges:
        if a == b or import_graph.has_edge(a, b) or import_graph.has_edge(b, a):
            continue
        key = (a, b) if a < b else (b, a)
        candidate = (a, b, float(conf))
        current = best.get(key)
        if current is None or (-candidate[2], candidate[0]) < (-current[2], current[0]):
            best[key] = candidate
    return [GraphEdge(a, b, "hidden_coupling", w) for a, b, w in sorted(best.values())]


def _extras(
    paths: Sequence[str],
    build: BuildResult,
    activity: Mapping[str, tuple[tuple[str, int], ...]],
    costs: Mapping[str, CostEstimate],
) -> dict[str, FileExtra]:
    """``FileExtra`` per ranked path, in path order: blob, functions, direct neighbours, activity and cost."""
    graph = build.import_graph
    out: dict[str, FileExtra] = {}
    for path in paths:
        importers = sorted(graph.predecessors(path))
        is_test = {q: bool(graph.nodes[q].get("is_test", False)) for q in importers}
        out[path] = FileExtra(
            blob_sha=build.blob_shas[path],
            functions=build.syntax[path].functions,
            dependents=tuple(q for q in importers if not is_test[q]),
            dependencies=tuple(sorted(graph.successors(path))),
            test_importers=tuple(q for q in importers if is_test[q]),
            commit_activity=activity[path],
            cost=costs[path],
        )
    return out

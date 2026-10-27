"""Budgeted fix plans: choose files by benefit per hour within a budget, then order them for fixing."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import networkx as nx
import pandas as pd

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import CostEstimate, FileScore, GraphEdge, Plan, PlanStep
from tdp_core.planning.knapsack import knapsack
from tdp_core.planning.schedule import schedule

logger = logging.getLogger(__name__)


def benefit(risk: float, transitive_dependents: float) -> float:
    """Expected value of fixing a file: ``risk * (1 + log2(1 + transitive_dependents))``.

    ``risk`` is the calibrated probability (0..1) and ``transitive_dependents`` the number of files that
    depend on this one directly or indirectly. A file nobody depends on is worth its risk; each doubling of
    dependents adds one more risk. Raises ``ValueError`` unless ``0 <= risk <= 1`` and
    ``transitive_dependents`` is finite and >= 0.
    """
    r = float(risk)
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"risk must be within [0, 1], got {risk!r}")
    deps = float(transitive_dependents)
    if not math.isfinite(deps) or deps < 0:
        raise ValueError(f"transitive_dependents must be finite and >= 0, got {transitive_dependents!r}")
    return r * (1.0 + math.log2(1.0 + deps))


def make_plan(
    scores: Sequence[FileScore],
    features: pd.DataFrame,
    costs: Mapping[str, CostEstimate],
    import_graph: nx.DiGraph[str],
    budget_hours: float,
    min_priority: int = DEFAULT_SETTINGS.min_plan_priority,
) -> Plan:
    """Pick the files worth fixing within ``budget_hours`` and order them for fixing.

    Candidates are the scored files with ``priority >= min_priority`` that have an entry in ``costs`` with
    ``hours > 0``; only ``path``, ``priority`` and ``risk`` are read from each score. Each candidate's benefit
    is :func:`benefit` of its risk and its ``transitive_dependents`` feature, and its return on effort is
    benefit per hour. :func:`~tdp_core.planning.knapsack.knapsack` chooses the files and
    :func:`~tdp_core.planning.schedule.schedule` orders them by the import edges among them.

    ``features`` is either indexed by path (the ``AnalysisResult`` shape) or has a ``path`` column (the
    ``BuildResult`` shape); it needs a ``transitive_dependents`` column. Steps are numbered from 1 in fix
    order and each step's ``cost`` is the very object from ``costs``. ``total_hours`` never exceeds
    ``budget_hours`` beyond float rounding (the costs are rounded up to 3 minute units). An empty selection
    gives ``Plan(budget_hours, (), 0.0, 0.0)``.

    Raises ``ValueError`` for a budget that is not finite and >= 0, repeated score paths, a non-unique feature
    index, or a candidate cost that is not finite; ``KeyError`` when the ``transitive_dependents`` column is
    missing or a candidate has no feature row.
    """
    budget = float(budget_hours)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError(f"budget_hours must be finite and >= 0, got {budget_hours!r}")
    counts = Counter(s.path for s in scores)
    if len(counts) != len(scores):
        repeated = sorted(p for p, n in counts.items() if n > 1)
        raise ValueError(f"duplicate paths in scores: {', '.join(repeated)}")

    frame = features.set_index("path") if "path" in features.columns else features
    if not frame.index.is_unique:
        raise ValueError("the feature frame has repeated paths")
    dependents = dict(zip(frame.index, frame["transitive_dependents"].to_numpy(dtype=float), strict=True))

    items: list[tuple[str, float, float]] = []
    benefits: dict[str, float] = {}
    roi: dict[str, float] = {}
    for score in sorted(scores, key=lambda s: s.path):
        cost = costs.get(score.path)
        if score.priority < min_priority or cost is None:
            continue
        hours = float(cost.hours)
        if not math.isfinite(hours):
            raise ValueError(f"cost hours of {score.path!r} must be finite, got {cost.hours!r}")
        if hours <= 0:
            continue
        if score.path not in dependents:
            raise KeyError(f"no feature row for {score.path!r}")
        value = benefit(score.risk, dependents[score.path])
        items.append((score.path, value, hours))
        benefits[score.path] = value
        roi[score.path] = value / hours

    chosen = knapsack(items, budget)
    steps = tuple(
        PlanStep(
            order=order,
            path=path,
            benefit=benefits[path],
            cost=costs[path],
            roi=roi[path],
            group=group,
            reason=reason,
        )
        for order, (path, group, reason) in enumerate(schedule(chosen, import_graph, roi), start=1)
    )
    total_benefit = math.fsum(step.benefit for step in steps)
    total_hours = math.fsum(float(step.cost.hours) for step in steps)
    logger.debug(
        "plan: %d of %d candidates, %.2f of %.2f hours, benefit %.3f",
        len(steps),
        len(items),
        total_hours,
        budget,
        total_benefit,
    )
    return Plan(budget_hours=budget, steps=steps, total_benefit=total_benefit, total_hours=total_hours)


def import_graph_from_edges(edges: Iterable[GraphEdge]) -> nx.DiGraph[str]:
    """Rebuild an import graph from stored graph edges (for example ``GraphData.edges``).

    Keeps only edges of kind ``"import"`` (``source`` imports ``target``) and drops self loops; nodes are the
    files those edges touch. Pass the result to :func:`make_plan`.
    """
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_edges_from(
        sorted({(e.source, e.target) for e in edges if e.kind == "import" and e.source != e.target})
    )
    return graph

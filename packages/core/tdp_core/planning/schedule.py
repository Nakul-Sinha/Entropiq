"""Fix order for the chosen files: dependencies first, import cycles together, best return on effort next.

An import edge ``a -> b`` means ``a`` imports ``b``, so ``b`` is fixed before ``a``. Files in one import cycle
(a strongly connected component of the chosen files) form one group and are fixed together.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Final

import networkx as nx

logger = logging.getLogger(__name__)

REASON_CYCLE: Final = "fix together with {others} (dependency cycle)"
REASON_BEFORE_DEPENDENT: Final = "fix before {dependent}, which depends on it"
REASON_BEST_ROI: Final = "best return on effort among the remaining files"
MAX_LISTED_PATHS: Final = 3
"""Most paths named in a cycle reason; the rest are counted (``and 2 more``)."""


def schedule(
    chosen: Sequence[str], import_graph: nx.DiGraph[str], roi: Mapping[str, float]
) -> list[tuple[str, int, str]]:
    """Order ``chosen`` for fixing and return ``(path, group, reason)`` tuples in fix order.

    Only the import edges between chosen files count (an edge ``a -> b``: ``a`` imports ``b``, so ``b`` comes
    first); edges through files outside ``chosen`` and self loops are ignored, and a chosen file missing from
    ``import_graph`` is scheduled as an isolated file. Each import cycle among the chosen files is one group;
    groups are numbered 1..k in fix order and the members of a group are adjacent, ordered by ``roi``
    descending, then path. Among the groups whose dependencies are all fixed, the one with the highest member
    ``roi`` goes next (ties: the smallest member path), so the result does not depend on the input order.

    Reasons, first match wins: a group of several files gives ``REASON_CYCLE`` naming the other members in
    plan order (at most ``MAX_LISTED_PATHS``, then ``and N more``); a file that chosen files import gives
    ``REASON_BEFORE_DEPENDENT`` naming the importer that comes first in the plan; otherwise
    ``REASON_BEST_ROI``.

    Raises ``ValueError`` for a repeated path in ``chosen`` or a non-finite ``roi``, and ``KeyError`` for a
    chosen path missing from ``roi``. An empty ``chosen`` gives ``[]``.
    """
    nodes = sorted(set(chosen))
    if len(nodes) != len(chosen):
        repeated = sorted(path for path, n in Counter(chosen).items() if n > 1)
        raise ValueError(f"duplicate paths in schedule input: {', '.join(repeated)}")
    if not nodes:
        return []
    value: dict[str, float] = {}
    for path in nodes:
        r = float(roi[path])
        if not math.isfinite(r):
            raise ValueError(f"roi of {path!r} must be finite, got {r!r}")
        value[path] = r

    sub: nx.DiGraph[str] = nx.DiGraph()
    sub.add_nodes_from(nodes)
    sub.add_edges_from(sorted((u, v) for u, v in import_graph.subgraph(nodes).edges() if u != v))

    cond = nx.condensation(sub)
    members: dict[int, list[str]] = {}
    key: dict[int, tuple[float, str]] = {}
    for gid in cond.nodes:
        group = sorted(cond.nodes[gid]["members"], key=lambda p: (-value[p], p))
        members[gid] = group
        key[gid] = (-value[group[0]], min(group))

    waiting = {gid: len(cond.succ[gid]) for gid in cond.nodes}
    heap = [(key[gid], gid) for gid, n in waiting.items() if n == 0]
    heapq.heapify(heap)
    groups: list[list[str]] = []
    while heap:
        _, gid = heapq.heappop(heap)
        groups.append(members[gid])
        for importer in cond.predecessors(gid):
            waiting[importer] -= 1
            if waiting[importer] == 0:
                heapq.heappush(heap, (key[importer], importer))

    position = {path: i for i, path in enumerate(path for group in groups for path in group)}
    out: list[tuple[str, int, str]] = []
    for number, group in enumerate(groups, start=1):
        for path in group:
            out.append((path, number, _reason(path, group, sub, position)))
    logger.debug("scheduled %d files in %d groups", len(out), len(groups))
    return out


def _reason(path: str, group: Sequence[str], sub: nx.DiGraph[str], position: Mapping[str, int]) -> str:
    if len(group) > 1:
        return REASON_CYCLE.format(others=_list_paths([p for p in group if p != path]))
    dependents = list(sub.predecessors(path))
    if dependents:
        return REASON_BEFORE_DEPENDENT.format(dependent=min(dependents, key=position.__getitem__))
    return REASON_BEST_ROI


def _list_paths(paths: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``, or ``a, b, c and 2 more``."""
    if len(paths) > MAX_LISTED_PATHS:
        return f"{', '.join(paths[:MAX_LISTED_PATHS])} and {len(paths) - MAX_LISTED_PATHS} more"
    if len(paths) == 1:
        return paths[0]
    return f"{', '.join(paths[:-1])} and {paths[-1]}"

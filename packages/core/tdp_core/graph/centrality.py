"""Impact features from the import graph (family I, plus ``test_importers`` of family T).

All features except ``test_importers`` are computed on the non-test subgraph: the nodes whose ``is_test``
attribute is false and the edges between them. Edges point from importer to imported file.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

from tdp_core.config import DEFAULT_SETTINGS, Settings

logger = logging.getLogger(__name__)

GRAPH_FEATURE_COLUMNS: tuple[str, ...] = (
    "fan_in",
    "fan_out",
    "transitive_dependents",
    "pagerank",
    "betweenness",
    "scc_size",
    "test_importers",
)
_FLOAT_COLUMNS = frozenset({"pagerank", "betweenness"})

PAGERANK_ALPHA = 0.85
BETWEENNESS_SAMPLE_ABOVE = 2000  # exact betweenness up to this many non-test nodes
BETWEENNESS_SAMPLES = 500  # source nodes sampled above the threshold


def graph_features(G: nx.DiGraph[str], settings: Settings = DEFAULT_SETTINGS) -> pd.DataFrame:
    """Per-file graph features for the non-test files of an import graph.

    ``G`` is the output of :func:`tdp_core.graph.imports.build_import_graph`: nodes are repo-relative paths
    with a boolean ``is_test`` attribute (missing means false), edges go importer -> imported.

    Returns a DataFrame indexed by the non-test paths (sorted, index name ``"path"``) with the columns of
    :data:`GRAPH_FEATURE_COLUMNS`:

    - ``fan_in`` (int64): number of non-test files importing the file.
    - ``fan_out`` (int64): number of non-test files the file imports.
    - ``transitive_dependents`` (int64): number of other non-test files that can reach the file along
      import edges, counted on the condensation DAG (members of its own import cycle are included).
    - ``pagerank`` (float64): ``nx.pagerank(alpha=0.85)`` on the non-test graph, so rank flows to imported
      modules; sums to 1 over the returned rows.
    - ``betweenness`` (float64): normalised betweenness centrality in 0..1, exact up to 2,000 non-test
      files, above that estimated from ``min(500, n)`` source nodes sampled with ``settings.seed``.
    - ``scc_size`` (int64): size of the strongly connected component holding the file (1 if no cycle).
    - ``test_importers`` (int64): number of test files with an import edge to the file.

    A graph without non-test nodes gives an empty frame with the same columns and dtypes.
    """
    is_test = {node: bool(G.nodes[node].get("is_test", False)) for node in G}
    paths = sorted(node for node in G if not is_test[node])
    index = pd.Index(paths, dtype="str", name="path")
    if not paths:
        return pd.DataFrame(
            {col: np.empty(0, dtype=_dtype(col)) for col in GRAPH_FEATURE_COLUMNS}, index=index
        )

    sub: nx.DiGraph[str] = nx.DiGraph()
    sub.add_nodes_from(paths)
    sub.add_edges_from(sorted((u, v) for u, v in G.edges if not is_test[u] and not is_test[v]))
    n = sub.number_of_nodes()

    test_importers = dict.fromkeys(paths, 0)
    for u, v in G.edges:
        if is_test[u] and not is_test[v]:
            test_importers[v] += 1

    scc_size, dependents = _cycles_and_dependents(sub)
    pagerank = nx.pagerank(sub, alpha=PAGERANK_ALPHA)
    k = min(BETWEENNESS_SAMPLES, n) if n > BETWEENNESS_SAMPLE_ABOVE else None
    betweenness = nx.betweenness_centrality(sub, k=k, normalized=True, seed=settings.seed)
    logger.debug("graph features: %d non-test files, %d edges, betweenness k=%s", n, sub.number_of_edges(), k)

    columns = {
        "fan_in": [sub.in_degree(p) for p in paths],
        "fan_out": [sub.out_degree(p) for p in paths],
        "transitive_dependents": [dependents[p] for p in paths],
        "pagerank": [pagerank[p] for p in paths],
        "betweenness": [betweenness[p] for p in paths],
        "scc_size": [scc_size[p] for p in paths],
        "test_importers": [test_importers[p] for p in paths],
    }
    return pd.DataFrame(
        {col: np.asarray(columns[col], dtype=_dtype(col)) for col in GRAPH_FEATURE_COLUMNS}, index=index
    )


def _dtype(column: str) -> type[np.generic]:
    return np.float64 if column in _FLOAT_COLUMNS else np.int64


def _cycles_and_dependents(graph: nx.DiGraph[str]) -> tuple[dict[str, int], dict[str, int]]:
    """SCC size and transitive dependent count for every node of ``graph``.

    Works on the condensation DAG in topological order, keeping for each component a bitset (Python int)
    of the original nodes that can reach it, so the cost is about ``edges * nodes / 64`` word operations.
    """
    condensed = nx.condensation(graph)
    position = {node: i for i, node in enumerate(graph)}
    members: dict[int, list[str]] = {c: sorted(condensed.nodes[c]["members"]) for c in condensed}
    member_bits: dict[int, int] = {}
    for c, nodes in members.items():
        bits = 0
        for node in nodes:
            bits |= 1 << position[node]
        member_bits[c] = bits

    reaching: dict[int, int] = {}
    for c in nx.topological_sort(condensed):
        bits = 0
        for pred in condensed.predecessors(c):
            bits |= reaching[pred] | member_bits[pred]
        reaching[c] = bits

    scc_size: dict[str, int] = {}
    dependents: dict[str, int] = {}
    for c, nodes in members.items():
        outside = reaching[c].bit_count()
        for node in nodes:
            scc_size[node] = len(nodes)
            dependents[node] = outside + len(nodes) - 1
    return scc_size, dependents

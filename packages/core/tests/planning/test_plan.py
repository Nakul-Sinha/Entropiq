from __future__ import annotations

import json
import math
import random
import sys
import time
from collections.abc import Iterable, Mapping

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import (
    FAMILY_ORDER,
    CostEstimate,
    FileScore,
    GraphData,
    GraphEdge,
    Plan,
    PlanStep,
    band_for,
)
from tdp_core.planning.knapsack import capacity_units, cost_units
from tdp_core.planning.plan import benefit, import_graph_from_edges, make_plan
from tdp_core.planning.schedule import REASON_BEST_ROI
from tdp_core.serialize import from_jsonable, to_jsonable


def traced() -> bool:
    """True when coverage or a debugger slows the interpreter down."""
    return sys.gettrace() is not None or sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID) is not None


def score(path: str, priority: int, risk: float) -> FileScore:
    return FileScore(
        path=path,
        score=float(priority) / 10.0,
        priority=priority,
        band=band_for(priority),
        risk=risk,
        risk_low=max(0.0, risk - 0.1),
        risk_high=min(1.0, risk + 0.1),
        reasons=(),
        mitigations=(),
        debt_types=(),
        shap_by_family=dict.fromkeys(FAMILY_ORDER, 0.0),
    )


def cost(hours: float) -> CostEstimate:
    return CostEstimate(
        hours=hours, low=0.8 * hours, high=1.2 * hours, breakdown={"complexity": 60.0 * hours}
    )


def frame(dependents: Mapping[str, float]) -> pd.DataFrame:
    """A path-indexed float32 feature frame, as in ``AnalysisResult.features``."""
    paths = sorted(dependents)
    return pd.DataFrame(
        {
            "fan_in": np.ones(len(paths)),
            "transitive_dependents": [dependents[p] for p in paths],
            "sloc": np.full(len(paths), 100.0),
        },
        index=pd.Index(paths, name="path"),
        dtype=np.float32,
    )


def digraph(edges: Iterable[tuple[str, str]]) -> nx.DiGraph[str]:
    g: nx.DiGraph[str] = nx.DiGraph()
    g.add_edges_from(edges)
    return g


SCORES = [
    score("low.py", 40, 0.9),
    score("nocost.py", 80, 0.7),
    score("free.py", 70, 0.6),
    score("core.py", 90, 0.5),
    score("app.py", 60, 0.8),
]
COSTS = {"low.py": cost(1.0), "free.py": cost(0.0), "core.py": cost(2.0), "app.py": cost(1.0)}
FEATURES = frame({"low.py": 0, "nocost.py": 5, "free.py": 1, "core.py": 3, "app.py": 0})
GRAPH = digraph([("app.py", "core.py"), ("low.py", "app.py"), ("tests/test_core.py", "core.py")])


@pytest.mark.parametrize(
    ("risk", "dependents", "expected"),
    [(0.5, 0, 0.5), (0.5, 1, 1.0), (0.5, 3, 1.5), (0.2, 7, 0.8), (0.0, 100, 0.0), (1.0, 0, 1.0)],
)
def test_benefit(risk: float, dependents: float, expected: float) -> None:
    value = benefit(risk, dependents)
    assert value == pytest.approx(expected)
    assert type(value) is float


def test_benefit_accepts_numpy_scalars() -> None:
    value = benefit(np.float32(0.5), np.float32(3.0))  # type: ignore[arg-type]
    assert type(value) is float
    assert value == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("risk", "dependents"),
    [(-0.1, 0), (1.1, 0), (math.nan, 0), (0.5, -1), (0.5, math.nan), (0.5, math.inf)],
)
def test_benefit_rejects_bad_values(risk: float, dependents: float) -> None:
    with pytest.raises(ValueError):
        benefit(risk, dependents)


def test_handcrafted_plan() -> None:
    # core.py: 0.5 * (1 + log2(1 + 3)) = 1.5 over 2 h; app.py imports core.py: 0.8 * 1 = 0.8 over 1 h
    plan = make_plan(SCORES, FEATURES, COSTS, GRAPH, 5.0)
    assert plan.steps == (
        PlanStep(1, "core.py", 1.5, COSTS["core.py"], 0.75, 1, "fix before app.py, which depends on it"),
        PlanStep(2, "app.py", 0.8, COSTS["app.py"], 0.8, 2, REASON_BEST_ROI),
    )
    assert plan.budget_hours == 5.0
    assert plan.total_hours == 3.0
    assert plan.total_benefit == pytest.approx(2.3)
    for step in plan.steps:
        assert step.cost is COSTS[step.path]
        assert step.roi == step.benefit / step.cost.hours
    assert all(type(v) is float for v in (plan.budget_hours, plan.total_benefit, plan.total_hours))
    assert all(type(s.benefit) is float and type(s.roi) is float for s in plan.steps)


def test_tight_budget_picks_the_better_file() -> None:
    plan = make_plan(SCORES, FEATURES, COSTS, GRAPH, 2.5)
    assert [(s.order, s.path, s.group, s.reason) for s in plan.steps] == [(1, "core.py", 1, REASON_BEST_ROI)]
    assert plan.total_hours == 2.0
    assert plan.total_benefit == pytest.approx(1.5)


def test_min_priority_is_respected() -> None:
    assert DEFAULT_SETTINGS.min_plan_priority == 50
    everything = make_plan(SCORES, FEATURES, COSTS, GRAPH, 10.0, min_priority=0)
    assert [s.path for s in everything.steps] == ["core.py", "app.py", "low.py"]
    assert everything.steps[1].reason == "fix before low.py, which depends on it"
    critical = make_plan(SCORES, FEATURES, COSTS, GRAPH, 10.0, min_priority=90)
    assert [s.path for s in critical.steps] == ["core.py"]


def test_budget_below_every_candidate_gives_an_empty_plan() -> None:
    assert make_plan(SCORES, FEATURES, COSTS, GRAPH, 0.5) == Plan(0.5, (), 0.0, 0.0)
    assert make_plan(SCORES, FEATURES, COSTS, GRAPH, 0) == Plan(0.0, (), 0.0, 0.0)
    assert make_plan([], FEATURES, {}, GRAPH, 40.0) == Plan(40.0, (), 0.0, 0.0)


def test_path_column_and_shuffled_scores_give_the_same_plan() -> None:
    expected = make_plan(SCORES, FEATURES, COSTS, GRAPH, 5.0)
    with_column = FEATURES.reset_index()
    assert "path" in with_column.columns
    assert make_plan(SCORES, with_column, COSTS, GRAPH, 5.0) == expected
    shuffled = list(SCORES)
    rng = random.Random(5)
    for _ in range(5):
        rng.shuffle(shuffled)
        assert make_plan(shuffled, FEATURES.sample(frac=1.0, random_state=1), COSTS, GRAPH, 5.0) == expected


def test_invalid_input() -> None:
    with pytest.raises(ValueError, match=r"core\.py"):
        make_plan([*SCORES, score("core.py", 10, 0.1)], FEATURES, COSTS, GRAPH, 5.0)
    for budget in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="budget_hours"):
            make_plan(SCORES, FEATURES, COSTS, GRAPH, budget)
    with pytest.raises(ValueError, match=r"core\.py"):
        make_plan(SCORES, FEATURES, {**COSTS, "core.py": cost(math.nan)}, GRAPH, 5.0)
    with pytest.raises(ValueError, match="repeated"):
        make_plan(SCORES, pd.concat([FEATURES, FEATURES.iloc[:1]]), COSTS, GRAPH, 5.0)
    with pytest.raises(KeyError):
        make_plan(SCORES, FEATURES.drop(columns="transitive_dependents"), COSTS, GRAPH, 5.0)
    with pytest.raises(KeyError, match=r"app\.py"):
        make_plan(SCORES, FEATURES.drop(index="app.py"), COSTS, GRAPH, 5.0)
    with pytest.raises(ValueError, match="risk"):
        make_plan([score("core.py", 90, 1.5)], FEATURES, COSTS, GRAPH, 5.0)


def test_files_outside_the_candidates_need_no_feature_row() -> None:
    plan = make_plan(SCORES, FEATURES.drop(index=["low.py", "nocost.py", "free.py"]), COSTS, GRAPH, 5.0)
    assert [s.path for s in plan.steps] == ["core.py", "app.py"]


def random_inputs(
    rng: random.Random, n: int
) -> tuple[list[FileScore], pd.DataFrame, dict[str, CostEstimate], nx.DiGraph[str]]:
    paths = [f"pkg/m{i:04d}.py" for i in range(n)]
    scores = [score(p, rng.randint(0, 100), rng.random()) for p in paths]
    costs = {p: cost(rng.uniform(0.5, 20.0)) for p in paths if rng.random() < 0.95}
    features = frame({p: float(rng.randint(0, 50)) for p in paths})
    g = nx.relabel_nodes(
        nx.gnm_random_graph(n, 3 * n, seed=rng.randint(0, 10_000), directed=True),
        dict(enumerate(paths)),
    )
    return scores, features, costs, g


@pytest.mark.parametrize("seed", range(8))
def test_random_plans_stay_within_budget(seed: int) -> None:
    rng = random.Random(seed)
    scores, features, costs, g = random_inputs(rng, rng.randint(1, 300))
    budget = rng.choice([0.0, 0.5, 3.3, 40.0, 123.45, 5000.0])
    plan = make_plan(scores, features, costs, g, budget)
    assert plan.budget_hours == budget
    assert plan.total_hours <= budget + 1e-9
    assert [s.order for s in plan.steps] == list(range(1, len(plan.steps) + 1))
    assert plan.total_hours == math.fsum(s.cost.hours for s in plan.steps)
    assert plan.total_benefit == math.fsum(s.benefit for s in plan.steps)
    assert sum(cost_units(s.cost.hours) for s in plan.steps) <= capacity_units(budget)
    by_path = {s.path: s for s in scores}
    groups = [s.group for s in plan.steps]
    assert groups == sorted(groups)
    for step in plan.steps:
        assert by_path[step.path].priority >= DEFAULT_SETTINGS.min_plan_priority
        assert step.roi == pytest.approx(step.benefit / step.cost.hours)
        assert step.cost is costs[step.path]


def test_plan_round_trips_through_json() -> None:
    plan = make_plan(SCORES, FEATURES, COSTS, GRAPH, 5.0)
    data = to_jsonable(plan)
    assert from_jsonable(Plan, json.loads(json.dumps(data))) == plan
    assert from_jsonable(Plan, to_jsonable(Plan(1.0, (), 0.0, 0.0))) == Plan(1.0, (), 0.0, 0.0)


def test_import_graph_from_edges() -> None:
    edges = (
        GraphEdge("app.py", "core.py", "import", 1.0),
        GraphEdge("app.py", "app.py", "import", 1.0),
        GraphEdge("core.py", "util.py", "hidden_coupling", 0.8),
        GraphEdge("util.py", "core.py", "import", 1.0),
        GraphEdge("app.py", "core.py", "import", 1.0),
    )
    g = import_graph_from_edges(from_jsonable(GraphData, to_jsonable(GraphData((), edges))).edges)
    assert sorted(g.edges()) == [("app.py", "core.py"), ("util.py", "core.py")]
    assert sorted(g.nodes()) == ["app.py", "core.py", "util.py"]
    assert make_plan(SCORES, FEATURES, COSTS, g, 5.0) == make_plan(SCORES, FEATURES, COSTS, GRAPH, 5.0)
    assert import_graph_from_edges([]).number_of_nodes() == 0


def test_five_thousand_files_within_two_seconds() -> None:
    rng = random.Random(42)
    n = 5000
    paths = [f"src/pkg{i % 50:02d}/mod{i:04d}.py" for i in range(n)]
    g = nx.relabel_nodes(nx.gnm_random_graph(n, 15_000, seed=42, directed=True), dict(enumerate(paths)))
    scores = [score(p, rng.randint(0, 100), rng.random()) for p in paths]
    costs = {p: cost(rng.uniform(0.5, 20.0)) for p in paths}
    features = frame({p: float(rng.randint(0, 200)) for p in paths})
    start = time.perf_counter()
    plan = make_plan(scores, features, costs, g, 200.0)
    elapsed = time.perf_counter() - start
    assert plan.steps
    assert plan.total_hours <= 200.0
    assert len({s.path for s in plan.steps}) == len(plan.steps)
    if not traced():
        assert elapsed < 2.0, f"{elapsed:.2f} s"

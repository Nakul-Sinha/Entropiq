from __future__ import annotations

import logging
import math
import random
import sys
import time
from collections.abc import Sequence
from unittest import mock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdp_core.planning import knapsack as knapsack_module
from tdp_core.planning.knapsack import DEFAULT_STEP_HOURS, capacity_units, cost_units, knapsack

Item = tuple[str, float, float]

BENEFITS = st.one_of(
    st.integers(0, 3).map(float),
    st.floats(0, 10, allow_nan=False, allow_infinity=False, allow_subnormal=False),
)
HOURS = st.floats(0, 15, allow_nan=False, allow_infinity=False, allow_subnormal=False)
BUDGETS = st.floats(0, 40, allow_nan=False, allow_infinity=False, allow_subnormal=False)


def traced() -> bool:
    """True when coverage or a debugger slows the interpreter down."""
    return sys.gettrace() is not None or sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID) is not None


def named(pairs: Sequence[tuple[float, float]]) -> list[Item]:
    return [(f"f{i:02d}.py", benefit, hours) for i, (benefit, hours) in enumerate(pairs)]


def subsets(n: int) -> np.ndarray:
    """Every subset of ``n`` items as a 0/1 matrix with one row per subset."""
    return (np.arange(1 << n)[:, None] >> np.arange(n)) & 1


def brute_force(items: Sequence[Item], allowed: np.ndarray) -> float:
    """Largest total benefit over the subsets marked in ``allowed`` (one flag per subset)."""
    if not items:
        return 0.0
    values = subsets(len(items)) @ np.array([b for _, b, _ in items], dtype=float)
    return float(values[allowed].max())


def unit_optimum(items: Sequence[Item], budget: float, step: float) -> float:
    if not items:
        return 0.0
    units = np.array([cost_units(c, step) for _, _, c in items])
    return brute_force(items, subsets(len(items)) @ units <= capacity_units(budget, step))


def value(items: Sequence[Item], chosen: Sequence[str]) -> float:
    by_path = {p: b for p, b, _ in items}
    return math.fsum(by_path[p] for p in chosen)


def hours(items: Sequence[Item], chosen: Sequence[str]) -> float:
    by_path = {p: c for p, _, c in items}
    return math.fsum(by_path[p] for p in chosen)


@pytest.mark.parametrize(
    ("cost", "step", "units"),
    [(1.0, 0.5, 2), (1.01, 0.5, 3), (0.0, 0.5, 1), (0.5, 0.05, 10), (0.3, 0.1, 3), (10.3, 0.05, 206)],
)
def test_cost_units(cost: float, step: float, units: int) -> None:
    assert cost_units(cost, step) == units


def test_cost_units_default_step() -> None:
    assert DEFAULT_STEP_HOURS == 0.05
    assert cost_units(0.5) == 10
    assert cost_units(0.51) == 11


@pytest.mark.parametrize(
    ("budget", "step", "units"),
    [(40, 0.05, 800), (0.3, 0.1, 3), (0.04, 0.05, 0), (0, 0.5, 0), (200, 0.05, 4000)],
)
def test_capacity_units(budget: float, step: float, units: int) -> None:
    assert capacity_units(budget, step) == units


@pytest.mark.parametrize("bad_step", [0.0, -0.5, math.nan, math.inf, 1e-320])
def test_unit_helpers_reject_bad_steps(bad_step: float) -> None:
    with pytest.raises(ValueError):
        cost_units(1.0, bad_step)
    with pytest.raises(ValueError):
        capacity_units(1.0, bad_step)


@pytest.mark.parametrize("bad_hours", [-0.1, math.nan, math.inf, -math.inf])
def test_unit_helpers_reject_bad_hours(bad_hours: float) -> None:
    with pytest.raises(ValueError, match="cost_hours"):
        cost_units(bad_hours)
    with pytest.raises(ValueError, match="budget_hours"):
        capacity_units(bad_hours)


@settings(max_examples=300, deadline=None, derandomize=True)
@given(
    pairs=st.lists(st.tuples(BENEFITS, HOURS), max_size=12),
    budget=BUDGETS,
    step=st.sampled_from([0.05, 0.25, 0.5, 1.0]),
)
def test_matches_brute_force_on_units(pairs: list[tuple[float, float]], budget: float, step: float) -> None:
    items = named(pairs)
    chosen = knapsack(items, budget, step)
    assert chosen == sorted(set(chosen))
    assert math.isclose(value(items, chosen), unit_optimum(items, budget, step), rel_tol=1e-9, abs_tol=1e-9)
    units = {p: cost_units(c, step) for p, _, c in items}
    assert sum(units[p] for p in chosen) <= capacity_units(budget, step)
    assert hours(items, chosen) <= budget + 1e-6
    assert all(b > 0 for p, b, _ in items if p in chosen)


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    pairs=st.lists(st.tuples(BENEFITS, st.integers(1, 20)), max_size=12),
    budget=BUDGETS,
    step=st.sampled_from([0.25, 0.5, 1.0]),
)
def test_exact_on_grid_costs(pairs: list[tuple[float, int]], budget: float, step: float) -> None:
    items = named([(benefit, k * step) for benefit, k in pairs])
    chosen = knapsack(items, budget, step)
    if not items:
        assert chosen == []
        return
    true_hours = subsets(len(items)) @ np.array([c for _, _, c in items])
    optimum = brute_force(items, true_hours <= budget + 1e-9 * step)
    assert math.isclose(value(items, chosen), optimum, rel_tol=1e-9, abs_tol=1e-9)
    assert hours(items, chosen) <= budget + 1e-9


def test_budget_below_every_item() -> None:
    items = [("a.py", 1.0, 2.0), ("b.py", 5.0, 3.0), ("c.py", 0.5, 1.5)]
    assert knapsack(items, 1.0) == []
    assert knapsack(items, 1.0, step=0.5) == []


def test_empty_inputs() -> None:
    assert knapsack([], 40.0) == []
    assert knapsack([("a.py", 1.0, 0.5)], 0.0) == []
    assert knapsack([("a.py", 0.0, 0.5)], 40.0) == []


def test_everything_fits_except_worthless_items() -> None:
    items = [("c.py", 1.0, 2.0), ("a.py", 0.0, 1.0), ("b.py", 3.0, 0.5), ("d.py", 0.2, 0.0)]
    assert knapsack(items, 40.0) == ["b.py", "c.py", "d.py"]


def test_oversized_item_is_dropped_before_the_search() -> None:
    items = [("big.py", 100.0, 50.0), ("a.py", 1.0, 3.0), ("b.py", 2.0, 3.0), ("c.py", 1.5, 3.0)]
    assert knapsack(items, 6.0) == ["b.py", "c.py"]


@pytest.mark.parametrize("step", [0.5, DEFAULT_STEP_HOURS])
def test_rounding_never_overfills_the_budget(step: float) -> None:
    # round(0.6 / 0.5) = 1 unit each would take both files (1.2 hours) for a 1 hour budget
    items = [("a.py", 1.0, 0.6), ("b.py", 1.0, 0.6)]
    chosen = knapsack(items, 1.0, step)
    assert chosen == ["a.py"]
    assert hours(items, chosen) <= 1.0


def test_ties_go_to_the_first_path_whatever_the_input_order() -> None:
    items = [("b.py", 2.0, 1.0), ("a.py", 2.0, 1.0), ("c.py", 1.0, 1.0)]
    assert knapsack(items, 1.0) == ["a.py"]
    rng = random.Random(7)
    for _ in range(5):
        rng.shuffle(items)
        assert knapsack(items, 1.0) == ["a.py"]
        assert knapsack(items, 2.0) == ["a.py", "b.py"]


def test_exact_search_beats_greedy() -> None:
    # benefit per hour favours a.py, but b.py and c.py together are worth more
    items = [("a.py", 6.0, 2.5), ("b.py", 5.0, 2.0), ("c.py", 5.0, 2.0)]
    assert knapsack(items, 4.0) == ["b.py", "c.py"]


def test_greedy_falls_back_to_the_best_single_item(caplog: pytest.LogCaptureFixture) -> None:
    items = [("a.py", 2.0, 0.5), ("b.py", 10.0, 5.0)]
    with (
        mock.patch.object(knapsack_module, "MAX_DP_CELLS", 0),
        caplog.at_level(logging.INFO, logger="tdp_core.planning.knapsack"),
    ):
        assert knapsack(items, 5.0, step=0.5) == ["b.py"]
    assert "greedy" in caplog.text


def test_greedy_skips_items_that_do_not_fit() -> None:
    items = [("y.py", 5.0, 2.5), ("x.py", 9.0, 4.0), ("z.py", 1.8, 1.0)]
    with mock.patch.object(knapsack_module, "MAX_DP_CELLS", 0):
        assert knapsack(items, 5.0, step=0.5) == ["x.py", "z.py"]


@pytest.mark.parametrize("scale", [1e5, 1e6])
def test_huge_budgets_use_greedy_without_allocating(scale: float, caplog: pytest.LogCaptureFixture) -> None:
    # 1e5: 9e6 cells but 3e6 capacity units (over MAX_DP_CAPACITY); 1e6: 9e7 cells (over MAX_DP_CELLS)
    items = [("a.py", 1.0, scale), ("b.py", 2.0, scale), ("c.py", 0.5, 10.0)]
    with caplog.at_level(logging.INFO, logger="tdp_core.planning.knapsack"):
        assert knapsack(items, 1.5 * scale) == ["b.py", "c.py"]
    assert "greedy" in caplog.text


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    pairs=st.lists(st.tuples(BENEFITS, HOURS), max_size=12),
    budget=BUDGETS,
    step=st.sampled_from([0.25, 0.5, 1.0]),
)
def test_greedy_keeps_half_the_optimum(pairs: list[tuple[float, float]], budget: float, step: float) -> None:
    items = named(pairs)
    with mock.patch.object(knapsack_module, "MAX_DP_CELLS", 0):
        chosen = knapsack(items, budget, step)
    assert chosen == sorted(set(chosen))
    assert value(items, chosen) >= 0.5 * unit_optimum(items, budget, step) - 1e-9
    units = {p: cost_units(c, step) for p, _, c in items}
    assert sum(units[p] for p in chosen) <= capacity_units(budget, step)
    assert hours(items, chosen) <= budget + 1e-6


@pytest.mark.parametrize(
    ("items", "budget", "step"),
    [
        ([("a.py", 1.0, 1.0)], 5.0, 0.0),
        ([("a.py", 1.0, 1.0)], 5.0, math.nan),
        ([("a.py", 1.0, 1.0)], -1.0, 0.5),
        ([("a.py", 1.0, 1.0)], math.nan, 0.5),
        ([("a.py", 1.0, 1.0)], math.inf, 0.5),
        ([], -1.0, 0.5),
        ([("a.py", 1.0, 1.0), ("a.py", 2.0, 1.0)], 5.0, 0.5),
        ([("a.py", -1.0, 1.0)], 5.0, 0.5),
        ([("a.py", math.nan, 1.0)], 5.0, 0.5),
        ([("a.py", math.inf, 1.0)], 5.0, 0.5),
        ([("a.py", 1.0, -1.0)], 5.0, 0.5),
        ([("a.py", 1.0, math.inf)], 5.0, 0.5),
        ([("a.py", 0.0, math.nan)], 5.0, 0.5),
    ],
)
def test_invalid_input_raises(items: list[Item], budget: float, step: float) -> None:
    with pytest.raises(ValueError):
        knapsack(items, budget, step)


def test_duplicate_path_is_named() -> None:
    with pytest.raises(ValueError, match=r"dup\.py"):
        knapsack([("dup.py", 1.0, 1.0), ("ok.py", 1.0, 1.0), ("dup.py", 0.0, 9.0)], 5.0)


def test_five_thousand_items_within_two_seconds() -> None:
    rng = random.Random(42)
    items = [(f"f{i:04d}.py", rng.uniform(0, 3), rng.uniform(0.5, 20)) for i in range(5000)]
    start = time.perf_counter()
    chosen = knapsack(items, 200.0)
    elapsed = time.perf_counter() - start
    assert chosen
    assert hours(items, chosen) <= 200.0
    assert sum(cost_units(c) for p, _, c in items if p in set(chosen)) <= capacity_units(200.0)
    if not traced():
        assert elapsed < 2.0, f"{elapsed:.2f} s"

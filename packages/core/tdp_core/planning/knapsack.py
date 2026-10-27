"""0/1 knapsack: the set of files with the largest total benefit whose fix cost fits a time budget.

Costs and the budget are measured in whole units of ``step`` hours. Item costs round **up** and the budget
rounds **down**, so the true hours of the chosen files never exceed the budget (beyond float rounding): plain
rounding would let two 0.6 hour files into a 1 hour budget. The search is an exact dynamic programme over
capacity units; inputs too large for it fall back to a greedy method that keeps at least half of the optimum.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_STEP_HOURS: Final = 0.05
"""Capacity unit in hours (3 minutes)."""
MAX_DP_CELLS: Final = 50_000_000
"""Largest ``items * (capacity + 1)`` table solved exactly (one byte per cell)."""
MAX_DP_CAPACITY: Final = 1_000_000
"""Largest capacity in units solved exactly (keeps each float array of the programme under 8 MB)."""
_UNIT_TOLERANCE: Final = 1e-9
"""Slack in units, so float noise (``40 / 0.05``, ``0.3 / 0.1``) never gains or loses a unit."""


def cost_units(cost_hours: float, step: float = DEFAULT_STEP_HOURS) -> int:
    """Capacity units an item of ``cost_hours`` occupies: ``max(1, ceil(cost_hours / step))``.

    Rounds up (with a 1e-9 unit tolerance for float noise), so the units never understate the true cost; a
    free item still takes one unit. Raises ``ValueError`` unless ``step`` is finite and > 0 and ``cost_hours``
    is finite and >= 0.
    """
    return max(1, math.ceil(_ratio("cost_hours", cost_hours, step) - _UNIT_TOLERANCE))


def capacity_units(budget_hours: float, step: float = DEFAULT_STEP_HOURS) -> int:
    """Whole units of ``step`` hours that fit in ``budget_hours``: ``floor(budget_hours / step)``.

    Rounds down (with a 1e-9 unit tolerance for float noise). Raises ``ValueError`` unless ``step`` is finite
    and > 0 and ``budget_hours`` is finite and >= 0.
    """
    return math.floor(_ratio("budget_hours", budget_hours, step) + _UNIT_TOLERANCE)


def knapsack(
    items: Sequence[tuple[str, float, float]], budget_hours: float, step: float = DEFAULT_STEP_HOURS
) -> list[str]:
    """Paths of the subset with the largest total benefit whose cost fits ``budget_hours``, sorted ascending.

    ``items`` are ``(path, benefit, cost_hours)``. Costs and the budget are converted with :func:`cost_units`
    and :func:`capacity_units`, so the true cost of the result never exceeds the budget by more than float
    rounding. Items with zero benefit, and items that alone exceed the budget, are never chosen. When
    everything fits, every item with a positive benefit is returned. Otherwise an exact dynamic programme runs
    over items in path order, taking an item only on a strict improvement, so ties go to the paths that sort
    first and the result does not depend on the input order. When ``items * (capacity + 1)`` exceeds
    ``MAX_DP_CELLS`` (or the capacity exceeds ``MAX_DP_CAPACITY``), a greedy method is used instead (benefit
    per unit, plus the best single item), which keeps at least half of the optimum.

    Raises ``ValueError`` for a step that is not finite and > 0, a budget, benefit or cost that is not finite
    and >= 0, or a repeated path. Empty ``items`` or a zero budget give ``[]``.
    """
    cap = capacity_units(budget_hours, step)
    seen: set[str] = set()
    parsed: list[tuple[str, float, int]] = []
    for path, benefit, cost in items:
        if path in seen:
            raise ValueError(f"duplicate path in knapsack items: {path!r}")
        seen.add(path)
        value = float(benefit)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"benefit of {path!r} must be finite and >= 0, got {benefit!r}")
        parsed.append((path, value, cost_units(cost, step)))

    work = sorted((item for item in parsed if item[1] > 0 and item[2] <= cap), key=lambda item: item[0])
    if not work:
        return []
    if sum(units for _, _, units in work) <= cap:
        return [path for path, _, _ in work]
    if len(work) * (cap + 1) > MAX_DP_CELLS or cap > MAX_DP_CAPACITY:
        logger.info(
            "knapsack: %d items x %d capacity units is too large for the exact search; using greedy",
            len(work),
            cap,
        )
        return _greedy(work, cap)
    return _exact(work, cap)


def _ratio(name: str, hours: float, step: float) -> float:
    step_f = float(step)
    if not math.isfinite(step_f) or step_f <= 0:
        raise ValueError(f"step must be finite and > 0, got {step!r}")
    hours_f = float(hours)
    if not math.isfinite(hours_f) or hours_f < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {hours!r}")
    ratio = hours_f / step_f
    if not math.isfinite(ratio):
        raise ValueError(f"{name} / step overflows: {hours!r} / {step!r}")
    return ratio


def _exact(work: Sequence[tuple[str, float, int]], cap: int) -> list[str]:
    """Exact 0/1 programme; ``best[c]`` is the largest benefit using at most ``c`` units."""
    best = np.zeros(cap + 1, dtype=np.float64)
    keep = np.zeros((len(work), cap + 1), dtype=np.bool_)
    for i, (_, benefit, units) in enumerate(work):
        take = best[: cap + 1 - units] + benefit
        rest = best[units:]
        better = take > rest
        keep[i, units:] = better
        np.copyto(rest, take, where=better)
    chosen: list[str] = []
    c = cap
    for i in range(len(work) - 1, -1, -1):
        if keep[i, c]:
            path, _, units = work[i]
            chosen.append(path)
            c -= units
    return sorted(chosen)


def _greedy(work: Sequence[tuple[str, float, int]], cap: int) -> list[str]:
    """Greedy by benefit per unit (skipping items that do not fit), or the best single item if worth more."""
    chosen: list[str] = []
    gained: list[float] = []
    left = cap
    for path, benefit, units in sorted(work, key=lambda item: (-item[1] / item[2], item[0])):
        if units <= left:
            chosen.append(path)
            gained.append(benefit)
            left -= units
    single_path, single_benefit, _ = min(work, key=lambda item: (-item[1], item[0]))
    if single_benefit > math.fsum(gained):
        return [single_path]
    return sorted(chosen)

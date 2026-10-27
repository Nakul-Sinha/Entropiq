"""Remediation cost estimates: how many hours it would take to pay down one file's debt.

The cost model is a flat mapping of nine constants (see ``cost_defaults.yaml`` next to this module, which is
copied into every model bundle as ``cost_model.yaml``). Minutes are added up per issue kind, scaled by a blast
radius multiplier that grows with the number of direct importers, converted to hours and floored at
``min_hours``.
"""

from __future__ import annotations

import functools
import logging
import math
import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml

from tdp_core.contracts import CloneStats, CostEstimate, FileSyntax, FunctionInfo

logger = logging.getLogger(__name__)

COST_DEFAULTS_PATH: Final[Path] = Path(__file__).with_name("cost_defaults.yaml")
COST_MODEL_KEYS: Final[tuple[str, ...]] = (
    "complexity_threshold",
    "complexity_per_point_over",
    "duplication_per_group",
    "duplication_per_line",
    "tests_per_public_function",
    "cycle_per_file",
    "satd_per_item",
    "blast_radius_factor",
    "min_hours",
)
COST_KINDS: Final[tuple[str, ...]] = ("complexity", "duplication", "tests", "cycle", "satd")
LOW_FACTOR: Final = 0.8
HIGH_FACTOR: Final = 1.2
ROW_COLUMNS: Final[tuple[str, ...]] = ("test_importers", "scc_size", "satd_count", "fan_in")


def load_cost_model(path: Path | None = None) -> dict[str, float]:
    """Load a cost model YAML file and return all nine constants as floats.

    ``path=None`` returns a fresh copy of the package defaults. A file is merged over the defaults: keys it
    leaves out keep their default value, an empty file gives the defaults, and unknown keys are ignored with
    one warning. Raises ``ValueError`` when the top level is not a mapping or a known value is not a finite
    number >= 0 (booleans and strings are rejected). ``FileNotFoundError`` and ``yaml.YAMLError`` propagate.
    """
    if path is None:
        return dict(_default_params())
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError(f"cost model {path} must be a mapping, got {type(data).__name__}")
    unknown = sorted(str(key) for key in data if key not in COST_MODEL_KEYS)
    if unknown:
        logger.warning("ignoring unknown cost model keys in %s: %s", path, ", ".join(unknown))
    return _resolve(data)


def public_function_names(functions: Sequence[FunctionInfo]) -> tuple[str, ...]:
    """Sorted distinct qualified names of the public, directly testable functions.

    A function is public when the last segment of its qualified name is a Python identifier that does not
    start with ``_`` (so ``Class.__init__``, ``Class._step`` and the parser's ``<unknown>`` fallback are not
    public). A function nested inside another function (any proper dotted prefix of its name is itself a
    function in the list, as in ``outer.inner`` or ``factory.Impl.run``) is not public either: a closure
    cannot be tested directly. Methods of nested classes (``A.B.method``) are public. Names that repeat (a
    property getter and setter, or ``@overload`` stubs) count once.
    """
    names = {f.name for f in functions}
    public: set[str] = set()
    for name in names:
        parts = name.split(".")
        last = parts[-1]
        if not last.isidentifier() or last.startswith("_"):
            continue
        if any(".".join(parts[:i]) in names for i in range(1, len(parts))):
            continue
        public.add(name)
    return tuple(sorted(public))


def estimate_cost(
    row: pd.Series, syntax: FileSyntax, clones: CloneStats, cost_model: Mapping[str, Any]
) -> CostEstimate:
    """Estimate the hours needed to fix one file.

    ``row`` is the file's feature row: it must hold ``test_importers``, ``scc_size``, ``satd_count`` and
    ``fan_in`` (a missing column raises ``KeyError``; a NaN, infinite or negative value raises
    ``ValueError``). ``cost_model`` is merged over the package defaults, so ``{}`` or a partial mapping works
    and unknown keys are ignored; a known value that is not a finite number >= 0 raises ``ValueError``.

    Minutes per kind: ``complexity`` sums ``max(0, cyclomatic - threshold) * per_point`` over functions
    (module-level code is not costed); ``duplication`` is ``clone_groups * per_group + dup_lines * per_line``;
    ``tests`` charges every public function (see :func:`public_function_names`) only when no test file imports
    the file; ``cycle`` applies when ``scc_size > 1``; ``satd`` is per self-admitted debt comment. The total
    is multiplied by ``1 + blast_radius_factor * log2(1 + fan_in)``, converted to hours and floored at
    ``min_hours``.

    Returns ``CostEstimate(hours, 0.8 * hours, 1.2 * hours, breakdown)``. ``breakdown`` always has the five
    keys of ``COST_KINDS`` (zeros included) and holds raw minutes, before the multiplier and the floor. Every
    value is a built-in ``float``.
    """
    p = _resolve(cost_model)
    values = {col: _row_value(row, col) for col in ROW_COLUMNS}

    threshold = p["complexity_threshold"]
    complexity = math.fsum(
        max(0.0, float(f.cyclomatic) - threshold) * p["complexity_per_point_over"] for f in syntax.functions
    )
    duplication = (
        float(clones.clone_groups) * p["duplication_per_group"]
        + float(clones.dup_lines) * p["duplication_per_line"]
    )
    tests = (
        len(public_function_names(syntax.functions)) * p["tests_per_public_function"]
        if values["test_importers"] == 0
        else 0.0
    )
    cycle = p["cycle_per_file"] if values["scc_size"] > 1 else 0.0
    satd = values["satd_count"] * p["satd_per_item"]

    breakdown = dict(zip(COST_KINDS, (complexity, duplication, tests, cycle, satd), strict=True))
    minutes = math.fsum(breakdown.values())
    multiplier = 1.0 + p["blast_radius_factor"] * math.log2(1.0 + values["fan_in"])
    hours = float(max(p["min_hours"], minutes * multiplier / 60.0))
    return CostEstimate(hours=hours, low=LOW_FACTOR * hours, high=HIGH_FACTOR * hours, breakdown=breakdown)


@functools.cache
def _default_params() -> tuple[tuple[str, float], ...]:
    data = yaml.safe_load(COST_DEFAULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{COST_DEFAULTS_PATH} must be a mapping")
    missing = [key for key in COST_MODEL_KEYS if key not in data]
    if missing:
        raise ValueError(f"{COST_DEFAULTS_PATH} is missing keys: {', '.join(missing)}")
    return tuple((key, _number(key, data[key])) for key in COST_MODEL_KEYS)


def _number(key: str, value: object) -> float:
    """``value`` as a float; raises ``ValueError`` unless it is a finite real number >= 0 (bools rejected)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"cost model value {key!r} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"cost model value {key!r} must be finite and >= 0, got {value!r}")
    return number


def _resolve(cost_model: Mapping[str, Any]) -> dict[str, float]:
    """The package defaults overlaid with the known keys of ``cost_model`` (unknown keys are ignored)."""
    params = dict(_default_params())
    for key in COST_MODEL_KEYS:
        if key in cost_model:
            params[key] = _number(key, cost_model[key])
    return params


def _row_value(row: pd.Series, column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"feature {column!r} must be finite and >= 0, got {value!r}")
    return value

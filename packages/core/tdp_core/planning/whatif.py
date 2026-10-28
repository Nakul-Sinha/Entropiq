"""What-if simulation: how planned fixes would change one file's priority and calibrated risk.

A what-if takes the full feature frame of one analysed repository, one file and a set of fix actions
from ``WHATIF_ACTIONS``. It edits that file's static, impact, test and SATD features the way the actions
would (``ACTION_COLUMNS`` lists which), recomputes the file's ``pct_`` copies of the changed columns against
the other files, scores every file again with the bundle's ranker and reports the file's priority, band and
risk before and after. History features (families E, O and D) are never touched, and the other files keep
their features, percentiles and scores (a ceteris paribus simulation).

Two rules keep the simulation honest:

- **Never worsen.** Each action column has a target value, but the new value is ``max(old, target)`` for the
  columns a fix raises (``RAISED_COLUMNS``: test signals) and ``min(old, target)`` for every other column, so
  no action ever moves a feature the wrong way (splitting a simple function never raises ``cc_max`` to the
  floor, and an already tested file keeps its higher test counts).
- **Same percentile rule as feature assembly.** A changed column ``c`` gets
  ``pct_c = (count(other < new) + 0.5 * count(other == new) + 1) / n`` over the other ``n - 1`` files, which
  is exactly ``rank(pct=True, method="average")`` of the column with the file's new value. Columns that did
  not change keep their stored percentile bit for bit, so a no-op is exactly a no-op.

All comparisons happen in ``float32``, the precision the ranker sees. Importing this module does not import
XGBoost: the bundle is only used, never constructed, here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tdp_core.contracts import FEATURE_COLUMNS, FEATURE_INPUT_COLUMNS, WHATIF_ACTIONS, Band, band_for
from tdp_core.model.priority import priority_from_scores

if TYPE_CHECKING:
    from tdp_core.model.bundle import Bundle

logger = logging.getLogger(__name__)

type _Matrix = npt.NDArray[np.float32]

ACTION_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "add_tests": ("has_test_file", "test_importers", "test_cochange_rate"),
        "split_longest_function": ("cc_max", "cog_max", "nesting_max", "longest_function"),
        "remove_duplication": ("dup_ratio",),
        "resolve_satd": ("satd_count", "satd_density", "satd_age_days"),
        "break_cycle": ("scc_size", "hidden_coupling"),
    }
)
"""Raw feature columns each action may change, keyed in ``WHATIF_ACTIONS`` order (read-only).

The column sets are disjoint and never include an evolution, ownership or defect history column.
``pkg_test_ratio`` is never changed: one new test file barely moves a package's ratio."""

RAISED_COLUMNS: Final[frozenset[str]] = frozenset({"has_test_file", "test_importers", "test_cochange_rate"})
"""Action columns that a fix raises (``new = max(old, target)``); every other action column can only fall."""

CC_MAX_FLOOR: Final = 10.0
"""``split_longest_function`` halves ``cc_max`` but not below this (cyclomatic points); a file already below
it keeps its value. Equal to the cost model's default ``complexity_threshold``."""

_POS: Final[Mapping[str, int]] = MappingProxyType({c: j for j, c in enumerate(FEATURE_INPUT_COLUMNS)})
_PCT_POS: Final[Mapping[str, int]] = MappingProxyType(
    {c: _POS[f"pct_{c}"] for c in FEATURE_COLUMNS if f"pct_{c}" in _POS}
)


@dataclass(frozen=True, slots=True)
class WhatIfResult:
    """The outcome of :func:`what_if` for one file.

    ``before`` and ``after`` are ``(priority, band, risk)``: priority is an integer percent 0..100 within the
    repository, band follows ``band_for`` and risk is the calibrated probability 0..1 (built-in ``int``,
    ``str`` and ``float``). ``changed_features`` maps each raw feature the actions changed to
    ``(old, new)`` (built-in floats holding the exact ``float32`` values) in ``FEATURE_COLUMNS`` order; it
    is ``{}`` when nothing changed. The recomputed ``pct_`` copies of those columns are not listed.
    Instances compare by value; they are not hashable because ``changed_features`` is a dict.
    """

    path: str
    before: tuple[int, Band, float]
    after: tuple[int, Band, float]
    changed_features: dict[str, tuple[float, float]]

    def to_json(self) -> dict[str, Any]:
        """Return the body of ``POST /analyses/{id}/whatif``.

        Shape: ``{"before": {"priority", "band", "risk"}, "after": {...}, "changed_features": {name: [old,
        new]}}``. The path is not included (the request names the file). Every value is a plain JSON value
        and never NaN, so ``json.dumps(..., allow_nan=False)`` works.
        """
        return {
            "before": _state_json(self.before),
            "after": _state_json(self.after),
            "changed_features": {name: [old, new] for name, (old, new) in self.changed_features.items()},
        }


def apply_actions(
    features: pd.DataFrame, path: str, actions: Sequence[str]
) -> tuple[pd.Series, dict[str, tuple[float, float]]]:
    """Return the model input row of ``path`` after ``actions``, plus the raw features that changed.

    ``features`` is the full feature frame of one repository: indexed by path (the ``AnalysisResult`` shape)
    or with a ``path`` column (the ``BuildResult`` and dataset shape). It needs every column of
    ``FEATURE_INPUT_COLUMNS`` in any order; other columns are ignored and float64 frames (for example rebuilt
    from JSON) work. ``actions`` is a sequence of names from ``WHATIF_ACTIONS`` with set semantics: order and
    duplicates do not matter and ``[]`` means no change.

    Targets per action (``v`` is the file's old value; the never-worsen rule then keeps the old value
    whenever the target would be worse):

    - ``add_tests``: ``has_test_file = 1``, ``test_importers = max(v, 1)``, ``test_cochange_rate = max(v,
      median rate of the other files with test_importers > 0)``; the rate is kept when no other file is
      tested.
    - ``split_longest_function``: ``cc_max = min(v, max(CC_MAX_FLOOR, v / 2))``, ``cog_max`` and
      ``longest_function`` (lines) halved (halves can be fractional, for example 20.5), ``nesting_max =
      max(0, v - 1)``. A file without functions (all four 0) does not change.
    - ``remove_duplication``: ``dup_ratio = 0``.
    - ``resolve_satd``: ``satd_count``, ``satd_density`` and ``satd_age_days`` (days) set to 0.
    - ``break_cycle``: ``scc_size = 1`` and ``hidden_coupling = 0`` (an ``scc_size`` of 0 stays 0).

    Each changed column with a ``pct_`` copy gets a new percentile against the other files (see the module
    docstring); every other entry is copied bit for bit. A single-file frame gives every changed percentile
    1.0. The returned Series is ``float32``, indexed by ``FEATURE_INPUT_COLUMNS`` and named ``path``; the
    dict maps raw feature names to ``(old, new)`` built-in floats in ``FEATURE_COLUMNS`` order. The function
    is pure: ``features`` is never modified.

    Raises ``TypeError`` when ``actions`` is a bare string; ``ValueError`` for an unknown action, repeated
    paths, missing model input columns, or non-numeric, NaN or infinite model inputs; ``KeyError`` when the
    frame has no row for ``path``.
    """
    wanted = _normalise_actions(actions)
    index, matrix = _model_matrix(features)
    i = _row_index(index, path)
    row, changed = _simulate(matrix, i, wanted)
    logger.debug("what-if row for %s with %s: %d features changed", path, list(wanted), len(changed))
    return pd.Series(row, index=pd.Index(list(FEATURE_INPUT_COLUMNS)), name=path), changed


def what_if(features: pd.DataFrame, path: str, actions: Sequence[str], bundle: Bundle) -> WhatIfResult:
    """Simulate ``actions`` on the file ``path`` and return its priority, band and risk before and after.

    ``features``, ``path`` and ``actions`` are as in :func:`apply_actions`, which describes the feature
    changes. The original frame and the changed row are scored in one ranker call (``n + 1`` rows); per-row
    margins do not depend on the batch, so this equals scoring the original frame and the modified frame
    separately. ``before`` uses the original scores of all ``n`` files; ``after`` replaces only this file's
    score, so the other files keep their scores (ceteris paribus). Priorities follow
    ``priority_from_scores`` (strictly lower scores count, ties share a priority) and risks
    ``bundle.calibrator.predict``. ``before`` therefore equals the stored analysis values when ``bundle`` is
    the model version that produced them.

    Edge cases: a no-op (for example ``[]``, or ``resolve_satd`` on a file without SATD) gives
    ``after == before`` and ``changed_features == {}``; a single-file frame has priority 100 before and
    after while its risk may still change. With the dev bundle (monotone in every action column) ``after``
    is never worse than ``before``; a real model without such constraints can report a worse ``after``,
    which is returned as it is. The function is pure and thread safe (the ranker uses lock-free inplace
    prediction).

    Raises the same errors as :func:`apply_actions`, plus whatever the ranker or calibrator raise.
    """
    wanted = _normalise_actions(actions)
    index, matrix = _model_matrix(features)
    i = _row_index(index, path)
    row, changed = _simulate(matrix, i, wanted)
    n = len(matrix)
    batch = pd.DataFrame(np.vstack([matrix, row[np.newaxis, :]]), columns=list(FEATURE_INPUT_COLUMNS))
    scores = bundle.ranker.score(batch)
    before_scores = scores[:n]
    after_scores = before_scores.copy()
    after_scores[i] = scores[n]
    p_before = int(priority_from_scores(before_scores)[i])
    p_after = int(priority_from_scores(after_scores)[i])
    risk = bundle.calibrator.predict(np.array([scores[i], scores[n]], dtype=np.float64))[0]
    result = WhatIfResult(
        path=path,
        before=(p_before, band_for(p_before), float(risk[0])),
        after=(p_after, band_for(p_after), float(risk[1])),
        changed_features=changed,
    )
    logger.debug(
        "what-if %s with %s over %d files: priority %d -> %d, risk %.4f -> %.4f, %d features changed",
        path,
        list(wanted),
        n,
        p_before,
        p_after,
        result.before[2],
        result.after[2],
        len(changed),
    )
    return result


def _state_json(state: tuple[int, Band, float]) -> dict[str, Any]:
    priority, band, risk = state
    return {"priority": priority, "band": band, "risk": risk}


def _normalise_actions(actions: Sequence[str]) -> tuple[str, ...]:
    """Validate ``actions`` and return the distinct names in ``WHATIF_ACTIONS`` order."""
    if isinstance(actions, str):
        raise TypeError("actions must be a sequence of action names, not a string")
    requested = list(actions)
    unknown: list[object] = []
    for action in requested:
        if action not in WHATIF_ACTIONS and action not in unknown:
            unknown.append(action)
    if unknown:
        raise ValueError(
            f"unknown what-if action(s) {', '.join(map(repr, unknown))}; "
            f"allowed actions are {', '.join(WHATIF_ACTIONS)}"
        )
    return tuple(a for a in WHATIF_ACTIONS if a in requested)


def _model_matrix(features: pd.DataFrame) -> tuple[pd.Index, _Matrix]:
    """Return the path index and the ``float32`` model input matrix (columns ``FEATURE_INPUT_COLUMNS``)."""
    frame = features.set_index("path") if "path" in features.columns else features
    if not frame.index.is_unique:
        raise ValueError("the feature frame has repeated paths")
    missing = [c for c in FEATURE_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"the feature frame lacks {len(missing)} model input columns ({', '.join(missing[:5])})"
        )
    try:
        # values beyond the float32 range become infinite here and are rejected below
        with np.errstate(over="ignore"):
            matrix = frame.loc[:, list(FEATURE_INPUT_COLUMNS)].to_numpy(dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the feature frame has non-numeric model inputs: {exc}") from exc
    if matrix.shape[1] != len(FEATURE_INPUT_COLUMNS):
        raise ValueError("the feature frame repeats model input columns")
    finite = np.isfinite(matrix)
    if not finite.all():
        bad = [FEATURE_INPUT_COLUMNS[j] for j in np.flatnonzero(~finite.all(axis=0))]
        raise ValueError(
            f"the feature frame has NaN or infinite values in {len(bad)} columns ({', '.join(bad[:5])})"
        )
    return frame.index, matrix


def _row_index(index: pd.Index, path: str) -> int:
    """Position of ``path`` in a unique ``index``; ``KeyError`` when it is absent (or the frame is empty)."""
    i = int(index.get_indexer(pd.Index([path]))[0])
    if i < 0:
        raise KeyError(f"no feature row for {path!r}")
    return i


def _targets(matrix: _Matrix, i: int, actions: tuple[str, ...]) -> dict[str, float]:
    """Target value (float64) of every action column, before the never-worsen rule."""

    def old(column: str) -> float:
        return float(matrix[i, _POS[column]])

    targets: dict[str, float] = {}
    if "add_tests" in actions:
        targets["has_test_file"] = 1.0
        targets["test_importers"] = 1.0
        tested = matrix[:, _POS["test_importers"]] > 0
        tested[i] = False
        if tested.any():
            rates = matrix[tested, _POS["test_cochange_rate"]].astype(np.float64)
            targets["test_cochange_rate"] = float(np.median(rates))
    if "split_longest_function" in actions:
        targets["cc_max"] = max(CC_MAX_FLOOR, old("cc_max") / 2.0)
        targets["cog_max"] = old("cog_max") / 2.0
        targets["nesting_max"] = max(0.0, old("nesting_max") - 1.0)
        targets["longest_function"] = old("longest_function") / 2.0
    if "remove_duplication" in actions:
        targets["dup_ratio"] = 0.0
    if "resolve_satd" in actions:
        for column in ACTION_COLUMNS["resolve_satd"]:
            targets[column] = 0.0
    if "break_cycle" in actions:
        targets["scc_size"] = 1.0
        targets["hidden_coupling"] = 0.0
    return targets


def _simulate(
    matrix: _Matrix, i: int, actions: tuple[str, ...]
) -> tuple[_Matrix, dict[str, tuple[float, float]]]:
    """Apply ``actions`` to a copy of row ``i``; return the new row and the changed raw features."""
    row = matrix[i].copy()
    targets = _targets(matrix, i, actions)
    changed: dict[str, tuple[float, float]] = {}
    n = len(matrix)
    for column in FEATURE_COLUMNS:
        if column not in targets:
            continue
        j = _POS[column]
        old32 = row[j]
        if column in RAISED_COLUMNS:
            new = max(float(old32), targets[column])
        else:
            new = min(float(old32), targets[column])
        new32 = np.float32(new)
        if new32 == old32:
            continue
        row[j] = new32
        changed[column] = (float(old32), float(new32))
        if column in _PCT_POS:
            row[_PCT_POS[column]] = _percentile(np.delete(matrix[:, j], i), new32, n)
    return row, changed


def _percentile(others: _Matrix, value: np.float32, n: int) -> np.float32:
    """``rank(pct=True, method="average")`` of ``value`` among ``others`` plus itself (``n`` values)."""
    less = int(np.count_nonzero(others < value))
    equal = int(np.count_nonzero(others == value))
    return np.float32((less + 0.5 * equal + 1.0) / n)

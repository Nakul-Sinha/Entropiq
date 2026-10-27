"""Isotonic risk calibration with bootstrap ranges.

A calibrator maps a raw ranker score (unitless margin) to the probability that the file has at least one
latent defect in the next horizon. It is fitted once with scikit-learn's ``IsotonicRegression`` and stored as
four threshold arrays (``x``, ``y``, ``y_low``, ``y_high``); inference only needs ``numpy.interp``, so
scikit-learn is imported lazily inside :meth:`IsotonicCalibrator.fit`.

``y_low`` and ``y_high`` are the 5th and 95th percentiles of bootstrap refits evaluated on the same ``x``
grid. They are clamped so that ``y_low <= y <= y_high`` everywhere and kept non-decreasing, because the
scoring output and the dashboard rely on ``risk_low <= risk <= risk_high``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

LOW_PERCENTILE: Final = 5.0
HIGH_PERCENTILE: Final = 95.0
JSON_KEYS: Final = ("x", "y", "y_low", "y_high")
_TOL: Final = 1e-9

type FloatArray = npt.NDArray[np.float64]


class IsotonicCalibrator:
    """Monotone map from ranker scores to calibrated risk, with a bootstrap range.

    The four arrays are thresholds of a piecewise linear curve: ``x`` holds scores (strictly increasing),
    ``y`` the calibrated probability at each score, and ``y_low`` / ``y_high`` the bounds of its range.
    Arrays are stored as read-only ``float64`` copies.
    """

    __slots__ = ("_x", "_y", "_y_high", "_y_low")

    def __init__(
        self, x: npt.ArrayLike, y: npt.ArrayLike, y_low: npt.ArrayLike, y_high: npt.ArrayLike
    ) -> None:
        """Build a calibrator from threshold arrays.

        Raises ``ValueError`` unless all four arrays are one-dimensional, of equal length >= 1 and finite;
        ``x`` is strictly increasing; ``y``, ``y_low`` and ``y_high`` lie in [0, 1] and are non-decreasing;
        and ``y_low <= y <= y_high``. A tolerance of 1e-9 is allowed on the last three rules; values are
        then clipped into range so the stored curves satisfy them exactly.
        """
        arrays = {
            k: np.array(v, dtype=np.float64) for k, v in zip(JSON_KEYS, (x, y, y_low, y_high), strict=True)
        }
        n = arrays["x"].size
        for key, arr in arrays.items():
            if arr.ndim != 1:
                raise ValueError(f"{key} must be one-dimensional, got shape {arr.shape}")
            if arr.size != n:
                raise ValueError(f"{key} has {arr.size} values but x has {n}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{key} must be finite (no NaN or infinity)")
        if n == 0:
            raise ValueError("a calibrator needs at least one threshold")
        if np.any(np.diff(arrays["x"]) <= 0):
            raise ValueError("x must be strictly increasing")
        for key in ("y", "y_low", "y_high"):
            arr = arrays[key]
            if np.any(arr < -_TOL) or np.any(arr > 1 + _TOL):
                raise ValueError(f"{key} must lie in [0, 1]")
            if np.any(np.diff(arr) < -_TOL):
                raise ValueError(f"{key} must be non-decreasing")
        if np.any(arrays["y_low"] > arrays["y"] + _TOL):
            raise ValueError("y_low must not exceed y")
        if np.any(arrays["y_high"] < arrays["y"] - _TOL):
            raise ValueError("y_high must not be below y")
        cy = np.maximum.accumulate(np.clip(arrays["y"], 0.0, 1.0))
        lo = np.minimum(np.maximum.accumulate(np.clip(arrays["y_low"], 0.0, 1.0)), cy)
        hi = np.maximum(np.maximum.accumulate(np.clip(arrays["y_high"], 0.0, 1.0)), cy)
        self._x = _frozen(arrays["x"])
        self._y = _frozen(cy)
        self._y_low = _frozen(lo)
        self._y_high = _frozen(hi)

    @property
    def x(self) -> FloatArray:
        """Score thresholds (raw ranker margins), strictly increasing, read-only."""
        return self._x

    @property
    def y(self) -> FloatArray:
        """Calibrated probability (0..1) at each threshold, non-decreasing, read-only."""
        return self._y

    @property
    def y_low(self) -> FloatArray:
        """Lower end of the bootstrap range (0..1) at each threshold, read-only."""
        return self._y_low

    @property
    def y_high(self) -> FloatArray:
        """Upper end of the bootstrap range (0..1) at each threshold, read-only."""
        return self._y_high

    def __repr__(self) -> str:
        return f"IsotonicCalibrator(thresholds={self._x.size})"

    @classmethod
    def fit(
        cls, scores: npt.ArrayLike, y: npt.ArrayLike, *, n_boot: int = 200, seed: int = 42
    ) -> IsotonicCalibrator:
        """Fit isotonic regression of ``y`` on ``scores`` plus ``n_boot`` bootstrap refits.

        ``scores`` are raw ranker margins; ``y`` are outcomes in [0, 1] (booleans are accepted), usually
        ``has_latent_defect``. The main fit is ``IsotonicRegression(increasing=True, out_of_bounds="clip",
        y_min=0, y_max=1)``. Each bootstrap refit resamples rows with replacement (``numpy`` generator seeded
        with ``seed``) and is evaluated on the main ``x`` grid; ``y_low`` and ``y_high`` are the 5th and 95th
        percentiles of those curves, clamped to bracket ``y``. ``n_boot=0`` gives ``y_low = y_high = y``.

        Runtime is roughly linear in ``n_boot * len(scores)`` (about 0.5 s for 200 refits on 6,000 rows).
        Raises ``ValueError`` for empty, mismatched, non-finite or multi-dimensional input, labels outside
        [0, 1], or ``n_boot < 0``.
        """
        s = np.asarray(scores, dtype=np.float64)
        t = np.asarray(y, dtype=np.float64)
        if s.ndim != 1 or t.ndim != 1:
            raise ValueError("scores and y must be one-dimensional")
        if s.size != t.size:
            raise ValueError(f"scores has {s.size} values but y has {t.size}")
        if s.size == 0:
            raise ValueError("cannot fit a calibrator on zero samples")
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(t))):
            raise ValueError("scores and y must be finite (no NaN or infinity)")
        if np.any(t < 0.0) or np.any(t > 1.0):
            raise ValueError("labels must lie in [0, 1]")
        if n_boot < 0:
            raise ValueError(f"n_boot must be >= 0, got {n_boot}")

        main = _isotonic().fit(s, t)
        x = np.asarray(main.X_thresholds_, dtype=np.float64)
        curve = np.asarray(main.y_thresholds_, dtype=np.float64)
        if n_boot == 0:
            lo = curve.copy()
            hi = curve.copy()
        else:
            n = s.size
            rng = np.random.default_rng(seed)
            boots = np.empty((n_boot, x.size), dtype=np.float64)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                fb = _isotonic().fit(s[idx], t[idx])
                boots[b] = np.interp(
                    x,
                    np.asarray(fb.X_thresholds_, dtype=np.float64),
                    np.asarray(fb.y_thresholds_, dtype=np.float64),
                )
            lo, hi = np.percentile(boots, [LOW_PERCENTILE, HIGH_PERCENTILE], axis=0)
        lo = np.clip(np.minimum(np.maximum.accumulate(lo), curve), 0.0, 1.0)
        hi = np.clip(np.maximum(np.maximum.accumulate(hi), curve), 0.0, 1.0)
        logger.debug(
            "fitted isotonic calibrator: %d samples, %d thresholds, %d bootstraps", s.size, x.size, n_boot
        )
        return cls(x, curve, lo, hi)

    def predict(self, scores: npt.ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``(risk, low, high)`` probabilities (0..1) for raw ranker ``scores``.

        Each value is a linear interpolation between thresholds; scores outside ``[x[0], x[-1]]`` get the end
        value (the same as clipping). A calibrator with one threshold returns constants. An empty input gives
        three empty arrays. ``low <= risk <= high`` holds element-wise. Raises ``ValueError`` if ``scores``
        is not one-dimensional or holds NaN or infinite values.
        """
        s = np.asarray(scores, dtype=np.float64)
        if s.ndim != 1:
            raise ValueError(f"scores must be one-dimensional, got shape {s.shape}")
        if not np.all(np.isfinite(s)):
            raise ValueError("scores must be finite (no NaN or infinity)")
        risk = np.interp(s, self._x, self._y)
        low = np.interp(s, self._x, self._y_low)
        high = np.interp(s, self._x, self._y_high)
        return (
            np.asarray(risk, dtype=np.float64),
            np.asarray(low, dtype=np.float64),
            np.asarray(high, dtype=np.float64),
        )

    def to_json(self) -> dict[str, list[float]]:
        """Return ``{"x": [...], "y": [...], "y_low": [...], "y_high": [...]}`` as Python floats.

        The values round-trip exactly through ``json.dumps`` / ``json.loads`` and :meth:`from_json`.
        """
        return {
            "x": self._x.tolist(),
            "y": self._y.tolist(),
            "y_low": self._y_low.tolist(),
            "y_high": self._y_high.tolist(),
        }

    @classmethod
    def from_json(cls, d: Mapping[str, Sequence[float]]) -> IsotonicCalibrator:
        """Rebuild a calibrator from :meth:`to_json` output.

        Extra keys are ignored. Raises ``ValueError`` if ``d`` is not a mapping, a key is missing, or the
        arrays break any rule checked by the constructor.
        """
        if not isinstance(d, Mapping):
            raise ValueError(f"calibrator payload must be a mapping, got {type(d).__name__}")
        missing = [k for k in JSON_KEYS if k not in d]
        if missing:
            raise ValueError(f"calibrator payload is missing {', '.join(missing)}")
        try:
            return cls(d["x"], d["y"], d["y_low"], d["y_high"])
        except TypeError as exc:
            raise ValueError(f"calibrator arrays must hold numbers: {exc}") from exc


def _frozen(arr: FloatArray) -> FloatArray:
    out = np.array(arr, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _isotonic() -> Any:
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)

"""XGBoost ranker runtime: load a booster and score feature frames.

Scores are raw LambdaMART margins (unitless, higher means more expected maintenance pain). They are only
comparable within one model; :mod:`tdp_core.model.priority` turns them into percentiles and
:mod:`tdp_core.model.calibration` into probabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb

from tdp_core.contracts import FEATURE_INPUT_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_NTHREAD: Final = 4
"""Thread cap for scoring (the laptop and worker budget); raise it with ``booster.set_param``."""


class Ranker:
    """A validated XGBoost booster that scores frames with columns ``FEATURE_INPUT_COLUMNS``.

    Scoring is thread safe (it uses ``inplace_predict``) and every boosted round is used, so early-stopped
    boosters must be trimmed before they get here (``save_bundle`` does that).
    """

    __slots__ = ("_booster",)

    def __init__(self, booster: xgb.Booster) -> None:
        """Wrap ``booster`` after checking it takes exactly the model input columns.

        Raises ``ValueError`` unless the booster has ``len(FEATURE_INPUT_COLUMNS)`` (89) features and its
        feature names are either unset or exactly ``FEATURE_INPUT_COLUMNS`` in order. Logs a warning when the
        booster carries a ``best_iteration`` attribute smaller than its last round, because scoring then uses
        more trees than early stopping selected.
        """
        n_expected = len(FEATURE_INPUT_COLUMNS)
        n_features = booster.num_features()
        if n_features != n_expected:
            raise ValueError(f"ranker expects {n_expected} features, booster has {n_features}")
        names = booster.feature_names
        if names is not None and list(names) != list(FEATURE_INPUT_COLUMNS):
            raise ValueError("booster feature names differ from FEATURE_INPUT_COLUMNS")
        best = booster.attributes().get("best_iteration")
        if best is not None and int(best) + 1 < booster.num_boosted_rounds():
            logger.warning(
                "booster has best_iteration=%s but %d rounds; scoring uses every round",
                best,
                booster.num_boosted_rounds(),
            )
        self._booster = booster

    @classmethod
    def from_json(cls, path: Path, *, nthread: int = DEFAULT_NTHREAD) -> Ranker:
        """Load a booster saved with ``save_model("ranker.json")`` and validate it.

        The booster is configured for CPU scoring with ``nthread`` threads (default 4), so GPU-trained
        models load anywhere. Raises ``FileNotFoundError`` if ``path`` is not a file, ``ValueError`` if it
        is empty, ``nthread < 1``, or XGBoost cannot parse it (``XGBoostError`` is a ``ValueError``).
        """
        path = Path(path)
        if nthread < 1:
            raise ValueError(f"nthread must be >= 1, got {nthread}")
        if not path.is_file():
            raise FileNotFoundError(f"ranker file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"ranker file is empty: {path}")
        booster = xgb.Booster(params={"nthread": nthread, "device": "cpu"}, model_file=str(path))
        return cls(booster)

    def __repr__(self) -> str:
        return f"Ranker(features={self._booster.num_features()}, rounds={self._booster.num_boosted_rounds()})"

    @property
    def booster(self) -> xgb.Booster:
        """The underlying booster (for example for ``shap.TreeExplainer``)."""
        return self._booster

    def score(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return raw ranker margins, shape ``(len(X),)``, dtype ``float64``.

        ``X`` must have exactly the columns ``FEATURE_INPUT_COLUMNS`` in that order; values are cast to
        ``float32`` and NaN is treated as missing. The index is ignored. An empty frame gives an empty array.
        Raises ``ValueError`` naming how the columns differ.
        """
        cols = list(X.columns)
        expected = list(FEATURE_INPUT_COLUMNS)
        if cols != expected:
            missing = sorted(set(expected) - set(cols))
            extra = sorted(set(cols) - set(expected), key=str)
            if missing or extra:
                raise ValueError(
                    f"feature columns differ: {len(missing)} missing ({', '.join(missing[:5])}), "
                    f"{len(extra)} extra ({', '.join(map(str, extra[:5]))})"
                )
            raise ValueError("feature columns are in the wrong order or repeated; use FEATURE_INPUT_COLUMNS")
        if len(X) == 0:
            return np.zeros(0, dtype=np.float64)
        arr = X.to_numpy(dtype=np.float32)
        out = self._booster.inplace_predict(arr, predict_type="margin", validate_features=False)
        return np.asarray(out, dtype=np.float64).reshape(-1)

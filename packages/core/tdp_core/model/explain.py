"""Explanations for ranker scores: TreeSHAP values, reasons, mitigations and debt types.

:class:`Explainer` runs TreeSHAP on a bundle's XGBoost ranker. Every SHAP value (``phi``) is in ranker-margin
units: for one file the values of its 89 input columns add up to ``score - baseline``. The model baseline
is :attr:`Explainer.expected_value` (the TreeSHAP bias); :meth:`Explainer.explain` by default centres the
values on the average file of the analysed repository instead, because priorities are percentiles within one
repository and the ranking objective never fixes a repository's overall score level.

The rules that turn one row of ``phi`` into text are plain functions, so they can be tested on handcrafted
rows:

- :func:`family_totals` adds up the raw and ``pct_`` contributions of each feature family;
- :func:`render_reasons` keeps the families that raise the score (at most four, each above 5% of the total
  absolute contribution) and the one family that lowers it most (if it outweighs a quarter of the positive
  total), and renders a sentence for the strongest feature of each from the templates in ``reasons.yaml``;
- :func:`assign_debt_types` labels the file with at most two of ``DEBT_TYPES`` from threshold rules on its
  feature values, falling back to its dominant family.

shap is imported lazily by :class:`Explainer`, so loading or validating templates stays cheap.
"""

from __future__ import annotations

import copy
import logging
import math
import string
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from tdp_core.contracts import (
    BINARY_FEATURES,
    DEBT_TYPES,
    FAMILY_ORDER,
    FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    FEATURE_INPUT_COLUMNS,
    Reason,
    family_of,
)

if TYPE_CHECKING:
    from tdp_core.model.bundle import Bundle

logger = logging.getLogger(__name__)

MAX_REASONS: Final = 4
"""Default maximum number of reasons per file."""
REASON_MIN_SHARE: Final = 0.05
"""A family is a reason only if its total exceeds this share of ``sum(|phi|)``."""
MITIGATION_MIN_SHARE: Final = 0.25
"""A family is a mitigation only if ``-total`` exceeds this share of the sum of positive family totals."""
MAX_DEBT_TYPES: Final = 2
"""Maximum number of debt types per file."""
TEMPLATES_FILE: Final = "reasons.yaml"
"""Packaged reason templates (``tdp_core/model/reasons.yaml``), copied into every bundle."""
TEMPLATE_FIELDS: Final = frozenset({"adj", "value", "percentile", "top", "s"})
"""Format fields a template sentence may use."""
TEMPLATE_KEYS: Final = frozenset({"template", "mitigation", "adjectives", "exact"})
"""Keys a template entry may have (``template`` and ``mitigation`` are required)."""

DEBT_TYPE_FAMILIES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "complexity": ("S",),
        "duplication": ("S",),
        "test": ("T",),
        "architecture": ("I",),
        "knowledge": ("O",),
        "admitted": ("Q",),
        "defect_hotspot": ("D", "E"),
    }
)
"""Families whose total ranks a triggered debt type (the largest of them counts)."""

DOMINANT_DEBT_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "S": "complexity",
        "E": "defect_hotspot",
        "D": "defect_hotspot",
        "O": "knowledge",
        "I": "architecture",
        "T": "test",
        "Q": "admitted",
    }
)
"""Debt type of a file whose rules all stay quiet, by its dominant (largest positive) family."""

type Baseline = Literal["repo", "model"]

_N_INPUT: Final = len(FEATURE_INPUT_COLUMNS)
_POS: Final[Mapping[str, int]] = MappingProxyType({c: i for i, c in enumerate(FEATURE_INPUT_COLUMNS)})
_FAMILY_INDEX: Final[Mapping[str, npt.NDArray[np.intp]]] = MappingProxyType(
    {
        fam: np.array([i for i, c in enumerate(FEATURE_INPUT_COLUMNS) if family_of(c) == fam], dtype=np.intp)
        for fam in FAMILY_ORDER
    }
)
_RAW_INDEX: Final = np.array([_POS[c] for c in FEATURE_COLUMNS], dtype=np.intp)
_HAS_PCT: Final = np.array([c not in BINARY_FEATURES for c in FEATURE_COLUMNS], dtype=np.bool_)
_PCT_INDEX: Final = np.array([_POS.get(f"pct_{c}", 0) for c in FEATURE_COLUMNS], dtype=np.intp)
_FEATURE_POS: Final[Mapping[str, int]] = MappingProxyType({c: k for k, c in enumerate(FEATURE_COLUMNS)})
_FAMILY_FEATURES: Final[Mapping[str, npt.NDArray[np.intp]]] = MappingProxyType(
    {fam: np.array([_FEATURE_POS[c] for c in FEATURE_FAMILIES[fam]], dtype=np.intp) for fam in FAMILY_ORDER}
)
_FAMILY_RANK: Final[Mapping[str, int]] = MappingProxyType({f: i for i, f in enumerate(FAMILY_ORDER)})
_DEBT_RANK: Final[Mapping[str, int]] = MappingProxyType({t: i for i, t in enumerate(DEBT_TYPES)})

# Debt rule thresholds, compared with float32-cleaned values (see _clean).
_CC_MAX: Final = 15.0
_NESTING_MAX: Final = 4.0
_DUP_RATIO: Final = 0.10
_PCT_HIDDEN_COUPLING: Final = 0.9
_PCT_BETWEENNESS: Final = 0.95
_PCT_AUTHOR_ENTROPY: Final = 0.9
_PCT_SATD_DENSITY: Final = 0.9

# Values and percentiles every template must render for (validate_templates).
_CHECK_VALUES: Final = (0.0, 1.0, 2.5, 1234.5)
_CHECK_PERCENTILES: Final = (0.0, 50.0, 100.0)
# Characters a rendered sentence must not contain (safe in Markdown tables, HTML and PR comments).
_FORBIDDEN_CHARS: Final = frozenset("{}|<>")


@dataclass(frozen=True, slots=True)
class Explanation:
    """The explanation of one file's score.

    ``reasons`` hold at most ``max_reasons`` families with a positive total, strongest first;
    ``mitigations`` hold zero or one family with a negative total. ``Reason.shap`` is the family total.
    ``debt_types`` is a subset of ``DEBT_TYPES`` with at most two entries (possibly empty).
    ``shap_by_family`` has the seven keys of ``FAMILY_ORDER`` and ``shap_by_feature`` the 46 keys of
    ``FEATURE_COLUMNS`` (each the raw plus ``pct_`` contribution); both are in ranker-margin units relative to
    the baseline used (they add up to ``score - baseline``).
    """

    reasons: tuple[Reason, ...]
    mitigations: tuple[Reason, ...]
    debt_types: tuple[str, ...]
    shap_by_family: dict[str, float]
    shap_by_feature: dict[str, float]


class Explainer:
    """TreeSHAP explanations for the ranker of one bundle.

    Construction validates the bundle's reason templates and builds a ``shap.TreeExplainer`` (path
    dependent, raw margin output); it takes about 0.01 s for the dev bundle and well under a second for large
    models, plus the one-off ``import shap`` (a few seconds). One explainer can serve any number of frames.
    """

    __slots__ = ("_expected_value", "_templates", "_tree", "_version")

    def __init__(self, bundle: Bundle) -> None:
        """Build an explainer for ``bundle`` (from ``load_bundle``).

        Raises ``ValueError`` if ``bundle.reasons`` is not a complete, valid set of templates (see
        :func:`validate_templates`), so a broken bundle fails here rather than in the middle of an analysis.
        """
        validate_templates(bundle.reasons)
        self._templates: dict[str, Any] = copy.deepcopy(dict(bundle.reasons))
        self._version: str = bundle.version

        import shap
        import xgboost as xgb

        booster = bundle.ranker.booster
        self._tree: Any = shap.TreeExplainer(booster)
        # The bias column of XGBoost's own TreeSHAP is the same for every row. shap's expected_value is not
        # used: right after construction it misses the learned base score of XGBoost 3 ranking boosters.
        probe = booster.predict(
            xgb.DMatrix(np.zeros((1, _N_INPUT), dtype=np.float32)),
            pred_contribs=True,
            validate_features=False,
        )
        self._expected_value: float = float(np.asarray(probe, dtype=np.float64)[0, -1])
        logger.debug("explainer for bundle %s: expected value %.6g", self._version, self._expected_value)

    @property
    def expected_value(self) -> float:
        """The TreeSHAP bias in ranker-margin units: ``score == expected_value + phi.sum()`` for every row."""
        return self._expected_value

    def __repr__(self) -> str:
        return f"Explainer(version={self._version!r}, expected_value={self._expected_value:.6g})"

    def shap_values(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return raw TreeSHAP values, shape ``(len(X), 89)``, dtype ``float64``, in ranker-margin units.

        ``X`` must have exactly the columns ``FEATURE_INPUT_COLUMNS`` in that order (the same check and
        message as ``Ranker.score``); values are cast to ``float32`` like the ranker does, NaN is treated as
        missing and the index is ignored. Column ``j`` of the result is the contribution of
        ``FEATURE_INPUT_COLUMNS[j]``, and each row satisfies ``phi.sum() + expected_value == score`` up to
        float32 accumulation error (about 1e-6). Every boosting round is used, like ``Ranker.score``, even
        when the booster carries a ``best_iteration`` attribute. An empty frame gives shape ``(0, 89)``.

        Raises ``ValueError`` naming how the columns differ.
        """
        _check_columns(X)
        if len(X) == 0:
            return np.zeros((0, _N_INPUT), dtype=np.float64)
        arr = X.to_numpy(dtype=np.float32)
        phi = self._tree.shap_values(arr, tree_limit=-1, check_additivity=False)
        return np.array(phi, dtype=np.float64).reshape(len(X), _N_INPUT)

    def explain(
        self, X: pd.DataFrame, *, baseline: Baseline = "repo", max_reasons: int = MAX_REASONS
    ) -> list[Explanation]:
        """Explain every row of ``X`` (one repository snapshot) and return the explanations in row order.

        ``X`` has exactly the columns ``FEATURE_INPUT_COLUMNS`` and finite values (no NaN). With the default
        ``baseline="repo"`` the SHAP values are centred on their mean over the rows of ``X``, so each
        explanation is relative to the average file of this repository: ``shap_by_family`` (and therefore the
        reasons and mitigations) add up to ``score - mean(score of X)``. Pass all files of one repository and
        nothing else. A single row is its own mean, so it falls back to the model baseline. With
        ``baseline="model"`` the values add up to ``score - expected_value``.

        Each row is then turned into text exactly as :func:`render_reasons` and :func:`assign_debt_types` do
        on the (centred) SHAP row. Cost: one SHAP pass plus well under a millisecond per file. An empty frame
        gives ``[]``.

        Raises ``ValueError`` for an unknown ``baseline``, ``max_reasons < 0``, wrong columns, or a value
        that is NaN or infinite (the message names the row label and column).
        """
        if baseline not in ("repo", "model"):
            raise ValueError(f"baseline must be 'repo' or 'model', got {baseline!r}")
        if max_reasons < 0:
            raise ValueError(f"max_reasons must be >= 0, got {max_reasons}")
        _check_columns(X)
        if len(X) == 0:
            return []
        values = X.to_numpy(dtype=np.float64)
        bad = ~np.isfinite(values)
        if bad.any():
            i, j = (int(k) for k in np.argwhere(bad)[0])
            label = X.index[i]
            if isinstance(label, np.generic):
                label = label.item()
            raise ValueError(f"row {label!r}: feature {FEATURE_INPUT_COLUMNS[j]} is not finite")
        phi = self.shap_values(X)
        if baseline == "repo" and len(X) > 1:
            phi = phi - phi.mean(axis=0)
        out: list[Explanation] = []
        for i in range(len(X)):
            totals = _family_totals(phi[i])
            reasons, mitigations = _reasons(values[i], phi[i], totals, self._templates, max_reasons)
            debt = _debt_types(values[i], totals)
            out.append(Explanation(reasons, mitigations, debt, totals, _feature_totals(phi[i])))
        return out


def load_templates(path: Path | None = None) -> dict[str, Any]:
    """Load and validate reason templates from ``path``, or the packaged ``reasons.yaml`` when None.

    Returns the parsed mapping (feature name to entry). Raises ``FileNotFoundError`` if ``path`` is not a
    file and ``ValueError`` if it is not UTF-8 YAML, not a mapping, or fails :func:`validate_templates`.
    """
    if path is None:
        source = f"packaged {TEMPLATES_FILE}"
        text = (files("tdp_core.model") / TEMPLATES_FILE).read_text(encoding="utf-8")
    else:
        path = Path(path)
        source = str(path)
        if not path.is_file():
            raise FileNotFoundError(f"reason templates not found: {path}")
        text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{source}: not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{source}: reason templates must be a YAML mapping, got {type(data).__name__}")
    validate_templates(data)
    return data


def validate_templates(templates: Mapping[str, Any]) -> None:
    """Check that ``templates`` has a valid entry for every feature in ``FEATURE_COLUMNS``.

    Each entry is a mapping with only the keys in ``TEMPLATE_KEYS``:

    - ``template`` and ``mitigation``: non-empty sentences;
    - ``adjectives`` (optional): a non-empty list of ``[threshold, text]`` pairs, thresholds numbers from
      100 down to 0, strictly decreasing, the last exactly 0 (so every percentile gets an adjective);
    - ``exact`` (optional): a list of ``[value, sentence]`` pairs, used instead of the sentence when the
      feature value equals ``value``.

    Sentences may only use the fields ``{adj}`` (requires ``adjectives``), ``{value}``, ``{percentile}``,
    ``{top}`` and ``{s}``, without conversions, attribute or index access, or nested fields. Every sentence
    must render for values 0, 1, 2.5 and 1234.5 at percentiles 0, 50 and 100 to one printable line without
    braces, pipes or angle brackets. Keys that are not features are ignored.

    Raises ``ValueError`` naming the feature and key of the first problem.
    """
    if not isinstance(templates, Mapping):
        raise ValueError(f"reason templates must be a mapping, got {type(templates).__name__}")
    missing = [f for f in FEATURE_COLUMNS if f not in templates]
    if missing:
        raise ValueError(
            f"reason templates miss {len(missing)} of {len(FEATURE_COLUMNS)} features: "
            f"{', '.join(missing[:5])}"
        )
    for feature in FEATURE_COLUMNS:
        _validate_entry(feature, templates[feature])


def render_text(
    templates: Mapping[str, Any],
    feature: str,
    kind: Literal["template", "mitigation"],
    value: float,
    percentile: float,
) -> str:
    """Render the ``kind`` sentence of ``feature`` for a raw ``value`` and a ``percentile`` (0 to 100).

    The first ``exact`` pair whose value equals ``value`` replaces the sentence. ``{adj}`` is the text of the
    first adjective whose threshold is at most ``percentile`` (empty if none), ``{top}`` is
    ``max(1, 100 - percentile)`` and ``{s}`` is ``""`` when ``value`` rounds to 1, else ``"s"``. ``value`` and
    ``percentile`` are used exactly as given.

    Raises ``ValueError`` if ``feature`` has no template entry, ``kind`` is unknown, ``value`` or
    ``percentile`` is not finite, or the sentence cannot be rendered.
    """
    if kind not in ("template", "mitigation"):
        raise ValueError(f"kind must be 'template' or 'mitigation', got {kind!r}")
    if not (math.isfinite(value) and math.isfinite(percentile)):
        raise ValueError(f"value and percentile must be finite, got {value!r} and {percentile!r}")
    entry = templates.get(feature)
    if not isinstance(entry, Mapping):
        raise ValueError(f"no reason template for feature {feature!r}")
    try:
        text = _pick_text(entry, kind, value)
        return _format(text, entry.get("adjectives"), value, percentile)
    except (KeyError, IndexError, ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"cannot render the {kind} of feature {feature!r}: {exc}") from exc


def family_totals(phi_row: npt.ArrayLike) -> dict[str, float]:
    """Sum one row of SHAP values (89 values in ``FEATURE_INPUT_COLUMNS`` order) per feature family.

    Each total adds the raw and ``pct_`` columns of the family (ranker-margin units). Keys follow
    ``FAMILY_ORDER`` and values are Python floats; the totals add up to ``sum(phi_row)``.

    Raises ``ValueError`` unless ``phi_row`` has shape ``(89,)`` and only finite values.
    """
    return _family_totals(_as_phi(phi_row))


def feature_totals(phi_row: npt.ArrayLike) -> dict[str, float]:
    """Sum one row of SHAP values per base feature: ``phi[x] + phi[pct_x]`` (binary features: ``phi[x]``).

    Keys follow ``FEATURE_COLUMNS`` (46 features), values are Python floats in ranker-margin units and add
    up to ``sum(phi_row)``.

    Raises ``ValueError`` unless ``phi_row`` has shape ``(89,)`` and only finite values.
    """
    return _feature_totals(_as_phi(phi_row))


def render_reasons(
    row: pd.Series,
    phi_row: npt.ArrayLike,
    templates: Mapping[str, Any],
    max_reasons: int = MAX_REASONS,
) -> tuple[tuple[Reason, ...], tuple[Reason, ...]]:
    """Return ``(reasons, mitigations)`` for one file.

    ``row`` holds the file's values under the labels ``FEATURE_INPUT_COLUMNS`` (other labels are ignored);
    ``phi_row`` its 89 SHAP values in that order. Families are ranked by their total (ties in
    ``FAMILY_ORDER``); the strongest ones become reasons while their total is positive and above
    ``REASON_MIN_SHARE`` (5%) of ``sum(|phi_row|)``, at most ``max_reasons``. The family with the most
    negative total becomes the single mitigation when ``-total`` exceeds ``MITIGATION_MIN_SHARE`` (25%) of
    the sum of positive totals (any negative family qualifies when none is positive); ``max_reasons`` does
    not limit it.

    Within a family the feature with the largest raw plus ``pct_`` contribution is reported (the smallest
    for a mitigation; ties go to the earlier feature in ``FEATURE_FAMILIES``). ``Reason.value`` is the
    feature value as the model saw it (rounded to float32, then to its shortest decimal, so 0.1 stays 0.1),
    ``Reason.percentile`` is ``100 * pct_x`` rounded to 2 decimals (binary features: 100 if the value is 1,
    else 0), ``Reason.shap`` the family total and ``Reason.text`` the rendered template or mitigation.
    All-zero SHAP values give ``((), ())``.

    Raises ``KeyError`` if ``row`` lacks a feature label; ``ValueError`` for ``max_reasons < 0``, a value
    that is not finite, ``phi_row`` not of shape ``(89,)`` or not finite, or a missing template.
    """
    if max_reasons < 0:
        raise ValueError(f"max_reasons must be >= 0, got {max_reasons}")
    values = _row_values(row)
    phi = _as_phi(phi_row)
    return _reasons(values, phi, _family_totals(phi), templates, max_reasons)


def assign_debt_types(row: pd.Series, totals: Mapping[str, float]) -> tuple[str, ...]:
    """Return the debt types of one file: at most ``MAX_DEBT_TYPES`` of ``DEBT_TYPES``, possibly none.

    ``row`` holds the file's values under the labels ``FEATURE_INPUT_COLUMNS``; ``totals`` its SHAP family
    totals (from :func:`family_totals`). Rules, on values rounded to float32 like the model input:

    - complexity: ``cc_max > 15`` or ``nesting_max > 4``; duplication: ``dup_ratio > 0.10``;
    - test: ``test_importers == 0`` and ``has_test_file == 0``;
    - architecture: ``scc_size > 1``, or ``pct_hidden_coupling > 0.9`` with ``hidden_coupling > 0``, or
      ``pct_betweenness > 0.95`` with ``betweenness > 0``;
    - knowledge: ``owner_active == 0``, or ``pct_author_entropy > 0.9`` with ``author_entropy > 0``;
    - admitted: ``satd_count > 0`` and ``pct_satd_density >= 0.9``;
    - defect_hotspot: D and E are the two largest family totals and both are positive.

    Triggered types are ordered by the total of their family (``DEBT_TYPE_FAMILIES``; the larger of D and E
    for defect_hotspot, ties in ``DEBT_TYPES`` order) and the first two are kept. If none triggers, the
    dominant family maps through ``DOMINANT_DEBT_TYPE`` when its total is positive; otherwise ``()``.

    Raises ``KeyError`` if ``row`` lacks a feature label or ``totals`` lacks a family; ``ValueError`` for a
    value or total that is not finite.
    """
    values = _row_values(row)
    tot: dict[str, float] = {}
    for fam in FAMILY_ORDER:
        t = float(totals[fam])
        if not math.isfinite(t):
            raise ValueError(f"family total {fam} is not finite ({t})")
        tot[fam] = t
    return _debt_types(values, tot)


# ---------------------------------------------------------------- internals


def _check_columns(X: pd.DataFrame) -> None:
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


def _as_phi(phi_row: npt.ArrayLike) -> npt.NDArray[np.float64]:
    phi = np.asarray(phi_row, dtype=np.float64)
    if phi.shape != (_N_INPUT,):
        raise ValueError(f"phi_row must have shape ({_N_INPUT},), got {phi.shape}")
    if not np.isfinite(phi).all():
        raise ValueError("phi_row must be finite (no NaN or infinity)")
    return phi


def _row_values(row: pd.Series) -> npt.NDArray[np.float64]:
    values = row.loc[list(FEATURE_INPUT_COLUMNS)].to_numpy(dtype=np.float64)
    if values.shape != (_N_INPUT,):
        raise ValueError("row holds repeated feature labels")
    bad = ~np.isfinite(values)
    if bad.any():
        j = int(np.argmax(bad))
        raise ValueError(f"feature {FEATURE_INPUT_COLUMNS[j]} is not finite ({values[j]})")
    return values


def _clean(value: float) -> float:
    """The shortest decimal of ``value`` rounded to float32 (what the model saw): 0.1 stays 0.1."""
    return float(str(np.float32(value)))


def _family_totals(phi: npt.NDArray[np.float64]) -> dict[str, float]:
    return {fam: float(phi[_FAMILY_INDEX[fam]].sum()) for fam in FAMILY_ORDER}


def _contributions(phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    contrib: npt.NDArray[np.float64] = phi[_RAW_INDEX] + np.where(_HAS_PCT, phi[_PCT_INDEX], 0.0)
    return contrib


def _feature_totals(phi: npt.NDArray[np.float64]) -> dict[str, float]:
    contrib = _contributions(phi)
    return {c: float(contrib[k]) for k, c in enumerate(FEATURE_COLUMNS)}


def _percentile(k: int, value: float, values: npt.NDArray[np.float64]) -> float:
    if not _HAS_PCT[k]:
        return 100.0 if value >= 0.5 else 0.0
    pct = round(100.0 * float(values[_PCT_INDEX[k]]), 2)
    return min(100.0, max(0.0, pct))


def _reason(
    family: str,
    k: int,
    shap: float,
    values: npt.NDArray[np.float64],
    templates: Mapping[str, Any],
    kind: Literal["template", "mitigation"],
) -> Reason:
    feature = FEATURE_COLUMNS[k]
    value = _clean(float(values[_RAW_INDEX[k]]))
    percentile = _percentile(k, value, values)
    text = render_text(templates, feature, kind, value, percentile)
    return Reason(family=family, feature=feature, value=value, percentile=percentile, shap=shap, text=text)


def _reasons(
    values: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    totals: Mapping[str, float],
    templates: Mapping[str, Any],
    max_reasons: int,
) -> tuple[tuple[Reason, ...], tuple[Reason, ...]]:
    contrib = _contributions(phi)
    floor = REASON_MIN_SHARE * float(np.abs(phi).sum())
    reasons: list[Reason] = []
    for fam in sorted(FAMILY_ORDER, key=lambda f: (-totals[f], _FAMILY_RANK[f])):
        total = totals[fam]
        if len(reasons) >= max_reasons or not (total > 0 and total > floor):
            break
        members = _FAMILY_FEATURES[fam]
        k = int(members[int(np.argmax(contrib[members]))])
        reasons.append(_reason(fam, k, total, values, templates, "template"))

    positive = sum(t for t in (totals[f] for f in FAMILY_ORDER) if t > 0)
    low = min(FAMILY_ORDER, key=lambda f: (totals[f], _FAMILY_RANK[f]))
    mitigations: tuple[Reason, ...] = ()
    if totals[low] < 0 and -totals[low] > MITIGATION_MIN_SHARE * positive:
        members = _FAMILY_FEATURES[low]
        k = int(members[int(np.argmin(contrib[members]))])
        mitigations = (_reason(low, k, totals[low], values, templates, "mitigation"),)
    return tuple(reasons), mitigations


def _debt_types(values: npt.NDArray[np.float64], tot: Mapping[str, float]) -> tuple[str, ...]:
    def v(name: str) -> float:
        return _clean(float(values[_POS[name]]))

    order = sorted(FAMILY_ORDER, key=lambda f: (-tot[f], _FAMILY_RANK[f]))
    rules = {
        "complexity": v("cc_max") > _CC_MAX or v("nesting_max") > _NESTING_MAX,
        "duplication": v("dup_ratio") > _DUP_RATIO,
        "test": v("test_importers") == 0 and v("has_test_file") == 0,
        "architecture": (
            v("scc_size") > 1
            or (v("pct_hidden_coupling") > _PCT_HIDDEN_COUPLING and v("hidden_coupling") > 0)
            or (v("pct_betweenness") > _PCT_BETWEENNESS and v("betweenness") > 0)
        ),
        "knowledge": v("owner_active") == 0
        or (v("pct_author_entropy") > _PCT_AUTHOR_ENTROPY and v("author_entropy") > 0),
        "admitted": v("satd_count") > 0 and v("pct_satd_density") >= _PCT_SATD_DENSITY,
        "defect_hotspot": set(order[:2]) == {"D", "E"} and tot["D"] > 0 and tot["E"] > 0,
    }
    triggered = [t for t in DEBT_TYPES if rules[t]]
    if triggered:
        related = {t: max(tot[f] for f in DEBT_TYPE_FAMILIES[t]) for t in triggered}
        triggered.sort(key=lambda t: (-related[t], _DEBT_RANK[t]))
        return tuple(triggered[:MAX_DEBT_TYPES])
    dominant = order[0]
    return (DOMINANT_DEBT_TYPE[dominant],) if tot[dominant] > 0 else ()


def _pick_text(entry: Mapping[str, Any], kind: str, value: float) -> str:
    for pair in entry.get("exact") or ():
        if float(pair[0]) == value:
            sentence = pair[1]
            if not isinstance(sentence, str):
                raise ValueError(f"exact sentence for {pair[0]!r} is not a string")
            return sentence
    text = entry.get(kind)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"no {kind} sentence")
    return text


def _format(text: str, adjectives: Any, value: float, percentile: float) -> str:
    adj = ""
    for threshold, label in adjectives or ():
        if threshold <= percentile:
            adj = label
            break
    top = max(1.0, 100.0 - percentile)
    s = "" if round(value) == 1 else "s"
    return text.format(adj=adj, value=value, percentile=percentile, top=top, s=s)


def _is_number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def _check_line(where: str, key: str, text: str) -> None:
    if not text.strip():
        raise ValueError(f"{where}: {key} renders an empty sentence")
    if not text.isprintable():
        raise ValueError(f"{where}: {key} must be one printable line (no newlines or tabs)")
    bad = sorted(_FORBIDDEN_CHARS.intersection(text))
    if bad:
        raise ValueError(f"{where}: {key} must not contain {' '.join(bad)}")


def _check_fields(where: str, key: str, text: str, has_adjectives: bool) -> None:
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError as exc:
        raise ValueError(f"{where}: {key} is not a valid format string: {exc}") from exc
    for _literal, name, spec, conversion in parsed:
        if name is None:
            continue
        if name not in TEMPLATE_FIELDS:
            raise ValueError(
                f"{where}: {key} uses unknown field {{{name}}}; allowed: {', '.join(sorted(TEMPLATE_FIELDS))}"
            )
        if conversion is not None:
            raise ValueError(f"{where}: {key} must not use a conversion such as !r in {{{name}}}")
        if spec and ("{" in spec or "}" in spec):
            raise ValueError(f"{where}: {key} must not nest fields in a format spec")
        if name == "adj" and not has_adjectives:
            raise ValueError(f"{where}: {key} uses {{adj}} but the entry has no adjectives")


def _validate_adjectives(where: str, adjectives: Any) -> None:
    if not isinstance(adjectives, list | tuple) or not adjectives:
        raise ValueError(f"{where}: adjectives must be a non-empty list of [threshold, text] pairs")
    previous = math.inf
    for k, pair in enumerate(adjectives):
        key = f"adjectives[{k}]"
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise ValueError(f"{where}: {key} must be a [threshold, text] pair")
        threshold, label = pair
        if not _is_number(threshold) or not 0 <= threshold <= 100:
            raise ValueError(f"{where}: {key} threshold must be a number from 0 to 100, got {threshold!r}")
        if threshold >= previous:
            raise ValueError(f"{where}: adjective thresholds must be strictly decreasing")
        if not isinstance(label, str):
            raise ValueError(f"{where}: {key} text must be a string")
        _check_line(where, key, label)
        previous = threshold
    if previous != 0:
        raise ValueError(f"{where}: the last adjective threshold must be 0, got {previous!r}")


def _validate_entry(feature: str, entry: Any) -> None:
    where = f"reason template {feature!r}"
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}: must be a mapping, got {type(entry).__name__}")
    unknown = sorted(map(str, set(entry) - TEMPLATE_KEYS))
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(TEMPLATE_KEYS))}"
        )
    texts: list[tuple[str, str, tuple[float, ...]]] = []
    for key in ("template", "mitigation"):
        text = entry.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}: {key} must be a non-empty string")
        texts.append((key, text, _CHECK_VALUES))
    has_adjectives = "adjectives" in entry
    adjectives = entry.get("adjectives")
    if has_adjectives:
        _validate_adjectives(where, adjectives)
    if "exact" in entry:
        exact = entry["exact"]
        if not isinstance(exact, list | tuple):
            raise ValueError(f"{where}: exact must be a list of [value, sentence] pairs")
        for k, pair in enumerate(exact):
            key = f"exact[{k}]"
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                raise ValueError(f"{where}: {key} must be a [value, sentence] pair")
            value, sentence = pair
            if not _is_number(value):
                raise ValueError(f"{where}: {key} value must be a finite number, got {value!r}")
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError(f"{where}: {key} sentence must be a non-empty string")
            texts.append((key, sentence, (float(value),)))
    for key, text, check_values in texts:
        _check_fields(where, key, text, has_adjectives)
        for value in check_values:
            for percentile in _CHECK_PERCENTILES:
                try:
                    rendered = _format(text, adjectives, value, percentile)
                except (KeyError, IndexError, ValueError, AttributeError, TypeError) as exc:
                    raise ValueError(
                        f"{where}: {key} does not render for value {value} at percentile {percentile}: {exc}"
                    ) from exc
                _check_line(where, key, rendered)

"""Model bundles: the directory format a trained model ships in, plus a synthetic dev bundle.

A bundle (``models/<version>/``) holds only JSON, YAML and Markdown:

- ``ranker.json``: the XGBoost booster (``save_model``), trimmed to its best iteration after early stopping;
- ``calibrator.json``: isotonic thresholds ``{"x", "y", "y_low", "y_high"}``;
- ``feature_schema.json``: ``{"extractor_version", "columns"}``, which must match this code exactly;
- ``reasons.yaml`` and ``cost_model.yaml`` (optional): reason templates and cost constants, copied from the
  package defaults when the trainer gives none, and read from the package defaults when absent;
- ``metadata.json``: ``version``, ``created_at``, ``manifest_sha256``, ``code_git_sha``, ``params``,
  ``metrics``, ``training_rows`` and any extra keys (dev bundles add ``synthetic: true``);
- ``model_card.md`` (optional): written by whoever builds the bundle.

:func:`load_bundle` raises ``FileNotFoundError`` for a missing directory and :class:`BundleSchemaError` for
anything wrong inside it. :func:`make_dev_bundle` trains a small ranker on synthetic data in a few seconds so
the platform can run before the real model exists; such bundles must never be promoted.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import xgboost as xgb
import yaml

from tdp_core.contracts import EXTRACTOR_VERSION, FEATURE_INPUT_COLUMNS
from tdp_core.model.calibration import IsotonicCalibrator
from tdp_core.model.ranker import Ranker
from tdp_core.model.synthetic import synthetic_dataset
from tdp_core.serialize import to_jsonable

logger = logging.getLogger(__name__)

RANKER_FILE: Final = "ranker.json"
CALIBRATOR_FILE: Final = "calibrator.json"
FEATURE_SCHEMA_FILE: Final = "feature_schema.json"
REASONS_FILE: Final = "reasons.yaml"
COST_MODEL_FILE: Final = "cost_model.yaml"
METADATA_FILE: Final = "metadata.json"
MODEL_CARD_FILE: Final = "model_card.md"
REQUIRED_FILES: Final = (RANKER_FILE, CALIBRATOR_FILE, FEATURE_SCHEMA_FILE, METADATA_FILE)
BUNDLE_FILES: Final = (*REQUIRED_FILES, REASONS_FILE, COST_MODEL_FILE, MODEL_CARD_FILE)

VERSION_PATTERN: Final = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
"""Bundle versions are safe directory names and database keys: 1 to 128 letters, digits, ``.``, ``_`` or
``-``, starting and ending with a letter or digit (``ranker-2026.10.1``, ``dev-20261028T101500Z``)."""

METADATA_DEFAULT_KEYS: Final = (
    "created_at",
    "manifest_sha256",
    "code_git_sha",
    "params",
    "metrics",
    "training_rows",
)

DEV_QUERIES: Final = 30
DEV_ROWS_PER_QUERY: Final = 200

_DEV_INCREASING: Final = (
    "commits_decayed",
    "churn_decayed",
    "prior_latent_bugs",
    "transitive_dependents",
    "cc_max",
    "cog_max",
    "nesting_max",
    "longest_function",
    "dup_ratio",
    "satd_count",
    "satd_density",
    "satd_age_days",
    "scc_size",
    "hidden_coupling",
)
_DEV_DECREASING: Final = ("has_test_file", "test_importers", "test_cochange_rate")


def _dev_constraints() -> dict[str, int]:
    signs: dict[str, int] = {}
    for base in _DEV_INCREASING:
        signs[base] = 1
        signs[f"pct_{base}"] = 1
    for base in _DEV_DECREASING:
        signs[base] = -1
        if f"pct_{base}" in FEATURE_INPUT_COLUMNS:
            signs[f"pct_{base}"] = -1
    return {c: signs[c] for c in FEATURE_INPUT_COLUMNS if c in signs}


DEV_MONOTONE_CONSTRAINTS: Final[Mapping[str, int]] = MappingProxyType(_dev_constraints())
"""Monotone constraints of the dev ranker, in ``FEATURE_INPUT_COLUMNS`` order (+1 raises the score, -1
lowers it). The dev ranker can therefore only keep or lower a file's score when tests are added or complexity,
duplication, SATD, cycles or hidden coupling are reduced."""

DEV_RANKER_PARAMS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "objective": "rank:ndcg",
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.2,
        "tree_method": "hist",
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 10,
        "n_jobs": 4,
    }
)
"""``XGBRanker`` parameters of the dev ranker (``random_state`` and the constraints are added at fit time)."""


class BundleSchemaError(ValueError):
    """A bundle directory exists but a file is missing, unreadable or does not match this code."""


@dataclass(frozen=True)
class Bundle:
    """A loaded model bundle. ``version`` comes from ``metadata.json``, never from the directory name."""

    version: str
    path: Path
    ranker: Ranker
    calibrator: IsotonicCalibrator
    reasons: dict[str, Any] = field(repr=False)
    cost_model: dict[str, Any] = field(repr=False)
    metadata: dict[str, Any] = field(repr=False)


def _package_default(bundle_file: str) -> Traversable | None:
    """Return the packaged default for an optional bundle YAML file, or None if it is not installed."""
    if bundle_file == REASONS_FILE:
        ref = files("tdp_core.model") / "reasons.yaml"
    elif bundle_file == COST_MODEL_FILE:
        ref = files("tdp_core.planning") / "cost_defaults.yaml"
    else:
        return None
    return ref if ref.is_file() else None


def load_bundle(path: Path) -> Bundle:
    """Load and validate the bundle in directory ``path``.

    Checks that every required file exists, that ``feature_schema.json`` names ``EXTRACTOR_VERSION`` and
    exactly ``FEATURE_INPUT_COLUMNS`` in order, that ``metadata.json`` is an object with a valid ``version``,
    and that the calibrator and ranker parse. ``reasons`` and ``cost_model`` come from the bundle's YAML
    files, else from the package defaults, else ``{}`` (an empty YAML file also gives ``{}``). The ranker is
    loaded for CPU scoring with 4 threads.

    Raises ``FileNotFoundError`` if ``path`` is not a directory and :class:`BundleSchemaError` (a
    ``ValueError``) naming the file for anything missing, unparsable or inconsistent. Other ``OSError``
    exceptions (for example permissions) propagate unchanged.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"model bundle directory not found: {path}")
    for name in REQUIRED_FILES:
        if not (path / name).is_file():
            raise BundleSchemaError(f"{name}: missing from bundle {path}")

    schema = _read_json(path / FEATURE_SCHEMA_FILE)
    _check_feature_schema(schema)

    metadata = _read_json(path / METADATA_FILE)
    if not isinstance(metadata, dict):
        raise BundleSchemaError(f"{METADATA_FILE}: must be a JSON object")
    version = metadata.get("version")
    if not isinstance(version, str) or not VERSION_PATTERN.match(version):
        raise BundleSchemaError(f"{METADATA_FILE}: invalid version {version!r}")

    payload = _read_json(path / CALIBRATOR_FILE)
    try:
        calibrator = IsotonicCalibrator.from_json(payload)
    except ValueError as exc:
        raise BundleSchemaError(f"{CALIBRATOR_FILE}: {exc}") from exc

    try:
        ranker = Ranker.from_json(path / RANKER_FILE)
    except ValueError as exc:
        raise BundleSchemaError(f"{RANKER_FILE}: {exc}") from exc

    reasons = _load_yaml_mapping(path, REASONS_FILE)
    cost_model = _load_yaml_mapping(path, COST_MODEL_FILE)
    logger.info("loaded model bundle %s from %s", version, path)
    return Bundle(
        version=version,
        path=path,
        ranker=ranker,
        calibrator=calibrator,
        reasons=reasons,
        cost_model=cost_model,
        metadata=metadata,
    )


def save_bundle(
    path: Path,
    *,
    version: str,
    booster: xgb.Booster,
    calibrator: IsotonicCalibrator,
    metadata: Mapping[str, Any],
    reasons: Mapping[str, Any] | None = None,
    cost_model: Mapping[str, Any] | None = None,
) -> Path:
    """Write a bundle into directory ``path`` (created if needed) and return ``path``.

    ``metadata`` is copied and completed: ``version`` is set, and ``created_at`` (now, UTC, whole seconds),
    ``manifest_sha256`` (None), ``code_git_sha`` (None), ``params`` ({}), ``metrics`` ({}) and
    ``training_rows`` (None) are filled in when absent. Datetimes become ISO 8601 strings ending in ``Z``
    and numpy scalars become numbers. An early-stopped booster (``best_iteration`` attribute) is trimmed to
    its first ``best_iteration + 1`` rounds so scoring uses exactly the trees early stopping chose.
    ``reasons`` / ``cost_model`` mappings are written as YAML; when None the package default is copied byte
    for byte (comments kept), or a stale copy in ``path`` is removed if there is no default.

    Every payload is serialised before anything is written, and ``metadata.json`` is written last (an
    existing one is removed first), so a failed save never leaves a bundle that looks complete. Existing
    bundle files are overwritten; ``model_card.md`` is left to the caller.

    Raises ``ValueError`` for an invalid version, a conflicting ``metadata["version"]``, or metadata that is
    not JSON (NaN and infinity are rejected); :class:`BundleSchemaError` if the booster does not take
    exactly ``FEATURE_INPUT_COLUMNS``.
    """
    path = Path(path)
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"invalid bundle version {version!r}: use letters, digits, '.', '_' or '-'")
    if "version" in metadata and metadata["version"] != version:
        raise ValueError(f"metadata version {metadata['version']!r} conflicts with version {version!r}")

    meta: dict[str, Any] = dict(metadata)
    meta["version"] = version
    meta.setdefault("created_at", datetime.now(UTC).replace(microsecond=0))
    meta.setdefault("manifest_sha256", None)
    meta.setdefault("code_git_sha", None)
    meta.setdefault("params", {})
    meta.setdefault("metrics", {})
    meta.setdefault("training_rows", None)
    metadata_text = _metadata_json(meta)

    best = booster.attributes().get("best_iteration")
    if best is not None and int(best) + 1 < booster.num_boosted_rounds():
        rounds = int(best) + 1
        logger.info(
            "trimming early-stopped booster from %d to %d rounds", booster.num_boosted_rounds(), rounds
        )
        booster = booster[:rounds]
    try:
        Ranker(booster)
    except ValueError as exc:
        raise BundleSchemaError(f"{RANKER_FILE}: {exc}") from exc

    calibrator_text = json.dumps(calibrator.to_json(), allow_nan=False)
    schema_text = json.dumps(
        {"extractor_version": EXTRACTOR_VERSION, "columns": list(FEATURE_INPUT_COLUMNS)}, indent=2
    )
    yaml_texts = {
        REASONS_FILE: None if reasons is None else _yaml_text(reasons, REASONS_FILE),
        COST_MODEL_FILE: None if cost_model is None else _yaml_text(cost_model, COST_MODEL_FILE),
    }

    path.mkdir(parents=True, exist_ok=True)
    (path / METADATA_FILE).unlink(missing_ok=True)
    booster.save_model(str(path / RANKER_FILE))
    _write_text(path / CALIBRATOR_FILE, calibrator_text + "\n")
    _write_text(path / FEATURE_SCHEMA_FILE, schema_text + "\n")
    for name, text in yaml_texts.items():
        target = path / name
        if text is not None:
            _write_text(target, text)
            continue
        default = _package_default(name)
        if default is not None:
            target.write_bytes(default.read_bytes())
        else:
            target.unlink(missing_ok=True)
    _write_text(path / METADATA_FILE, metadata_text + "\n")
    logger.info("saved model bundle %s to %s", version, path)
    return path


def make_dev_bundle(path: Path, *, seed: int = 42, version: str | None = None) -> Path:
    """Train a small ranker on synthetic data, save it as a bundle in ``path`` and return ``path``.

    Uses ``synthetic_dataset(seed=seed)`` (30 queries of 200 files), an ``XGBRanker`` with
    ``DEV_RANKER_PARAMS`` and ``DEV_MONOTONE_CONSTRAINTS``, and an isotonic calibrator on
    ``has_latent_defect`` with 200 bootstraps. ``version`` defaults to ``dev-<UTC time>`` such as
    ``dev-20261028T101500Z`` (one-second resolution, no colons, so it is a valid folder name everywhere).
    Metadata records ``synthetic: true`` and the seed, and a ``model_card.md`` says the bundle must never be
    promoted. The same seed gives byte-identical ``ranker.json`` and ``calibrator.json``. Takes a few
    seconds (well under 10 s once XGBoost is imported).

    Raises ``FileExistsError`` without writing anything if ``path`` is a file, or a directory that holds
    files other than bundle files or a ``metadata.json`` that is not a synthetic bundle's; ``ValueError``
    for an invalid ``version``.
    """
    path = Path(path)
    _guard_dev_target(path)
    if version is not None and not VERSION_PATTERN.match(version):
        raise ValueError(f"invalid bundle version {version!r}: use letters, digits, '.', '_' or '-'")

    df = synthetic_dataset(DEV_QUERIES, DEV_ROWS_PER_QUERY, seed=seed)
    X = df[list(FEATURE_INPUT_COLUMNS)]
    model = xgb.XGBRanker(
        **DEV_RANKER_PARAMS,
        monotone_constraints=dict(DEV_MONOTONE_CONSTRAINTS),
        random_state=seed,
    )
    model.fit(X, df["grade"].astype(np.int32), qid=df["qid"])
    booster = model.get_booster()
    calibrator = IsotonicCalibrator.fit(
        Ranker(booster).score(X), df["has_latent_defect"].to_numpy(), seed=seed
    )

    now = datetime.now(UTC).replace(microsecond=0)
    version = version if version is not None else f"dev-{now:%Y%m%dT%H%M%SZ}"
    metadata: dict[str, Any] = {
        "created_at": now,
        "params": {
            **dict(DEV_RANKER_PARAMS),
            "random_state": seed,
            "monotone_constraints": dict(DEV_MONOTONE_CONSTRAINTS),
        },
        "metrics": {},
        "training_rows": len(df),
        "synthetic": True,
        "seed": seed,
        "queries": DEV_QUERIES,
        "rows_per_query": DEV_ROWS_PER_QUERY,
    }
    save_bundle(path, version=version, booster=booster, calibrator=calibrator, metadata=metadata)
    _write_text(path / MODEL_CARD_FILE, _dev_model_card(version, seed, len(df), calibrator))
    return path


def _guard_dev_target(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"{path} exists and is not a directory")
    if not path.is_dir():
        return
    foreign = sorted(p.name for p in path.iterdir() if p.name not in BUNDLE_FILES)
    if foreign:
        raise FileExistsError(
            f"refusing to write a dev bundle into {path}: it holds {', '.join(foreign[:5])}"
        )
    meta_path = path / METADATA_FILE
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
        if not (isinstance(meta, dict) and meta.get("synthetic") is True):
            raise FileExistsError(f"refusing to overwrite the non-synthetic bundle in {path}")


def _dev_model_card(version: str, seed: int, rows: int, calibrator: IsotonicCalibrator) -> str:
    return f"""# Model card: {version}

## Intended use

Development only. This bundle lets the analysis pipeline, worker, API, CLI and dashboard run before a real
model exists. It must never be promoted to production or used to make decisions about real code.

## Data

Synthetic data from `tdp_core.model.synthetic.synthetic_dataset` with seed {seed}: {DEV_QUERIES} queries of
{DEV_ROWS_PER_QUERY} files ({rows} rows). Grades are driven mainly by recent churn, prior latent bugs and
transitive dependents, with smaller effects from complexity, duplication, self-admitted debt, import cycles
and tests. No real repository was used.

## Metrics

None. The ranker was trained and calibrated in-sample on synthetic data ({calibrator.x.size} calibration
thresholds), so any score it produces says nothing about real maintenance pain.

## Limitations

Scores, priorities, risks and reasons are placeholders. The ranker is monotone in the features the what-if
actions change, which makes it handy for tests, but its learned effects are artefacts of the generator.
"""


def _check_feature_schema(schema: Any) -> None:
    name = FEATURE_SCHEMA_FILE
    if not isinstance(schema, dict):
        raise BundleSchemaError(f"{name}: must be a JSON object")
    if schema.get("extractor_version") != EXTRACTOR_VERSION:
        raise BundleSchemaError(
            f"{name}: extractor_version {schema.get('extractor_version')!r} does not match "
            f"{EXTRACTOR_VERSION!r}; retrain the model for this code"
        )
    columns = schema.get("columns")
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise BundleSchemaError(f"{name}: columns must be a list of strings")
    expected = list(FEATURE_INPUT_COLUMNS)
    if columns != expected:
        missing = sorted(set(expected) - set(columns))
        extra = sorted(set(columns) - set(expected))
        if missing or extra:
            raise BundleSchemaError(f"{name}: {len(missing)} columns missing, {len(extra)} unexpected")
        raise BundleSchemaError(f"{name}: columns are not in FEATURE_INPUT_COLUMNS order")


def _read_json(file: Path) -> Any:
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BundleSchemaError(f"{file.name}: not valid UTF-8 JSON ({exc})") from exc


def _load_yaml_mapping(path: Path, name: str) -> dict[str, Any]:
    local = path / name
    source: Path | Traversable | None = local if local.is_file() else _package_default(name)
    if source is None:
        logger.debug("%s: not in bundle and no package default; using {}", name)
        return {}
    where = "bundle" if source is local else "package default"
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (ValueError, yaml.YAMLError) as exc:
        raise BundleSchemaError(f"{name}: not valid UTF-8 YAML ({where}): {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise BundleSchemaError(f"{name}: must be a YAML mapping ({where}), got {type(data).__name__}")
    logger.debug("%s: loaded from the %s", name, where)
    return data


def _metadata_json(meta: dict[str, Any]) -> str:
    try:
        plain = to_jsonable(meta)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata must hold only JSON values: {exc}") from exc
    try:
        return json.dumps(plain, indent=2, sort_keys=True, allow_nan=False)
    except ValueError as exc:
        raise ValueError("metadata must be JSON without NaN or Infinity") from exc


def _yaml_text(mapping: Mapping[str, Any], name: str) -> str:
    try:
        plain = to_jsonable(dict(mapping))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: must hold only plain values: {exc}") from exc
    text: str = yaml.safe_dump(plain, sort_keys=False, allow_unicode=True)
    return text


def _write_text(file: Path, text: str) -> None:
    file.write_text(text, encoding="utf-8", newline="\n")

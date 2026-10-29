"""Model registry: build production bundles from MLflow runs, check them with gates and publish them.

Three operations, each behind a ``tdp-ml`` command (:mod:`tdp_ml.commands.registry`):

- :func:`register_run` downloads the booster, calibrator and tuned parameters of a finished LOPO training
  run, writes a bundle with :func:`tdp_core.model.bundle.save_bundle` into
  ``$TDP_MODELS_DIR/ranker-YYYY.MM.N/`` together with ``model_card.md``, packs it into a deterministic
  ``ranker-YYYY.MM.N.tar.gz`` and writes a registry record ``registry/ranker-YYYY.MM.N.json`` (sha256 of the
  tarball and of every file, run ids, the full run metrics, the promotion history).
- :func:`run_golden` scores the pinned golden repositories (``ml/configs/golden.yaml``) with a bundle and
  checks every result: the analysed sha, priorities in 0..100 with matching bands, ordered risk ranges in
  0..1, a reason for the top file and SHAP values that add up to the scores.
- :func:`promote` runs the seven gates of :data:`GATES` and, only when each passes or is waived on the
  record, writes the production pointer files ``{"version", "uri", "sha256"}``. Gates reuse the metrics the
  training and calibration runs stored; nothing is retrained or rescored except the golden repositories.

MLflow is only read, never written. JSON files are written atomically. Timestamps are UTC strings
``YYYY-MM-DDTHH:MM:SSZ``. MLflow, the analysis pipeline, SHAP and the leakage checker are imported inside the
functions that need them.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse
from urllib.request import url2pathname

import pandas as pd
import yaml

from tdp_core.contracts import (
    EXTRACTOR_VERSION,
    FEATURE_COLUMNS,
    FEATURE_INPUT_COLUMNS,
    band_for,
)
from tdp_core.model.bundle import (
    METADATA_FILE,
    MODEL_CARD_FILE,
    VERSION_PATTERN,
    Bundle,
    load_bundle,
    save_bundle,
)
from tdp_core.model.calibration import IsotonicCalibrator
from tdp_core.model.ranker import Ranker
from tdp_ml.calibrate import ECE_TARGET, MODEL_NAME
from tdp_ml.pipeline.config import DEFAULT_CONFIG_DIR, ConfigError, slug_for
from tdp_ml.pipeline.layout import DataLayout, PipelineError, read_json, write_json
from tdp_ml.train import TunedParams, code_git_sha, file_sha256, load_params, monotone_constraints

if TYPE_CHECKING:
    from tdp_core.contracts import AnalysisResult
    from tdp_core.model.explain import Explainer

logger = logging.getLogger(__name__)

VERSION_RE: Final = re.compile(r"^ranker-(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<n>[1-9]\d*)$")
"""Production versions: ``ranker-YYYY.MM.N`` with the UTC month of registration and a counter from 1."""

RECORD_DIR: Final = "registry"
RECORD_FORMAT: Final = 1
REPORT_FORMAT: Final = 1
TARBALL_SUFFIX: Final = ".tar.gz"
PRODUCTION_FILE: Final = "production.json"
POINTER_KEYS: Final = ("version", "uri", "sha256")
DEFAULT_GOLDEN_CONFIG: Final[Path] = DEFAULT_CONFIG_DIR / "golden.yaml"

HOTSPOT_ALPHA: Final = 0.05
MAX_ECE: Final = ECE_TARGET
LEAKAGE_MIN_QUERIES: Final = 10
ADDITIVITY_TOL: Final = 1e-4
MAX_GOLDEN_ERRORS: Final = 5

GATES: Final = ("real_model", "integrity", "incumbent", "hotspot", "calibration", "leakage", "golden")
WAIVABLE_GATES: Final = frozenset({"incumbent", "hotspot", "calibration"})
"""Only the statistical gates can be waived (with a recorded reason); the others always block."""

_RANKING_KEYS: Final = (
    "ndcg_10",
    "ndcg_10_ci_low",
    "ndcg_10_ci_high",
    "precision_10",
    "recall_20loc",
    "popt",
    "hotspot_ndcg_10",
    "ndcg_10_p_greater_hotspot",
)
HEADLINE_METRICS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "lopo": _RANKING_KEYS,
        "temporal": _RANKING_KEYS,
        "calibration": ("ece", "ece_ci_low", "ece_ci_high", "brier", "brier_skill", "auc_pr"),
    }
)
"""The metrics copied into ``metadata.json``: ``lopo.k`` is LOPO run metric ``lopo_<k>``, ``temporal.k``
the temporal run's ``temporal_<k>`` and ``calibration.k`` the LOPO run's ``calibration_<k>``."""

_METRIC_PREFIX: Final = {"lopo": "lopo_", "temporal": "temporal_", "calibration": "calibration_"}
_SHA40: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_PATH: Final = re.compile(r"^[A-Za-z]:[\\/]")
_GOLDEN_TOP_KEYS: Final = frozenset({"repos"})
_GOLDEN_REPO_KEYS: Final = frozenset({"name", "sha"})
_MAX_REASON_CHARS: Final = 500


class RegistryError(RuntimeError):
    """Bad registry input: an unknown or unsuitable run, a version clash, a broken record or tarball."""


# ---------------------------------------------------------------------------------------------------------
# data classes


@dataclass(frozen=True, slots=True)
class RunInfo:
    """What the registry needs from a LOPO training run and its temporal sibling.

    ``params`` and ``tags`` are MLflow strings; ``metrics`` and ``temporal_metrics`` hold only the latest
    finite values. ``temporal_run_id`` is None when no temporal run was found.
    """

    run_id: str
    experiment_id: str
    run_name: str
    tracking_uri: str
    model: str
    params: dict[str, str]
    metrics: dict[str, float]
    tags: dict[str, str]
    temporal_run_id: str | None
    temporal_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class RunFiles:
    """Local copies of a run's ``model/ranker.json``, ``model/calibrator.json`` and tuned params file."""

    info: RunInfo
    ranker: Path
    calibrator: Path
    params: Path


@dataclass(frozen=True, slots=True)
class Registration:
    """A registered bundle. ``created`` is False when the run was already registered and nothing changed."""

    version: str
    bundle: Path
    tarball: Path
    sha256: str
    size_bytes: int
    record: Path
    created: bool


@dataclass(frozen=True, slots=True)
class GoldenRepo:
    """A pinned golden repository: ``owner/name`` and a full 40-hex commit sha."""

    name: str
    sha: str

    @property
    def slug(self) -> str:
        """The data slug (``owner__name``), which names the mirror ``mirrors/<slug>.git``."""
        return slug_for(self.name)

    @property
    def url(self) -> str:
        """The GitHub URL of the repository."""
        return f"https://github.com/{self.name}"


@dataclass(frozen=True, slots=True)
class GateResult:
    """The outcome of one promotion gate.

    ``value`` is the number the gate tested (None when it is missing), ``threshold`` says what passes,
    ``detail`` explains the outcome in one line. ``waived`` is True only for a failed waivable gate that the
    promotion waived on the record.
    """

    name: str
    passed: bool
    value: float | None
    threshold: str
    detail: str
    waivable: bool
    waived: bool = False

    def to_json(self) -> dict[str, Any]:
        """The fields as a JSON object (``value`` None for a missing or non-finite number)."""
        value = self.value if self.value is not None and math.isfinite(self.value) else None
        return {
            "name": self.name,
            "passed": self.passed,
            "value": value,
            "threshold": self.threshold,
            "detail": self.detail,
            "waivable": self.waivable,
            "waived": self.waived,
        }


@dataclass(frozen=True, eq=False, slots=True)
class PromotionOutcome:
    """The result of :func:`promote`. ``promoted`` is True only when the pointer files were written."""

    version: str
    sha256: str
    gates: tuple[GateResult, ...]
    promoted: bool
    dry_run: bool
    pointer: dict[str, Any]
    previous: dict[str, Any] | None
    waivers: tuple[str, ...]
    reason: str | None
    written: tuple[Path, ...]
    report: Path
    golden_report: Path | None
    created_at: str

    @property
    def failed(self) -> tuple[GateResult, ...]:
        """Gates that failed and were not waived (empty when the candidate is accepted)."""
        return tuple(g for g in self.gates if not g.passed and not g.waived)

    @property
    def accepted(self) -> bool:
        """True when every gate passed or was waived."""
        return not self.failed

    def to_json(self) -> dict[str, Any]:
        """The promotion report written to :func:`promotion_report_path`."""
        return {
            "format": REPORT_FORMAT,
            "created_at": self.created_at,
            "version": self.version,
            "sha256": self.sha256,
            "promoted": self.promoted,
            "dry_run": self.dry_run,
            "accepted": self.accepted,
            "gates": [g.to_json() for g in self.gates],
            "failed": [g.name for g in self.failed],
            "waivers": list(self.waivers),
            "reason": self.reason,
            "previous": self.previous,
            "pointer": self.pointer,
            "written": [p.as_posix() for p in self.written],
            "golden_report": self.golden_report.as_posix() if self.golden_report is not None else None,
        }


# ---------------------------------------------------------------------------------------------------------
# small helpers


def _stamp(when: datetime | None = None) -> str:
    moment = (when or datetime.now(UTC)).astimezone(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _remove_tree(path: Path) -> None:
    """Remove a folder, retrying read-only files (git and Windows mark some files read-only)."""

    def retry(func: Callable[[str], Any], name: str, _exc: BaseException) -> None:
        os.chmod(name, stat.S_IWRITE)
        func(name)

    if path.is_dir():
        shutil.rmtree(path, onexc=retry)


def _write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _local_dir(uri: str) -> Path | None:
    """The local folder a pointer ``uri`` names (absolute path, drive path or ``file:`` URI), else None."""
    if _WINDOWS_PATH.match(uri):
        return Path(uri)
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    if parsed.scheme == "" and Path(uri).is_absolute():
        return Path(uri)
    return None


# ---------------------------------------------------------------------------------------------------------
# versions and paths


def parse_version(version: str) -> tuple[int, int, int]:
    """``(year, month, n)`` of a ``ranker-YYYY.MM.N`` version.

    Raises :class:`RegistryError` unless the string matches :data:`VERSION_RE` with ``year >= 2000`` and
    ``1 <= month <= 12`` (``N`` starts at 1, without leading zeros).
    """
    match = VERSION_RE.match(version) if isinstance(version, str) else None
    if match is None:
        raise RegistryError(f"invalid version {version!r}: expected ranker-YYYY.MM.N")
    year, month, n = int(match["year"]), int(match["month"]), int(match["n"])
    if year < 2000 or not 1 <= month <= 12:
        raise RegistryError(f"invalid version {version!r}: bad year or month")
    return year, month, n


def bundle_dir(models_dir: Path, version: str) -> Path:
    """``models_dir/<version>``: the bundle folder (the version is validated)."""
    parse_version(version)
    return Path(models_dir) / version


def tarball_path(models_dir: Path, version: str) -> Path:
    """``models_dir/<version>.tar.gz``: the release tarball (the version is validated)."""
    parse_version(version)
    return Path(models_dir) / f"{version}{TARBALL_SUFFIX}"


def record_path(models_dir: Path, version: str) -> Path:
    """``models_dir/registry/<version>.json``: the registry record (the version is validated)."""
    parse_version(version)
    return Path(models_dir) / RECORD_DIR / f"{version}.json"


def _known_versions(models_dir: Path) -> set[str]:
    names: set[str] = set()
    models_dir = Path(models_dir)
    if models_dir.is_dir():
        for entry in models_dir.iterdir():
            if entry.is_dir():
                names.add(entry.name)
            elif entry.name.endswith(TARBALL_SUFFIX):
                names.add(entry.name[: -len(TARBALL_SUFFIX)])
    records = models_dir / RECORD_DIR
    if records.is_dir():
        names.update(p.stem for p in records.glob("*.json"))
    return {n for n in names if VERSION_RE.match(n)}


def next_version(models_dir: Path, *, now: datetime | None = None) -> str:
    """The next free ``ranker-YYYY.MM.N`` for the UTC month of ``now`` (default: the current time).

    ``N`` is one more than the highest ``N`` of that month among the bundle folders, tarballs and registry
    records in ``models_dir``, or 1. Other months and names that are not versions are ignored, and a
    missing ``models_dir`` gives ``.1``.
    """
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    highest = 0
    for name in _known_versions(models_dir):
        try:
            year, month, n = parse_version(name)
        except RegistryError:
            continue
        if (year, month) == (moment.year, moment.month):
            highest = max(highest, n)
    return f"ranker-{moment.year:04d}.{moment.month:02d}.{highest + 1}"


# ---------------------------------------------------------------------------------------------------------
# tarball


def _tar_info(name: str, kind: bytes, mode: int, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.mode = mode
    info.size = size
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def write_tarball(folder: Path, dest: Path) -> str:
    """Pack ``folder`` into a deterministic gzip tarball at ``dest`` and return its sha256 (hex).

    The archive is USTAR with the member ``<folder name>/`` (mode 755) followed by every file sorted by name
    (``<folder name>/<file>``, mode 644), all with mtime 0, uid and gid 0 and empty owner names; gzip uses
    level 9, mtime 0 and no file name. The same files therefore always give the same bytes and sha256 (for
    one zlib version). ``dest`` is written through a temporary file and ``os.replace``.

    Raises :class:`RegistryError` when ``folder`` is not a folder or holds anything but regular files (a
    subfolder or a symbolic link).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise RegistryError(f"bundle folder not found: {folder}")
    entries = sorted(folder.iterdir(), key=lambda p: p.name)
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise RegistryError(f"{entry}: a bundle holds only regular files")
    root = folder.name
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        tar.addfile(_tar_info(root, tarfile.DIRTYPE, 0o755, 0))
        for entry in entries:
            data = entry.read_bytes()
            tar.addfile(
                _tar_info(f"{root}/{entry.name}", tarfile.REGTYPE, 0o644, len(data)), io.BytesIO(data)
            )
    packed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=packed, compresslevel=9, mtime=0) as gz:
        gz.write(raw.getvalue())
    payload = packed.getvalue()
    _write_bytes(Path(dest), payload)
    return hashlib.sha256(payload).hexdigest()


def read_tarball(path: Path, *, root: str | None = None) -> dict[str, bytes]:
    """The files of a bundle tarball as ``{file name: bytes}`` (names without the top folder).

    Raises :class:`RegistryError` when the file is missing or unreadable, or a member has an absolute name,
    a ``..`` part or a backslash, is a link or device, sits in a nested folder or repeats, when there is
    not exactly one top folder, or when ``root`` is given and the top folder has another name.
    """
    path = Path(path)
    if not path.is_file():
        raise RegistryError(f"tarball not found: {path}")
    files: dict[str, bytes] = {}
    roots: set[str] = set()
    try:
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                name = member.name
                pure = PurePosixPath(name)
                if not name or pure.is_absolute() or ".." in pure.parts or "\\" in name:
                    raise RegistryError(f"{path}: unsafe member name {name!r}")
                if member.isdir():
                    if len(pure.parts) != 1:
                        raise RegistryError(f"{path}: nested folder {name!r}")
                    roots.add(pure.parts[0])
                    continue
                if not member.isfile():
                    raise RegistryError(f"{path}: {name!r} is not a regular file")
                if len(pure.parts) != 2:
                    raise RegistryError(f"{path}: {name!r} is not directly inside the top folder")
                roots.add(pure.parts[0])
                if pure.parts[1] in files:
                    raise RegistryError(f"{path}: {name!r} appears twice")
                handle = tar.extractfile(member)
                if handle is None:
                    raise RegistryError(f"{path}: cannot read {name!r}")
                files[pure.parts[1]] = handle.read()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise RegistryError(f"{path}: not a readable gzip tarball ({exc})") from exc
    if len(roots) != 1:
        raise RegistryError(f"{path}: expected one top folder, found {sorted(roots)}")
    if root is not None and roots != {root}:
        raise RegistryError(f"{path}: top folder is {next(iter(roots))!r}, expected {root!r}")
    return files


# ---------------------------------------------------------------------------------------------------------
# MLflow


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in sorted(metrics):
        value = _finite(metrics[key])
        if value is not None:
            out[str(key)] = value
    return out


def fetch_run(
    run_id: str, dest: Path, *, uri: str | None = None, temporal_run_id: str | None = None
) -> RunFiles:
    """Download what a bundle needs from LOPO training run ``run_id`` into ``dest``.

    The run must be FINISHED, tagged ``protocol=lopo`` (its booster is trained on every row) and tagged
    with a model name, and hold ``model/ranker.json``, ``model/calibrator.json`` (added by ``tdp-ml
    calibrate``) and exactly one ``results/*_params.json``. The temporal run is ``temporal_run_id`` when given
    (it must be a FINISHED temporal run of the same model), else the newest finished run tagged with the same
    model and ``protocol=temporal`` in the same experiment, else None with a warning. ``uri`` defaults to
    ``MLFLOW_TRACKING_URI`` through :func:`tdp_ml.tracking.configure_mlflow`. Nothing is written to MLflow.

    Raises :class:`RegistryError` for an unknown run or any requirement that does not hold.
    """
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    from tdp_ml.tracking import configure_mlflow

    used = configure_mlflow(uri)
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=used)
    try:
        run = client.get_run(run_id)
    except MlflowException as exc:
        raise RegistryError(f"MLflow run {run_id} not found in {used}") from exc
    tags = {str(k): str(v) for k, v in run.data.tags.items()}
    status = str(run.info.status)
    if status != "FINISHED":
        raise RegistryError(f"run {run_id} is {status}; only FINISHED runs can be registered")
    protocol = tags.get("protocol")
    if protocol != "lopo":
        raise RegistryError(
            f"run {run_id} is a {protocol} run; register the LOPO run, whose model is trained on every row"
        )
    model = tags.get("model", "")
    if not MODEL_NAME.match(model):
        raise RegistryError(f"run {run_id} has no valid model tag (got {model!r})")

    model_files = {str(f.path) for f in client.list_artifacts(run_id, "model")}
    if "model/ranker.json" not in model_files:
        raise RegistryError(f"run {run_id} has no model/ranker.json artifact")
    if "model/calibrator.json" not in model_files:
        raise RegistryError(
            f"run {run_id} has no model/calibrator.json; run `tdp-ml calibrate --model {model}` first"
        )
    params_files = sorted(
        str(f.path) for f in client.list_artifacts(run_id, "results") if str(f.path).endswith("_params.json")
    )
    if len(params_files) != 1:
        raise RegistryError(f"run {run_id} must hold exactly one results/*_params.json, found {params_files}")

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    local = {
        key: Path(client.download_artifacts(run_id, key, str(dest)))
        for key in ("model/ranker.json", "model/calibrator.json", params_files[0])
    }

    experiment_id = str(run.info.experiment_id)
    temporal_id, temporal_metrics = _temporal_run(client, experiment_id, model, temporal_run_id)
    info = RunInfo(
        run_id=str(run.info.run_id),
        experiment_id=experiment_id,
        run_name=str(run.info.run_name or ""),
        tracking_uri=used,
        model=model,
        params={str(k): str(v) for k, v in run.data.params.items()},
        metrics=_finite_metrics(run.data.metrics),
        tags=tags,
        temporal_run_id=temporal_id,
        temporal_metrics=temporal_metrics,
    )
    return RunFiles(
        info=info,
        ranker=local["model/ranker.json"],
        calibrator=local["model/calibrator.json"],
        params=local[params_files[0]],
    )


def _temporal_run(
    client: Any, experiment_id: str, model: str, run_id: str | None
) -> tuple[str | None, dict[str, float]]:
    from mlflow.exceptions import MlflowException

    if run_id is not None:
        try:
            run = client.get_run(run_id)
        except MlflowException as exc:
            raise RegistryError(f"temporal run {run_id} not found") from exc
        tags = run.data.tags
        if str(run.info.status) != "FINISHED" or tags.get("protocol") != "temporal":
            raise RegistryError(f"run {run_id} is not a FINISHED temporal run")
        if tags.get("model") != model:
            raise RegistryError(f"temporal run {run_id} is for model {tags.get('model')!r}, not {model!r}")
        return str(run.info.run_id), _finite_metrics(run.data.metrics)
    found = client.search_runs(
        [experiment_id],
        f"tags.model = '{model}' and tags.protocol = 'temporal' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not found:
        logger.warning("no finished temporal run of %s in experiment %s", model, experiment_id)
        return None, {}
    return str(found[0].info.run_id), _finite_metrics(found[0].data.metrics)


# ---------------------------------------------------------------------------------------------------------
# metadata and model card


def collect_metrics(info: RunInfo) -> dict[str, dict[str, float]]:
    """The :data:`HEADLINE_METRICS` of a run as ``{"lopo": {...}, "temporal": {...}, "calibration": {...}}``.

    The three groups are always present; a metric that is missing or not finite is left out, never NaN.
    """
    out: dict[str, dict[str, float]] = {}
    for group, keys in HEADLINE_METRICS.items():
        source = info.temporal_metrics if group == "temporal" else info.metrics
        prefix = _METRIC_PREFIX[group]
        values: dict[str, float] = {}
        for key in keys:
            value = _finite(source.get(prefix + key))
            if value is not None:
                values[key] = value
        out[group] = values
    return out


def _int_param(params: Mapping[str, str], key: str) -> int | None:
    raw = params.get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RegistryError(f"run param {key}={raw!r} is not an integer") from exc


def _monotone_columns(params: Mapping[str, str]) -> list[str]:
    if params.get("monotone") != "true":
        return []
    raw = params.get("monotone_columns", "")
    try:
        columns = json.loads(raw)
    except ValueError as exc:
        raise RegistryError(f"run param monotone_columns is not JSON: {raw!r}") from exc
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(c, str) and c in FEATURE_INPUT_COLUMNS for c in columns)
    ):
        raise RegistryError(f"run param monotone_columns must list model input columns, got {raw!r}")
    return [str(c) for c in columns]


def bundle_metadata(
    info: RunInfo, tuned: TunedParams, *, created_at: datetime, registry_code_sha: str
) -> dict[str, Any]:
    """The ``metadata.json`` content of a production bundle (``save_bundle`` adds ``version``).

    ``params`` are the tuned XGBRanker parameters plus ``monotone_constraints`` (``{column: 1}``) when the
    run trained with monotone constraints; ``metrics`` is :func:`collect_metrics`; ``training_rows`` and
    ``training_queries`` come from the run params, ``dataset_sha256``, ``manifest_sha256`` and
    ``code_git_sha`` from its tags; ``source`` names the store, experiment, runs and devices.

    Raises :class:`RegistryError` when ``training_rows`` is missing or not an integer, or when the monotone
    columns are not model input columns.
    """
    params = info.params
    rows = _int_param(params, "training_rows")
    if rows is None:
        raise RegistryError(f"run {info.run_id} has no training_rows param")
    model_params: dict[str, Any] = dict(tuned.params)
    columns = _monotone_columns(params)
    if columns:
        model_params["monotone_constraints"] = monotone_constraints(columns)
    tags = info.tags
    return {
        "created_at": _stamp(created_at),
        "manifest_sha256": tags.get("manifest_sha256"),
        "code_git_sha": tags.get("code_git_sha"),
        "params": model_params,
        "metrics": collect_metrics(info),
        "training_rows": rows,
        "training_queries": _int_param(params, "training_queries"),
        "dataset_sha256": tags.get("dataset_sha256"),
        "horizon_days": _int_param(params, "horizon_days"),
        "model": info.model,
        "monotone": bool(columns),
        "synthetic": False,
        "source": {
            "tracking_uri": info.tracking_uri,
            "experiment_id": info.experiment_id,
            "run_id": info.run_id,
            "run_name": info.run_name,
            "temporal_run_id": info.temporal_run_id,
            "tuned_in_run": tags.get("tuned_in_run"),
            "calibration_code_git_sha": tags.get("calibration_code_git_sha"),
            "device": tags.get("device"),
        },
        "registry_code_git_sha": registry_code_sha,
    }


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _with_ci(metrics: Mapping[str, float], key: str, digits: int = 3) -> str:
    value = _finite(metrics.get(key))
    if value is None:
        return "n/a"
    low, high = _finite(metrics.get(f"{key}_ci_low")), _finite(metrics.get(f"{key}_ci_high"))
    if low is None or high is None:
        return _num(value, digits)
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _ascii(text: str) -> str:
    cleaned = text.replace(chr(0x2014), ", ").replace(chr(0x2013), "-")
    return cleaned.encode("ascii", "replace").decode("ascii")


def model_card(
    version: str,
    metadata: Mapping[str, Any],
    info: RunInfo,
    *,
    dataset: Mapping[str, Any] | None = None,
) -> str:
    """The ``model_card.md`` text of a production bundle: intended use, model, data, evaluation, limits.

    ``metadata`` is the bundle's ``metadata.json``; numbers come from the run's full metric dictionaries
    (``info``), and missing numbers print ``n/a``. ``dataset`` is the dataset summary (``dataset.json``);
    when given, the training repositories are listed. The text is ASCII only, with no em or en dashes, and
    ends with a newline. Promotion gates are deliberately not part of the card (the registry record holds
    them), so the card never changes after the tarball is built.
    """
    params: Mapping[str, Any] = metadata.get("params") or {}
    horizon = metadata.get("horizon_days")
    horizon_text = f"{horizon} days" if horizon is not None else "the label horizon"
    lopo = {k[5:]: v for k, v in info.metrics.items() if k.startswith("lopo_")}
    temporal = {k[9:]: v for k, v in info.temporal_metrics.items() if k.startswith("temporal_")}
    calib = {k[12:]: v for k, v in info.metrics.items() if k.startswith("calibration_")}
    constraints = params.get("monotone_constraints") or {}
    lines: list[str] = [
        f"# Entropiq risk model {version}",
        "",
        "This model ranks the Python files of one repository snapshot by how much maintenance pain each is "
        f"likely to cause over the next {horizon_text}. It was registered at {metadata.get('created_at')} "
        f"from MLflow run {info.run_id} (model `{info.model}`).",
        "",
        "## Intended use",
        "",
        "- Ranks the Python files of one repository snapshot by expected maintenance pain over "
        f"{horizon_text}.",
        "- Priority is a percentile within the repository (0 to 100), so it compares files of the same "
        "snapshot, not files of different repositories.",
        "- Risk is a calibrated probability that the file holds a latent defect that is fixed within the "
        "horizon, shown with a 90% band.",
        "- It is decision support for planning work on code. It is not a measure of people, and it is not "
        "meant for code other than Python.",
        "",
        "## Model",
        "",
        f"- XGBoost LambdaMART (objective `{params.get('objective', 'n/a')}`), "
        f"{params.get('n_estimators', 'n/a')} rounds, max depth {params.get('max_depth', 'n/a')}, "
        f"learning rate {_num(_finite(params.get('learning_rate')), 4)}.",
        f"- {len(FEATURE_INPUT_COLUMNS)} inputs: {len(FEATURE_COLUMNS)} features and their within-repository "
        f"percentiles (feature extractor version {EXTRACTOR_VERSION}).",
    ]
    if constraints:
        bases = [c for c in constraints if not str(c).startswith("pct_")]
        lines.append(f"- Monotone increasing in: {', '.join(bases)} and their percentile copies.")
    else:
        lines.append("- No monotone constraints.")
    lines += [
        "- Isotonic calibration of the ranker score into risk, with bands from a repository bootstrap.",
        "",
        "## Training data",
        "",
        f"- {metadata.get('training_rows', 'n/a')} files in {metadata.get('training_queries', 'n/a')} "
        "repository snapshots.",
        f"- Dataset sha256 `{metadata.get('dataset_sha256')}`, manifest sha256 "
        f"`{metadata.get('manifest_sha256')}`.",
        f"- Labels: SZZ latent defects fixed in the next {horizon_text} plus future churn, turned into "
        "grades 0 to 4 within each snapshot.",
    ]
    if dataset is not None:
        passed = [r for r in dataset.get("per_repo", []) if isinstance(r, Mapping) and r.get("passed")]
        lines.append(f"- {len(passed)} repositories:")
        for repo in sorted(passed, key=lambda r: str(r.get("slug"))):
            lines.append(
                f"  - {repo.get('slug')} at {str(repo.get('sha', ''))[:12]}, {repo.get('rows')} rows"
            )
    lines += [
        "",
        "## Evaluation",
        "",
        "LOPO (leave one project out) trains on all repositories but one and tests on the last snapshot of "
        "the held-out repository. Temporal trains on the earlier snapshots of every repository and tests on "
        "each repository's last snapshot. Values are means with 95% repository bootstrap intervals.",
        "",
        "| protocol | NDCG@10 | P@10 | Recall@20%LOC | Popt | hotspot NDCG@10 | one-sided p vs hotspot |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, metrics in (("LOPO", lopo), ("temporal", temporal)):
        lines.append(
            f"| {label} | {_with_ci(metrics, 'ndcg_10')} | {_with_ci(metrics, 'precision_10')} | "
            f"{_with_ci(metrics, 'recall_20loc')} | {_with_ci(metrics, 'popt')} | "
            f"{_num(_finite(metrics.get('hotspot_ndcg_10')))} | "
            f"{_num(_finite(metrics.get('ndcg_10_p_greater_hotspot')), 4)} |"
        )
    lines += [
        "",
        f"Calibration (temporal test set, pooled): ECE {_with_ci(calib, 'ece', 4)}, Brier "
        f"{_with_ci(calib, 'brier', 4)}, Brier skill {_num(_finite(calib.get('brier_skill')))}, AUC-PR "
        f"{_with_ci(calib, 'auc_pr')}, per-query ECE {_num(_finite(calib.get('query_ece')), 4)}.",
        "",
        "## Limitations",
        "",
    ]
    p = _finite(lopo.get("ndcg_10_p_greater_hotspot"))
    if p is None or p >= HOTSPOT_ALPHA:
        lines.append(
            "- Does not significantly beat the hotspot heuristic: LOPO NDCG@10 "
            f"{_num(_finite(lopo.get('ndcg_10')))} vs {_num(_finite(lopo.get('hotspot_ndcg_10')))}, "
            f"one-sided Wilcoxon p {_num(p, 4)}."
        )
    else:
        lines.append(f"- Beats the hotspot heuristic on LOPO NDCG@10 (one-sided Wilcoxon p {_num(p, 4)}).")
    lines += [
        "- Risk is calibrated on the pooled population of files (per-query ECE "
        f"{_num(_finite(calib.get('query_ece')), 4)}): within one repository it can be off by several "
        "points.",
        "- Bug fixes are found from commit keywords and issue labels, so the labels undercount defects.",
        "- Training mines each repository's whole history, while an analysis mines the last three years.",
        "- At the extremes of the isotonic curve risk is exactly 0 or 1, and the band collapses there.",
        "- Python only.",
        "",
        "## Provenance",
        "",
        f"- MLflow store `{info.tracking_uri}`, experiment {info.experiment_id}.",
        f"- LOPO run {info.run_id}, temporal run {info.temporal_run_id or 'n/a'}, tuned in run "
        f"{info.tags.get('tuned_in_run') or 'n/a'}.",
        f"- Training code {metadata.get('code_git_sha') or 'n/a'}, calibration code "
        f"{info.tags.get('calibration_code_git_sha') or 'n/a'}, registry code "
        f"{metadata.get('registry_code_git_sha') or 'n/a'}.",
        "- Promotion gates are recorded in the registry, not in this card.",
        "",
    ]
    return _ascii("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------
# registration


def read_record(models_dir: Path, version: str) -> dict[str, Any]:
    """The registry record of ``version``. Raises :class:`RegistryError` when it is missing or not a format-1
    JSON object with a matching version and a sha256."""
    path = record_path(models_dir, version)
    if not path.is_file():
        raise RegistryError(f"{version} is not registered in {Path(models_dir)} (no {path.name})")
    try:
        data = read_json(path)
    except PipelineError as exc:
        raise RegistryError(str(exc)) from exc
    if not isinstance(data, dict) or data.get("format") != RECORD_FORMAT or data.get("version") != version:
        raise RegistryError(f"{path} is not a format {RECORD_FORMAT} record of {version}")
    if not isinstance(data.get("sha256"), str) or not _SHA256.match(data["sha256"]):
        raise RegistryError(f"{path} has no valid sha256")
    return data


def find_registration(models_dir: Path, run_id: str) -> dict[str, Any] | None:
    """The record whose ``source.run_id`` is ``run_id`` (the highest version when several), else None.

    Unreadable records are skipped with a warning.
    """
    folder = Path(models_dir) / RECORD_DIR
    if not folder.is_dir():
        return None
    found: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for path in folder.glob("*.json"):
        if not VERSION_RE.match(path.stem):
            continue
        try:
            record = read_record(models_dir, path.stem)
        except RegistryError as exc:
            logger.warning("skipping registry record %s: %s", path, exc)
            continue
        source = record.get("source")
        if isinstance(source, Mapping) and source.get("run_id") == run_id:
            found.append((parse_version(path.stem), record))
    if not found:
        return None
    return max(found, key=lambda item: item[0])[1]


def _registration(models_dir: Path, record: Mapping[str, Any], *, created: bool) -> Registration:
    version = str(record["version"])
    return Registration(
        version=version,
        bundle=bundle_dir(models_dir, version),
        tarball=tarball_path(models_dir, version),
        sha256=str(record["sha256"]),
        size_bytes=int(record.get("size_bytes", 0)),
        record=record_path(models_dir, version),
        created=created,
    )


def register_run(
    run_id: str,
    *,
    models_dir: Path,
    uri: str | None = None,
    temporal_run_id: str | None = None,
    version: str | None = None,
    force: bool = False,
    dataset: Mapping[str, Any] | None = None,
    production_version: str | None = None,
    now: datetime | None = None,
) -> Registration:
    """Build, pack and record the production bundle of LOPO run ``run_id`` in ``models_dir``.

    A run that is already registered is returned as it is (``created=False``) unless ``force`` is set,
    which rebuilds the same version in place (refused for ``production_version``, the version in
    production) and keeps its promotion history. A new registration takes ``version`` or
    :func:`next_version`. The bundle is ``save_bundle`` output (package reasons and cost model copied byte for
    byte) plus ``model_card.md`` (:func:`model_card`; ``dataset`` is used only when its manifest sha256
    matches the run's), validated with ``load_bundle``, then packed with :func:`write_tarball` and recorded
    in ``registry/<version>.json``. A failed new registration removes the folder and tarball it created.
    Takes about a minute on the laptop, mostly the MLflow import.

    Raises :class:`RegistryError` for an invalid or taken version, a version clash with an earlier
    registration of the run, an unsuitable run (see :func:`fetch_run`) or unreadable artifacts.
    """
    models_dir = Path(models_dir)
    if version is not None:
        parse_version(version)
    existing = find_registration(models_dir, run_id)
    if existing is not None and not force:
        logger.info("run %s is already registered as %s", run_id, existing["version"])
        return _registration(models_dir, existing, created=False)
    if existing is not None:
        target_version = str(existing["version"])
        if version is not None and version != target_version:
            raise RegistryError(f"run {run_id} is already registered as {target_version}")
        if production_version == target_version:
            raise RegistryError(f"{target_version} is in production; register a new version instead")
    else:
        target_version = version or next_version(models_dir, now=now)
        clashes = [
            p
            for p in (
                bundle_dir(models_dir, target_version),
                tarball_path(models_dir, target_version),
                record_path(models_dir, target_version),
            )
            if p.exists()
        ]
        if clashes:
            raise RegistryError(f"{target_version} already exists: {clashes[0]}")

    with tempfile.TemporaryDirectory(prefix="tdp-register-") as tmp:
        files = fetch_run(run_id, Path(tmp), uri=uri, temporal_run_id=temporal_run_id)
        info = files.info
        try:
            booster = Ranker.from_json(files.ranker).booster
        except ValueError as exc:
            raise RegistryError(f"model/ranker.json of run {run_id}: {exc}") from exc
        try:
            calibrator = IsotonicCalibrator.from_json(
                json.loads(files.calibrator.read_text(encoding="utf-8"))
            )
        except ValueError as exc:
            raise RegistryError(f"model/calibrator.json of run {run_id}: {exc}") from exc
        try:
            tuned = load_params(files.params)
        except ConfigError as exc:
            raise RegistryError(str(exc)) from exc
        rounds = booster.num_boosted_rounds()
        if rounds != tuned.n_estimators:
            logger.warning("booster has %d rounds but the tuned params say %d", rounds, tuned.n_estimators)

        if dataset is not None and dataset.get("manifest_sha256") != info.tags.get("manifest_sha256"):
            logger.warning(
                "dataset summary belongs to another manifest; the model card omits the repositories"
            )
            dataset = None
        created_at = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
        metadata = bundle_metadata(info, tuned, created_at=created_at, registry_code_sha=code_git_sha())

        target = bundle_dir(models_dir, target_version)
        tarball = tarball_path(models_dir, target_version)
        old_promotions: list[Any] = []
        if existing is not None:
            old_promotions = list(existing.get("promotions") or [])
            _remove_tree(target)
            tarball.unlink(missing_ok=True)
        try:
            save_bundle(
                target, version=target_version, booster=booster, calibrator=calibrator, metadata=metadata
            )
            saved = json.loads((target / METADATA_FILE).read_text(encoding="utf-8"))
            card = model_card(target_version, saved, info, dataset=dataset)
            (target / MODEL_CARD_FILE).write_text(card, encoding="utf-8", newline="\n")
            load_bundle(target)
            sha = write_tarball(target, tarball)
            record = {
                "format": RECORD_FORMAT,
                "version": target_version,
                "registered_at": _stamp(),
                "bundle": target_version,
                "tarball": tarball.name,
                "sha256": sha,
                "size_bytes": tarball.stat().st_size,
                "files": {p.name: file_sha256(p) for p in sorted(target.iterdir(), key=lambda p: p.name)},
                "source": saved["source"],
                "model": info.model,
                "dataset_sha256": saved.get("dataset_sha256"),
                "manifest_sha256": saved.get("manifest_sha256"),
                "code_git_sha": saved.get("code_git_sha"),
                "registry_code_git_sha": saved.get("registry_code_git_sha"),
                "run_metrics": info.metrics,
                "temporal_metrics": info.temporal_metrics,
                "promotions": old_promotions,
            }
            write_json(record_path(models_dir, target_version), record)
        except BaseException:
            if existing is None:
                try:
                    _remove_tree(target)
                    tarball.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("could not clean up %s: %s", target, exc)
            raise
    logger.info("registered %s from run %s (sha256 %s)", target_version, run_id, sha)
    return _registration(models_dir, record, created=True)


# ---------------------------------------------------------------------------------------------------------
# golden repositories


def load_golden_config(path: Path = DEFAULT_GOLDEN_CONFIG) -> tuple[GoldenRepo, ...]:
    """Read the pinned golden repositories from a YAML file ``{repos: [{name, sha}, ...]}``.

    Raises :class:`~tdp_ml.pipeline.config.ConfigError` naming the file when it is missing or not YAML, the
    top level is not a mapping with a non-empty ``repos`` list, a key is unknown, a name is not
    ``owner/name``, a sha is not 40 lowercase hex digits, or two entries share a slug.
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the file ({exc})") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping with a repos list")
    unknown = sorted(str(k) for k in set(data) - _GOLDEN_TOP_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s): {', '.join(unknown)}")
    entries = data.get("repos")
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{path}: repos must be a non-empty list")
    repos: list[GoldenRepo] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: repos[{i}] must be a mapping with name and sha")
        extra = sorted(str(k) for k in set(entry) - _GOLDEN_REPO_KEYS)
        if extra:
            raise ConfigError(f"{path}: unknown key(s) in repos[{i}]: {', '.join(extra)}")
        name, sha = entry.get("name"), entry.get("sha")
        try:
            slug = slug_for(name) if isinstance(name, str) else slug_for("")
        except ConfigError as exc:
            raise ConfigError(f"{path}: repos[{i}]: {exc}") from exc
        if not isinstance(sha, str) or not _SHA40.match(sha):
            raise ConfigError(f"{path}: repos[{i}] sha must be 40 lowercase hex digits, got {sha!r}")
        if slug in seen:
            raise ConfigError(f"{path}: repos[{i}] repeats {name}")
        seen.add(slug)
        repos.append(GoldenRepo(name=str(name), sha=sha))
    return tuple(repos)


def golden_path(results_dir: Path, version: str) -> Path:
    """``results_dir/golden_<version>.json``: where a golden report of ``version`` is written."""
    return Path(results_dir) / f"golden_{version}.json"


def check_result(
    result: AnalysisResult, repo: GoldenRepo, bundle: Bundle, explainer: Explainer
) -> tuple[list[str], float | None]:
    """Check one golden analysis; returns ``(errors, largest SHAP additivity error)``.

    Checks that the analysed sha is the pin and the bundle version was used, that files were ranked, that
    every priority is an int in 0..100 with ``band == band_for(priority)``, that
    ``0 <= risk_low <= risk <= risk_high <= 1`` and scores are finite, that the top file has a reason, that
    the feature rows are exactly the ranked files, and that raw SHAP values plus the expected value give each
    score within :data:`ADDITIVITY_TOL`. At most :data:`MAX_GOLDEN_ERRORS` messages are kept.
    """
    import numpy as np

    errors: list[str] = []
    if result.sha != repo.sha:
        errors.append(f"analysed {result.sha}, expected {repo.sha}")
    if result.model_version != bundle.version:
        errors.append(f"model version {result.model_version}, expected {bundle.version}")
    files = result.files
    if not files:
        errors.append("no files were ranked")
        return errors[:MAX_GOLDEN_ERRORS], None
    for f in files:
        if not isinstance(f.priority, int) or not 0 <= f.priority <= 100:
            errors.append(f"{f.path}: priority {f.priority!r} outside 0..100")
        elif f.band != band_for(f.priority):
            errors.append(f"{f.path}: band {f.band} does not match priority {f.priority}")
        if not (math.isfinite(f.score) and 0.0 <= f.risk_low <= f.risk <= f.risk_high <= 1.0):
            errors.append(
                f"{f.path}: score {f.score!r} or risk {f.risk_low!r} <= {f.risk!r} <= {f.risk_high!r} invalid"
            )
    if not files[0].reasons:
        errors.append(f"top file {files[0].path} has no reason")
    paths = [f.path for f in files]
    features = result.features
    if len(set(paths)) != len(paths) or set(features.index) != set(paths) or len(features) != len(paths):
        errors.append(f"{len(features)} feature rows do not match the {len(paths)} ranked files")
        return errors[:MAX_GOLDEN_ERRORS], None
    X = features.loc[paths, list(FEATURE_INPUT_COLUMNS)]
    phi = explainer.shap_values(X)
    totals = phi.sum(axis=1) + explainer.expected_value
    scores = np.array([f.score for f in files], dtype=np.float64)
    worst = float(np.max(np.abs(totals - scores)))
    if not worst <= ADDITIVITY_TOL:
        errors.append(f"SHAP values miss the scores by up to {worst:.2e} (tolerance {ADDITIVITY_TOL:.0e})")
    return errors[:MAX_GOLDEN_ERRORS], worst


def _top_files(result: AnalysisResult, top: int) -> list[dict[str, Any]]:
    return [
        {
            "path": f.path,
            "priority": f.priority,
            "band": f.band,
            "risk": round(f.risk, 6),
            "risk_low": round(f.risk_low, 6),
            "risk_high": round(f.risk_high, 6),
            "debt_types": list(f.debt_types),
            "reasons": [r.text for r in f.reasons],
        }
        for f in result.files[:top]
    ]


def run_golden(
    bundle_path: Path,
    repos: Sequence[GoldenRepo],
    *,
    mirrors: Path,
    jobs: int = 4,
    top: int = 5,
    sha256: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Analyse every golden repository with the bundle in ``bundle_path`` and return the report.

    Each repository is analysed from its mirror ``mirrors/<slug>.git`` at its pinned sha with default
    settings, no static cache and a temporary SZZ cache, then checked with :func:`check_result`. An
    analysis error is recorded as ``"<Type>: <message>"`` for that repository and never raised. The report
    is ``{format, created_at, version, bundle, bundle_sha256, passed, seconds, repos}``; ``passed`` needs at
    least one repository and every repository passing. Each repository entry holds ``name, slug, sha,
    passed, errors, files, priority_min, priority_max, bands, additivity_max_error, top, seconds`` (``top``
    lists the first ``top`` files with their risk range, debt types and reason texts). About 15 s for the
    three default repositories on the laptop.

    Raises ``FileNotFoundError`` or :class:`~tdp_core.model.bundle.BundleSchemaError` when the bundle does
    not load, and ``ValueError`` when its reason templates are invalid.
    """
    from tdp_core.analyze import analyze
    from tdp_core.model.explain import Explainer

    started = time.perf_counter()
    bundle = load_bundle(Path(bundle_path))
    explainer = Explainer(bundle)
    entries: list[dict[str, Any]] = []
    for repo in repos:
        began = time.perf_counter()
        entry: dict[str, Any] = {
            "name": repo.name,
            "slug": repo.slug,
            "sha": repo.sha,
            "passed": False,
            "errors": [],
            "files": 0,
            "priority_min": None,
            "priority_max": None,
            "bands": {"Critical": 0, "High": 0, "Medium": 0, "Low": 0},
            "additivity_max_error": None,
            "top": [],
            "seconds": 0.0,
        }
        mirror = Path(mirrors) / f"{repo.slug}.git"
        errors: list[str] = []
        if not mirror.is_dir():
            errors.append(f"mirror not found: {mirror.as_posix()}")
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="tdp-golden-szz-") as szz:
                    result = analyze(mirror, repo.sha, bundle, jobs=jobs, szz_cache_dir=Path(szz))
                errors, worst = check_result(result, repo, bundle, explainer)
                priorities = [f.priority for f in result.files]
                entry["files"] = len(result.files)
                entry["priority_min"] = min(priorities) if priorities else None
                entry["priority_max"] = max(priorities) if priorities else None
                for f in result.files:
                    entry["bands"][f.band] = entry["bands"].get(f.band, 0) + 1
                entry["additivity_max_error"] = worst
                entry["top"] = _top_files(result, top)
            except Exception as exc:  # recorded per repository, never raised
                logger.exception("golden analysis of %s failed", repo.name)
                errors.append(f"{type(exc).__name__}: {exc}")
        entry["errors"] = errors[:MAX_GOLDEN_ERRORS]
        entry["passed"] = not errors
        entry["seconds"] = round(time.perf_counter() - began, 1)
        entries.append(entry)
        logger.info("golden %s at %s: %s", repo.name, repo.sha[:7], "ok" if not errors else errors[0])
    return {
        "format": REPORT_FORMAT,
        "created_at": _stamp(now),
        "version": bundle.version,
        "bundle": Path(bundle_path).absolute().as_posix(),
        "bundle_sha256": sha256,
        "passed": bool(entries) and all(e["passed"] for e in entries),
        "seconds": round(time.perf_counter() - started, 1),
        "repos": entries,
    }


# ---------------------------------------------------------------------------------------------------------
# production pointers


def read_pointer(path: Path) -> dict[str, Any] | None:
    """The production pointer in ``path``, or None when the file is missing or its version is null.

    Raises :class:`RegistryError` when the file is not a JSON object or ``version``, ``uri`` or ``sha256``
    are not strings (or null).
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except PipelineError as exc:
        raise RegistryError(str(exc)) from exc
    if not isinstance(data, dict):
        raise RegistryError(f"{path} must hold a JSON object")
    for key in POINTER_KEYS:
        if data.get(key) is not None and not isinstance(data.get(key), str):
            raise RegistryError(f"{path}: {key} must be a string or null")
    if data.get("version") is None:
        return None
    return {key: data.get(key) for key in POINTER_KEYS}


def default_production_files() -> list[Path]:
    """The pointer files :func:`promote` writes by default, the tracked one first.

    These are the first ``models/production.json`` found walking up from this module (the checkout's
    tracked pointer, read by the ``entropiq`` CLI) and ``tdp_core.paths.models_dir()/production.json`` (read
    by the worker and the platform). Duplicates are dropped; a warning is logged when no tracked file exists
    (for example in an installed package).
    """
    from tdp_core.paths import models_dir

    found: list[Path] = []
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "models" / PRODUCTION_FILE
        if candidate.is_file():
            found.append(candidate)
            break
    if not found:
        logger.warning("no tracked models/%s found above %s", PRODUCTION_FILE, Path(__file__).parent)
    found.append(models_dir() / PRODUCTION_FILE)
    unique: list[Path] = []
    keys: set[str] = set()
    for path in found:
        key = os.path.normcase(str(path.resolve()))
        if key not in keys:
            keys.add(key)
            unique.append(path)
    return unique


# ---------------------------------------------------------------------------------------------------------
# gates


def _gate(name: str, passed: bool, value: float | None, threshold: str, detail: str) -> GateResult:
    return GateResult(
        name=name,
        passed=passed,
        value=value,
        threshold=threshold,
        detail=detail,
        waivable=name in WAIVABLE_GATES,
    )


def _metric(metadata: Mapping[str, Any], group: str, key: str) -> float | None:
    metrics = metadata.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    values = metrics.get(group)
    if not isinstance(values, Mapping):
        return None
    return _finite(values.get(key))


def gate_real_model(metadata: Mapping[str, Any]) -> GateResult:
    """Pass for a bundle trained on real data: not ``synthetic`` and a positive int ``training_rows``."""
    rows = metadata.get("training_rows")
    threshold = "not synthetic, training_rows > 0"
    if metadata.get("synthetic") is True:
        return _gate("real_model", False, None, threshold, "synthetic dev bundle; never promote it")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        return _gate("real_model", False, None, threshold, f"training_rows is {rows!r}")
    return _gate("real_model", True, float(rows), threshold, f"trained on {rows} rows")


def gate_integrity(models_dir: Path, version: str, record: Mapping[str, Any]) -> GateResult:
    """Pass when the bundle loads with its version, the tarball's sha256 is the recorded one and the
    tarball holds exactly the folder's files with the same bytes. Value: the number of files."""
    threshold = "bundle loads, tarball sha256 and files match"
    try:
        folder = bundle_dir(models_dir, version)
        bundle = load_bundle(folder)
        if bundle.version != version:
            raise RegistryError(f"metadata.json says {bundle.version}")
        tarball = tarball_path(models_dir, version)
        if not tarball.is_file():
            raise RegistryError(f"tarball not found: {tarball}")
        sha = file_sha256(tarball)
        if sha != record.get("sha256"):
            raise RegistryError(
                f"tarball sha256 {sha[:12]} is not the recorded {str(record.get('sha256'))[:12]}"
            )
        packed = read_tarball(tarball, root=version)
        local = {p.name: p for p in folder.iterdir()}
        if set(packed) != set(local):
            raise RegistryError(f"tarball files {sorted(packed)} differ from folder files {sorted(local)}")
        changed = sorted(n for n, data in packed.items() if local[n].read_bytes() != data)
        if changed:
            raise RegistryError(f"folder files differ from the tarball: {', '.join(changed)}")
    except Exception as exc:
        return _gate("integrity", False, None, threshold, str(exc))
    return _gate("integrity", True, float(len(packed)), threshold, f"{len(packed)} files, sha256 {sha[:12]}")


def _incumbent_metadata(previous: Mapping[str, Any], models_dir: Path) -> dict[str, Any] | None:
    version = str(previous.get("version"))
    folders: list[Path] = []
    uri = previous.get("uri")
    if isinstance(uri, str) and uri:
        local = _local_dir(uri)
        if local is not None:
            folders.append(local)
    if VERSION_PATTERN.match(version):
        folders.append(Path(models_dir) / version)
    for folder in folders:
        meta = folder / METADATA_FILE
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("version") == version:
            return data
    return None


def gate_incumbent(
    metadata: Mapping[str, Any], version: str, previous: Mapping[str, Any] | None, models_dir: Path
) -> GateResult:
    """Pass when the candidate's LOPO NDCG@10 is not below the production model's lower 95% bound.

    Passes without comparing when nothing is in production yet, when ``version`` is already in production
    (a re-promotion) or when the incumbent is synthetic or has no LOPO NDCG@10. Fails when the incumbent
    cannot be found on this machine (its ``uri`` folder or ``models_dir/<version>``) or the candidate has no
    LOPO NDCG@10. The incumbent's point value is used when it has no interval.
    """
    name = "incumbent"
    threshold = "LOPO NDCG@10 >= production model's CI low"
    candidate = _metric(metadata, "lopo", "ndcg_10")
    if previous is None:
        return _gate(name, True, candidate, threshold, "no production model yet")
    current = str(previous.get("version"))
    if current == version:
        return _gate(name, True, candidate, threshold, f"{version} is already in production")
    incumbent = _incumbent_metadata(previous, models_dir)
    if incumbent is None:
        return _gate(
            name, False, candidate, threshold, f"production model {current} not found on this machine"
        )
    if incumbent.get("synthetic") is True:
        return _gate(name, True, candidate, threshold, f"production model {current} is synthetic")
    if candidate is None:
        return _gate(name, False, None, threshold, "candidate has no LOPO NDCG@10")
    reference = _metric(incumbent, "lopo", "ndcg_10_ci_low")
    point = _metric(incumbent, "lopo", "ndcg_10")
    if reference is None and point is None:
        return _gate(name, True, candidate, threshold, f"production model {current} has no LOPO NDCG@10")
    bar = reference if reference is not None else point
    assert bar is not None
    kind = "CI low" if reference is not None else "point value"
    passed = candidate >= bar
    detail = f"{candidate:.4f} vs {current} {kind} {bar:.4f}"
    return _gate(name, passed, candidate, f"LOPO NDCG@10 >= {bar:.4f}", detail)


def gate_hotspot(metadata: Mapping[str, Any], *, alpha: float = HOTSPOT_ALPHA) -> GateResult:
    """Pass when the one-sided paired Wilcoxon p of LOPO NDCG@10 against the hotspot heuristic is below
    ``alpha`` (``metrics.lopo.ndcg_10_p_greater_hotspot``). A missing p fails."""
    p = _metric(metadata, "lopo", "ndcg_10_p_greater_hotspot")
    threshold = f"p < {alpha:g}"
    if p is None:
        return _gate("hotspot", False, None, threshold, "no LOPO p-value against the hotspot heuristic")
    ndcg, hot = _metric(metadata, "lopo", "ndcg_10"), _metric(metadata, "lopo", "hotspot_ndcg_10")
    detail = f"LOPO NDCG@10 {_num(ndcg, 4)} vs hotspot {_num(hot, 4)}, one-sided Wilcoxon p {p:.4f}"
    return _gate("hotspot", p < alpha, p, threshold, detail)


def gate_calibration(metadata: Mapping[str, Any], *, max_ece: float = MAX_ECE) -> GateResult:
    """Pass when the pooled ECE on the temporal test set (``metrics.calibration.ece``) is at most
    ``max_ece``. A missing value fails with a hint to run ``tdp-ml calibrate``."""
    ece = _metric(metadata, "calibration", "ece")
    threshold = f"ECE <= {max_ece:g}"
    if ece is None:
        return _gate("calibration", False, None, threshold, "no calibration ECE; run tdp-ml calibrate")
    return _gate("calibration", ece <= max_ece, ece, threshold, f"pooled ECE {ece:.4f}")


def gate_leakage(
    layout: DataLayout,
    metadata: Mapping[str, Any],
    *,
    report: Path | None = None,
    min_queries: int = LEAKAGE_MIN_QUERIES,
) -> GateResult:
    """Pass when a leakage check of the exact training dataset passed on enough queries.

    Reads ``report`` (default ``runs/check-leakage.json`` of ``layout``, written by ``tdp-ml check-leakage``
    or ``uv run pytest -m leakage``) and requires: ``passed`` true, ``checked`` at least
    ``min(min_queries, dataset queries)`` and equal to the number of listed queries, every listed query
    passed with the same row count as ``dataset.parquet`` has for its qid, and the dataset's sha256 equal to
    the bundle's ``dataset_sha256`` (the report itself holds no hash). Value: ``checked``.
    """
    name = "leakage"
    threshold = f">= {min_queries} queries rebuilt without mismatches"
    path = Path(report) if report is not None else layout.run_record("check-leakage")
    hint = "run `uv run pytest -m leakage` or `tdp-ml promote --run-leakage`"
    if not path.is_file():
        return _gate(name, False, None, threshold, f"no leakage report at {path.as_posix()}; {hint}")
    try:
        data = read_json(path)
        if not isinstance(data, dict):
            raise RegistryError("the report is not a JSON object")
        checked = data.get("checked")
        queries = data.get("queries")
        if isinstance(checked, bool) or not isinstance(checked, int) or not isinstance(queries, list):
            raise RegistryError("the report has no checked count or query list")
        if not layout.dataset.is_file():
            raise RegistryError(f"dataset not found: {layout.dataset}")
        sha = file_sha256(layout.dataset)
        if sha != metadata.get("dataset_sha256"):
            raise RegistryError(f"dataset sha256 {sha[:12]} is not the model's training dataset")
        counts = pd.read_parquet(layout.dataset, columns=["qid"])["qid"].value_counts()
        needed = min(min_queries, len(counts))
        if data.get("passed") is not True:
            failing = [q.get("qid") for q in queries if isinstance(q, dict) and not q.get("passed")]
            raise RegistryError(f"the report did not pass (failing qids {failing[:5]})")
        if checked < needed or len(queries) != checked:
            raise RegistryError(f"{checked} queries checked ({len(queries)} listed), need {needed}")
        for q in queries:
            if not isinstance(q, dict) or q.get("passed") is not True:
                raise RegistryError(f"query {q!r} did not pass")
            qid = q.get("qid")
            rows = int(counts.get(qid, -1)) if isinstance(qid, int) else -1
            if rows != q.get("rows"):
                raise RegistryError(f"qid {qid} has {q.get('rows')} rows in the report, not the dataset's")
    except (RegistryError, PipelineError, OSError, ValueError, KeyError) as exc:
        return _gate(name, False, None, threshold, f"{path.name}: {exc}")
    return _gate(
        name, True, float(checked), threshold, f"{checked} queries rebuilt identically ({path.name})"
    )


def gate_golden(report: Mapping[str, Any], version: str) -> GateResult:
    """Pass when the golden report passed for ``version``. Value: the number of passing repositories."""
    repos = [r for r in report.get("repos", []) if isinstance(r, Mapping)]
    good = sum(1 for r in repos if r.get("passed") is True)
    threshold = "every pinned repository scores without errors"
    if report.get("version") != version:
        return _gate("golden", False, float(good), threshold, f"report is for {report.get('version')}")
    failing = [
        f"{r.get('name')}: {(r.get('errors') or ['failed'])[0]}" for r in repos if r.get("passed") is not True
    ]
    if report.get("passed") is True and not failing and repos:
        return _gate("golden", True, float(good), threshold, f"{good} of {len(repos)} repositories passed")
    detail = "; ".join(failing[:3]) if failing else "no repositories"
    return _gate("golden", False, float(good), threshold, detail)


# ---------------------------------------------------------------------------------------------------------
# promotion


def promotion_report_path(results_dir: Path, version: str, *, promoted: bool, dry_run: bool) -> Path:
    """Where :func:`promote` writes its report in ``results_dir``.

    ``promotion_<version>.json`` for a promotion that wrote the pointers, ``promotion_<version>_dry_run.json``
    for a dry run and ``promotion_<version>_refused.json`` for a refused promotion, so a later check of a
    version never overwrites the report of its actual promotion.
    """
    suffix = "" if promoted else ("_dry_run" if dry_run else "_refused")
    return Path(results_dir) / f"promotion_{version}{suffix}.json"


def _check_waivers(waive: Collection[str], reason: str | None) -> tuple[tuple[str, ...], str | None]:
    for gate in sorted(set(waive)):
        if gate not in GATES:
            raise RegistryError(f"unknown gate {gate!r}; gates are {', '.join(GATES)}")
        if gate not in WAIVABLE_GATES:
            raise RegistryError(f"gate {gate} cannot be waived")
    waivers = tuple(g for g in GATES if g in set(waive))
    text = reason.strip() if reason is not None else None
    if waivers:
        if not text:
            raise RegistryError("a waiver needs --reason")
        if "\n" in text or "\r" in text or len(text) > _MAX_REASON_CHARS:
            raise RegistryError(f"the reason must be one line of at most {_MAX_REASON_CHARS} characters")
    elif text:
        raise RegistryError("--reason is only used with --waive")
    return waivers, (text or None)


def _evaluate(name: str, check: Callable[[], GateResult]) -> GateResult:
    try:
        return check()
    except Exception as exc:
        logger.exception("gate %s raised", name)
        return _gate(name, False, None, "", f"{type(exc).__name__}: {exc}")


def promote(
    version: str,
    *,
    models_dir: Path,
    results_dir: Path,
    layout: DataLayout,
    production_files: Sequence[Path],
    golden: Sequence[GoldenRepo],
    mirrors: Path,
    waive: Collection[str] = (),
    reason: str | None = None,
    leakage_report: Path | None = None,
    run_leakage: bool = False,
    golden_report: Mapping[str, Any] | None = None,
    jobs: int = 4,
    threads: int = 2,
    dry_run: bool = False,
    now: datetime | None = None,
) -> PromotionOutcome:
    """Run every promotion gate on registered ``version`` and publish it when all pass or are waived.

    The gates (:data:`GATES`, in order) are: a real model, bundle and tarball integrity, not worse than the
    incumbent, significantly better than the hotspot heuristic, calibrated, a passing leakage check of the
    training dataset and the golden repositories. ``waive`` lists failed gates to accept anyway (only
    :data:`WAIVABLE_GATES`, and only with a one-line ``reason``); a waiver for a gate that passed is logged
    and has no effect. ``run_leakage`` first runs the leakage check on :data:`LEAKAGE_MIN_QUERIES` sampled
    queries (minutes). The golden repositories are scored every time unless ``golden_report`` is given; the
    report goes to ``golden_<version>.json`` in ``results_dir``.

    When accepted and not ``dry_run``, the pointer ``{"version", "uri", "sha256"}`` (``uri`` is the bundle's
    absolute POSIX path, ``sha256`` the tarball's) is written to every file of ``production_files`` and the
    promotion is appended to the registry record. A report is always written to ``results_dir`` (see
    :func:`promotion_report_path`: dry runs and refusals get their own file). A refused promotion is
    returned, not raised.

    Raises :class:`RegistryError` for an unknown or unwaivable gate in ``waive``, a missing or malformed
    reason, a reason without waivers, no production files, or a version that is not registered.
    """
    waivers, reason_text = _check_waivers(waive, reason)
    if not production_files:
        raise RegistryError("no production pointer file to write")
    models_dir, results_dir = Path(models_dir), Path(results_dir)
    record = read_record(models_dir, version)
    folder = bundle_dir(models_dir, version)
    try:
        metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RegistryError(f"cannot read {folder / METADATA_FILE}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise RegistryError(f"{folder / METADATA_FILE} is not a JSON object")
    previous = read_pointer(Path(production_files[0]))

    if run_leakage:
        from tdp_ml.pipeline.leakage import check_leakage

        logger.info("running the leakage check on %d queries", LEAKAGE_MIN_QUERIES)
        check_leakage(layout, sample=LEAKAGE_MIN_QUERIES, jobs=jobs, threads=threads)

    golden_file: Path | None = None
    if golden_report is None:
        report = run_golden(folder, golden, mirrors=mirrors, jobs=jobs, sha256=str(record["sha256"]), now=now)
        golden_file = write_json(golden_path(results_dir, version), report)
    else:
        report = dict(golden_report)

    checks: dict[str, Callable[[], GateResult]] = {
        "real_model": lambda: gate_real_model(metadata),
        "integrity": lambda: gate_integrity(models_dir, version, record),
        "incumbent": lambda: gate_incumbent(metadata, version, previous, models_dir),
        "hotspot": lambda: gate_hotspot(metadata),
        "calibration": lambda: gate_calibration(metadata),
        "leakage": lambda: gate_leakage(layout, metadata, report=leakage_report),
        "golden": lambda: gate_golden(report, version),
    }
    gates: list[GateResult] = []
    for name in GATES:
        result = _evaluate(name, checks[name])
        if name in waivers:
            if result.passed:
                logger.info("gate %s passed; its waiver is not needed", name)
            else:
                result = GateResult(
                    name=result.name,
                    passed=False,
                    value=result.value,
                    threshold=result.threshold,
                    detail=result.detail,
                    waivable=result.waivable,
                    waived=True,
                )
        gates.append(result)

    stamp = _stamp(now)
    pointer = {
        "version": version,
        "uri": Path(os.path.abspath(folder)).as_posix(),
        "sha256": str(record["sha256"]),
    }
    accepted = all(g.passed or g.waived for g in gates)
    promoted = accepted and not dry_run
    report_path = promotion_report_path(results_dir, version, promoted=promoted, dry_run=dry_run)
    written: list[Path] = []
    if promoted:
        keys: set[str] = set()
        for path in production_files:
            key = os.path.normcase(os.path.abspath(path))
            if key in keys:
                continue
            keys.add(key)
            written.append(write_json(Path(path), pointer))
        promotions = list(record.get("promotions") or [])
        promotions.append(
            {
                "promoted_at": stamp,
                "previous": previous,
                "pointer": pointer,
                "gates": [g.to_json() for g in gates],
                "waivers": list(waivers),
                "reason": reason_text,
                "report": report_path.as_posix(),
            }
        )
        updated = dict(record)
        updated["promotions"] = promotions
        write_json(record_path(models_dir, version), updated)
    outcome = PromotionOutcome(
        version=version,
        sha256=str(record["sha256"]),
        gates=tuple(gates),
        promoted=promoted,
        dry_run=dry_run,
        pointer=pointer,
        previous=previous,
        waivers=waivers,
        reason=reason_text,
        written=tuple(written),
        report=report_path,
        golden_report=golden_file,
        created_at=stamp,
    )
    write_json(report_path, outcome.to_json())
    if promoted:
        logger.info("promoted %s", version)
    else:
        logger.warning("%s not promoted; failed gates: %s", version, [g.name for g in outcome.failed])
    return outcome

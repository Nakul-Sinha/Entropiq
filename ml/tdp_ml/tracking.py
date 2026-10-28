"""MLflow helpers shared by the training, calibration, experiment and registry commands.

MLflow is imported lazily (``import mlflow`` takes seconds), always after :func:`configure_mlflow` has set
its environment: file stores need ``MLFLOW_ALLOW_FILE_STORE=true`` with mlflow 3, telemetry is switched
off and the agent hint is silenced. The tracking URI comes from ``MLFLOW_TRACKING_URI`` (the plan's default
is ``file:./mlruns``).

File stores record absolute artifact locations in their ``meta.yaml`` files, so a store copied from another
machine (for example the GPU box) cannot serve its artifacts until :func:`relocate_file_store` rewrites
them for the new folder (``tdp-ml mlflow-relocate``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse
from urllib.request import url2pathname

import yaml

logger = logging.getLogger(__name__)

MAX_PARAM_CHARS: Final = 6000
"""MLflow's limit on the length of a parameter value."""

_WINDOWS_PATH: Final = re.compile(r"^[A-Za-z]:[\\/]")


def tracking_uri() -> str:
    """``MLFLOW_TRACKING_URI`` (stripped) when set, else the ``file:`` URI of ``./mlruns`` (absolute)."""
    value = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    return value or (Path.cwd() / "mlruns").absolute().as_uri()


def file_store_root(uri: str) -> Path | None:
    """The folder of a file store URI, or None for any other kind of store.

    ``file:`` URIs give their path (never resolved, so a symbolic link in the path is kept); bare paths,
    including Windows drive paths such as ``C:/x``, give ``Path(uri)``; other schemes (``s3://``,
    ``http://``, ``databricks``, ...) give None.
    """
    if _WINDOWS_PATH.match(uri):
        return Path(uri)
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    if parsed.scheme == "":
        return Path(uri)
    return None


def configure_mlflow(uri: str | None = None) -> str:
    """Prepare the environment, import MLflow and set its tracking URI; returns the URI used.

    ``uri`` defaults to :func:`tracking_uri`. Sets (only when unset) ``MLFLOW_ALLOW_FILE_STORE=true`` for
    file stores, ``MLFLOW_DISABLE_TELEMETRY=true`` and ``MLFLOW_DISABLE_AGENT_HINT=1``, all before MLflow
    is imported.
    """
    used = uri or tracking_uri()
    if file_store_root(used) is not None:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(used)
    return used


def _is_local(value: str) -> bool:
    if _WINDOWS_PATH.match(value):
        return True
    return urlparse(value).scheme in ("", "file")


def _rewrite(meta: Path, key: str, target: str) -> bool:
    data = yaml.safe_load(meta.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return False
    value = data.get(key)
    if not isinstance(value, str) or not _is_local(value) or value == target:
        return False
    data[key] = target
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    tmp = meta.with_name(f".{meta.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, meta)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True


def relocate_file_store(root: Path) -> list[Path]:
    """Point the artifact locations of a moved MLflow file store at its current folder.

    For every experiment folder ``root/<exp>`` (names starting with ``.`` skipped) the ``artifact_location``
    of ``meta.yaml`` becomes the ``file:`` URI of that folder, and for every run ``root/<exp>/<run>`` the
    ``artifact_uri`` becomes the URI of ``<run>/artifacts``. Only local values (``file:`` URIs and bare
    paths) that differ are rewritten (atomically); ``s3://`` and other remote locations stay. Returns the
    changed files, sorted; a second call returns an empty list. Raises ``FileNotFoundError`` when ``root``
    is not a folder.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MLflow file store not found: {root}")
    changed: list[Path] = []
    for exp in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        meta = exp / "meta.yaml"
        if not meta.is_file():
            continue
        if _rewrite(meta, "artifact_location", exp.absolute().as_uri()):
            changed.append(meta)
        for run in sorted(p for p in exp.iterdir() if p.is_dir() and not p.name.startswith(".")):
            run_meta = run / "meta.yaml"
            if run_meta.is_file() and _rewrite(
                run_meta, "artifact_uri", (run / "artifacts").absolute().as_uri()
            ):
                changed.append(run_meta)
    logger.info("relocated %d meta files under %s", len(changed), root)
    return sorted(changed)


def _param_value(key: str, value: object) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, Mapping | list | tuple):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    else:
        text = str(value)
    if len(text) > MAX_PARAM_CHARS:
        raise ValueError(f"parameter {key!r} is {len(text)} characters long; MLflow allows {MAX_PARAM_CHARS}")
    return text


@dataclass(frozen=True, slots=True)
class RunLog:
    """The active run of :func:`mlflow_run`; its methods log through MLflow's fluent API and are valid
    only inside that ``with`` block."""

    run_id: str
    experiment_id: str
    experiment: str
    tracking_uri: str

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log parameters as strings: booleans as ``true``/``false``, mappings and sequences as compact
        sorted JSON. Raises ``ValueError`` for a value longer than 6000 characters."""
        import mlflow

        values = {str(k): _param_value(str(k), v) for k, v in params.items()}
        if values:
            mlflow.log_params(values)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log numeric metrics (at ``step`` when given); NaN and infinite values are skipped."""
        import mlflow

        values = {str(k): float(v) for k, v in metrics.items() if math.isfinite(float(v))}
        if values:
            mlflow.log_metrics(values, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Copy a local file into the run's artifacts (under ``artifact_path`` when given)."""
        import mlflow

        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set run tags (values converted to strings)."""
        import mlflow

        mlflow.set_tags({str(k): str(v) for k, v in tags.items()})


@contextmanager
def mlflow_run(
    experiment: str, run_name: str, *, uri: str | None = None, tags: Mapping[str, str] | None = None
) -> Iterator[RunLog]:
    """Start an MLflow run in ``experiment`` (created when missing) and yield its :class:`RunLog`.

    The tracking URI is ``uri`` or :func:`tracking_uri`. The run ends FINISHED, or FAILED when the block
    raises (the exception propagates).
    """
    used = configure_mlflow(uri)
    import mlflow

    exp: Any = mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags={str(k): str(v) for k, v in (tags or {}).items()}) as run:
        yield RunLog(
            run_id=str(run.info.run_id),
            experiment_id=str(exp.experiment_id),
            experiment=experiment,
            tracking_uri=used,
        )

"""Model bundles for the workers: the production pointer and a per-process cache of loaded bundles.

``MODEL_DIR/production.json`` (plan section 3.5) names the production model version as
``{"version": "...", "uri": "...", "sha256": "..."}``; the bundle itself lives in ``MODEL_DIR/<version>/``.
Analyses record the model version they were created with, so a worker loads that version with
:func:`get_bundle` rather than whatever production points at now. :func:`get_production_bundle` reads the
pointer again on every call, so promoting a model needs no worker restart.

Loaded bundles are cached per process, keyed by the resolved model folder and the version. ``Bundle`` is
unhashable (frozen dataclass with dict fields), so the cache is a plain dict behind a lock and never
``functools.lru_cache``. Importing this module does not load xgboost: ``tdp_core.model.bundle`` is imported
on the first load.

Errors are :class:`BundleUnavailableError`, whose message is safe to show to users: it names model versions
but never filesystem paths. The details go to the log.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from tdp_platform.settings import get_settings

if TYPE_CHECKING:  # importing the bundle module loads xgboost
    from tdp_core.model.bundle import Bundle

logger = logging.getLogger(__name__)

PRODUCTION_FILE: Final = "production.json"

# Same rule as tdp_core.model.bundle.VERSION_PATTERN, kept here so reading the pointer never imports
# xgboost. A test checks that both patterns agree.
VERSION_RE: Final = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")

_NO_PRODUCTION: Final = "No production model is configured."
_BAD_POINTER: Final = "The production model pointer is invalid."

_cache: dict[tuple[str, str], Bundle] = {}
_cache_lock = threading.Lock()


class BundleUnavailableError(RuntimeError):
    """A model bundle cannot be used. ``str()`` is a full sentence that is safe to show to users."""


@dataclass(frozen=True, slots=True)
class ProductionPointer:
    """The parsed ``production.json``: the version plus the optional release ``uri`` and ``sha256``.

    ``sha256`` is kept as given and not verified here (its format belongs to the model registry).
    """

    version: str
    uri: str | None = None
    sha256: str | None = None


def _version_ok(version: object) -> bool:
    return isinstance(version, str) and VERSION_RE.fullmatch(version) is not None


def _model_root(model_dir: Path | None) -> Path:
    return Path(model_dir) if model_dir is not None else get_settings().model_dir


def read_production_pointer(model_dir: Path) -> ProductionPointer:
    """Read ``model_dir/production.json``.

    Raises :class:`BundleUnavailableError` with "No production model is configured." when the file is
    missing or its ``version`` is null or absent, and "The production model pointer is invalid." for
    unreadable JSON, a value that is not an object, a version that is not a valid bundle version, or a
    ``uri`` or ``sha256`` that is neither a string nor null.
    """
    file = Path(model_dir) / PRODUCTION_FILE
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("no %s in the model folder %s", PRODUCTION_FILE, model_dir)
        raise BundleUnavailableError(_NO_PRODUCTION) from None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("cannot read %s: %s", file, exc)
        raise BundleUnavailableError(_BAD_POINTER) from None
    try:
        data: Any = json.loads(text)
    except ValueError as exc:
        logger.warning("%s is not valid JSON: %s", file, exc)
        raise BundleUnavailableError(_BAD_POINTER) from None
    if not isinstance(data, dict):
        logger.warning("%s must hold a JSON object", file)
        raise BundleUnavailableError(_BAD_POINTER)
    version = data.get("version")
    if version is None:
        raise BundleUnavailableError(_NO_PRODUCTION)
    if not _version_ok(version):
        logger.warning("%s names an invalid version %r", file, version)
        raise BundleUnavailableError(_BAD_POINTER)
    extras: dict[str, str | None] = {}
    for key in ("uri", "sha256"):
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            logger.warning("%s: %s must be a string or null", file, key)
            raise BundleUnavailableError(_BAD_POINTER)
        extras[key] = value
    return ProductionPointer(version=version, uri=extras["uri"], sha256=extras["sha256"])


def production_version(model_dir: Path | None = None) -> str:
    """The production model version named by ``production.json`` (default folder: ``MODEL_DIR``)."""
    return read_production_pointer(_model_root(model_dir)).version


def bundle_path(model_dir: Path, version: str) -> Path:
    """The folder of bundle ``version`` under ``model_dir``.

    The version must match the bundle version pattern (letters, digits, ``.``, ``_`` and ``-``, at most 128
    characters), so it can never escape ``model_dir``; otherwise :class:`BundleUnavailableError` (the
    message does not repeat the input).
    """
    if not _version_ok(version):
        raise BundleUnavailableError("The model version is not valid.")
    return Path(model_dir) / version


def get_bundle(version: str, *, model_dir: Path | None = None) -> Bundle:
    """The loaded bundle of ``version`` from ``model_dir`` (default ``MODEL_DIR``), cached per process.

    Raises :class:`BundleUnavailableError`: "Model version X is not installed on this worker." when the
    folder is missing or its metadata names another version, and "Model version X could not be loaded."
    when the bundle is incomplete or inconsistent. A synthetic (development) bundle loads with a warning.
    """
    root = _model_root(model_dir)
    path = bundle_path(root, version)
    key = (str(root.resolve()), version)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        from tdp_core.model.bundle import BundleSchemaError, load_bundle

        try:
            bundle = load_bundle(path)
        except FileNotFoundError:
            logger.error("model bundle %s is missing from %s", version, root)
            raise BundleUnavailableError(
                f"Model version {version} is not installed on this worker."
            ) from None
        except BundleSchemaError:
            logger.exception("model bundle %s in %s is invalid", version, root)
            raise BundleUnavailableError(f"Model version {version} could not be loaded.") from None
        if bundle.version != version:
            logger.error("the folder of model %s holds a bundle of version %s", version, bundle.version)
            raise BundleUnavailableError(f"Model version {version} is not installed on this worker.")
        if bundle.metadata.get("synthetic"):
            logger.warning("model %s is a synthetic development bundle; rankings are not meaningful", version)
        _cache[key] = bundle
        return bundle


def get_production_bundle(*, model_dir: Path | None = None) -> Bundle:
    """The bundle that ``production.json`` currently names (the pointer is read again on each call)."""
    root = _model_root(model_dir)
    return get_bundle(production_version(root), model_dir=root)


def clear_cache() -> None:
    """Forget every loaded bundle of this process (tests, and forked worker processes)."""
    with _cache_lock:
        _cache.clear()

"""Find and load the model bundle the app scores with.

Resolution order (first match wins), done from JSON files only so startup never imports XGBoost:

1. ``--dev-model`` together with ``--model-dir`` is a usage error;
2. ``--dev-model``: the demo bundle in ``<home>/models/dev`` (built on first use);
3. ``--model-dir DIR``;
4. ``$TDP_MODEL_DIR``;
5. ``models/production.json`` found by walking up from this package: its ``version`` is loaded from its
   ``uri`` when that is a local folder (absolute, relative to the repository root, or ``file://``), else from
   a folder named after the version under ``$TDP_MODELS_DIR``, ``<home>/models`` or ``<repo>/models``. A
   candidate counts only when its ``metadata.json`` has that version;
6. otherwise the demo bundle, with the reason shown to the user.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal
from urllib.parse import unquote, urlparse

import tdp_cli
from tdp_cli.errors import CliError, UsageError
from tdp_cli.home import EntropiqHome

if TYPE_CHECKING:
    from tdp_core.model.bundle import Bundle

logger = logging.getLogger(__name__)

type ModelSource = Literal["option", "env", "production", "dev"]

ENV_MODEL_DIR: Final = "TDP_MODEL_DIR"
ENV_MODELS_DIR: Final = "TDP_MODELS_DIR"
PRODUCTION_FILE: Final = Path("models") / "production.json"
_METADATA: Final = "metadata.json"
_FEATURE_SCHEMA: Final = "feature_schema.json"
_BUNDLE_FILES: Final = frozenset(
    {
        "ranker.json",
        "calibrator.json",
        "feature_schema.json",
        "reasons.yaml",
        "cost_model.yaml",
        "metadata.json",
        "model_card.md",
    }
)


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """The bundle folder to score with, where it came from, and whether it is a demo (synthetic) model."""

    path: Path
    source: ModelSource
    version: str | None
    demo: bool
    reason: str

    @property
    def kind_label(self) -> str:
        """``"production model"``, ``"demo model"`` or ``"custom model"``."""
        if self.demo:
            return "demo model"
        if self.source == "production":
            return "production model"
        return "custom model"

    @property
    def notice(self) -> str | None:
        """The warning shown when a demo model is in use, else None."""
        if not self.demo:
            return None
        return (
            f"Demo model in use ({self.reason}): it was trained on synthetic data, so rankings show how "
            "Entropiq works, not real risk."
        )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata(folder: Path) -> dict[str, Any] | None:
    """The bundle's metadata.json as a dict, or None when missing or unreadable."""
    try:
        data = _read_json(folder / _METADATA)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_repo_root(start: Path | None = None) -> Path | None:
    """The first folder at or above ``start`` (default: this package) holding ``models/production.json``."""
    here = (start or Path(tdp_cli.__file__).resolve().parent).resolve()
    for folder in (here, *here.parents):
        if (folder / PRODUCTION_FILE).is_file():
            return folder
    return None


def _explicit(path: Path, source: ModelSource) -> ModelChoice:
    folder = path.expanduser().absolute().resolve()
    if not (folder / _METADATA).is_file():
        raise CliError(f"{folder} is not a model bundle (metadata.json is missing)", exit_code=2)
    meta = _metadata(folder)
    if meta is None or not isinstance(meta.get("version"), str) or not meta["version"]:
        raise CliError(f"{folder} is not a model bundle (metadata.json has no version)", exit_code=2)
    label = "--model-dir" if source == "option" else ENV_MODEL_DIR
    return ModelChoice(folder, source, meta["version"], meta.get("synthetic") is True, label)


def dev_choice(home: EntropiqHome, reason: str) -> ModelChoice:
    """The demo bundle in ``home`` (its version when it is built already)."""
    meta = _metadata(home.dev_model)
    version = meta.get("version") if meta and meta.get("synthetic") is True else None
    return ModelChoice(home.dev_model, "dev", version if isinstance(version, str) else None, True, reason)


def _uri_path(uri: str, repo_root: Path) -> Path | None:
    """A local folder named by ``uri`` (absolute path, path relative to the repo root, or file:// URI)."""
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        raw = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            raw = f"//{parsed.netloc}{raw}"
        if len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":  # file:///C:/models/x
            raw = raw[1:]
        candidate = Path(raw)
    elif len(parsed.scheme) > 1:  # http, https, s3 and other schemes are not local
        return None
    else:  # no scheme, or a one-letter Windows drive read as a scheme
        candidate = Path(uri)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
    return candidate if candidate.is_dir() else None


def _production(
    repo_root: Path, home: EntropiqHome, environ: Mapping[str, str]
) -> tuple[ModelChoice | None, str]:
    try:
        data = _read_json(repo_root / PRODUCTION_FILE)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except (OSError, ValueError):
        return None, "models/production.json is unreadable"
    version = data.get("version")
    if not isinstance(version, str) or not version:
        return None, "no production model has been published yet"
    candidates: list[Path] = []
    uri = data.get("uri")
    if isinstance(uri, str):
        local = _uri_path(uri, repo_root)
        if local is not None:
            candidates.append(local)
    models_dir = environ.get(ENV_MODELS_DIR, "").strip()
    if models_dir:
        candidates.append(Path(models_dir) / version)
    candidates.extend([home.models / version, repo_root / "models" / version])
    for folder in candidates:
        meta = _metadata(folder)
        if meta is not None and meta.get("version") == version:
            return (
                ModelChoice(
                    folder.absolute().resolve(),
                    "production",
                    version,
                    meta.get("synthetic") is True,
                    "models/production.json",
                ),
                "",
            )
    return None, f"production model {version} is not on this machine"


def resolve_model(
    model_dir: Path | None = None,
    *,
    dev: bool = False,
    home: EntropiqHome,
    environ: Mapping[str, str] | None = None,
    repo_root: Path | Literal["auto"] | None = "auto",
) -> ModelChoice:
    """Choose the model bundle (see the module docstring for the order). Reads JSON only.

    Raises :class:`UsageError` for ``dev`` with ``model_dir``, and :class:`CliError` (exit code 2) when an
    explicit folder (``--model-dir`` or ``$TDP_MODEL_DIR``) is not a bundle.
    """
    env = os.environ if environ is None else environ
    if dev and model_dir is not None:
        raise UsageError("use either --model-dir or --dev-model, not both")
    if dev:
        return dev_choice(home, "--dev-model")
    if model_dir is not None:
        return _explicit(Path(model_dir), "option")
    env_dir = env.get(ENV_MODEL_DIR, "").strip()
    if env_dir:
        return _explicit(Path(env_dir), "env")
    root = find_repo_root() if repo_root == "auto" else repo_root
    if root is None:
        return dev_choice(home, "no models/production.json found")
    choice, reason = _production(root, home, env)
    return choice if choice is not None else dev_choice(home, reason)


_BUNDLES: dict[str, Bundle] = {}


def _cache_key(path: Path) -> str:
    return os.path.normcase(str(path.absolute().resolve()))


def clear_model_cache() -> None:
    """Forget every loaded bundle (tests)."""
    _BUNDLES.clear()


def _build_dev(path: Path) -> None:
    from tdp_core.model.bundle import make_dev_bundle

    if path.exists():
        if not path.is_dir() or any(p.name not in _BUNDLE_FILES for p in path.iterdir()):
            raise CliError(
                f"{path} holds other files; remove it so the demo model can be built there", exit_code=1
            )
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.building-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        make_dev_bundle(tmp)
        os.replace(tmp, path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_model(choice: ModelChoice) -> Bundle:
    """Load (once per process) the bundle of ``choice``; the demo bundle is built when missing or broken.

    Raises :class:`CliError` (exit code 2) when a production or explicit bundle cannot be loaded.
    """
    from tdp_core.model.bundle import BundleSchemaError, load_bundle

    key = _cache_key(choice.path)
    cached = _BUNDLES.get(key)
    if cached is not None:
        return cached
    bundle: Bundle | None = None
    if choice.source == "dev":
        if (choice.path / _METADATA).is_file():
            try:
                bundle = load_bundle(choice.path)
            except (BundleSchemaError, ValueError, OSError) as exc:
                logger.warning("Rebuilding the demo model in %s: %s", choice.path, exc)
            if bundle is not None and bundle.metadata.get("synthetic") is not True:
                bundle = None
        if bundle is None:
            _build_dev(choice.path)
            bundle = load_bundle(choice.path)
    else:
        try:
            bundle = load_bundle(choice.path)
        except (FileNotFoundError, BundleSchemaError, ValueError) as exc:
            raise CliError(f"Cannot load the model in {choice.path}: {exc}", exit_code=2) from exc
    _BUNDLES[key] = bundle
    return bundle


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What ``/model`` shows about a bundle."""

    version: str
    path: Path
    source: ModelSource
    demo: bool
    kind_label: str
    reason: str
    created_at: str | None
    training_rows: int | None
    metrics: dict[str, float]
    n_features: int
    extractor_version: str | None
    code_git_sha: str | None


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], out)


def read_model_info(choice: ModelChoice) -> ModelInfo:
    """Read the bundle's metadata and feature schema. Raises :class:`CliError` when they are missing."""
    meta = _metadata(choice.path)
    try:
        schema = _read_json(choice.path / _FEATURE_SCHEMA)
    except (OSError, ValueError):
        schema = None
    if meta is None or not isinstance(schema, dict):
        raise CliError(f"{choice.path} is not a complete model bundle", exit_code=2)
    metrics: dict[str, float] = {}
    _flatten("", meta.get("metrics") or {}, metrics)
    rows = meta.get("training_rows")
    columns = schema.get("columns")
    created = meta.get("created_at")
    extractor = schema.get("extractor_version")
    code_sha = meta.get("code_git_sha")
    return ModelInfo(
        version=str(meta.get("version") or choice.version or "unknown"),
        path=choice.path,
        source=choice.source,
        demo=choice.demo or meta.get("synthetic") is True,
        kind_label=choice.kind_label,
        reason=choice.reason,
        created_at=created if isinstance(created, str) else None,
        training_rows=rows if isinstance(rows, int) and not isinstance(rows, bool) else None,
        metrics=metrics,
        n_features=len(columns) if isinstance(columns, list) else 0,
        extractor_version=extractor if isinstance(extractor, str) else None,
        code_git_sha=code_sha if isinstance(code_sha, str) else None,
    )

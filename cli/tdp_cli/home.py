"""The Entropiq home folder (``~/.entropiq`` or ``$ENTROPIQ_HOME``) and the user's settings.

Layout::

    <home>/history                 prompt history of the interactive shell
    <home>/settings.json           settings changed with /settings
    <home>/models/dev/             the demo model bundle, built on first use
    <home>/models/<version>/       production bundles copied here by hand (optional)
    <home>/models/production.json  production pointer read by an installed command (a copy)
    <home>/analyses/<slug>/<sha>.{json,meta.json,features.parquet}   saved analyses
    <home>/mirrors/<host>/<owner>/<name>.git                          lean mirrors of analysed URLs
    <home>/cache/szz/<slug>/       per-repository SZZ cache
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Final

from tdp_cli.errors import CliError, UsageError
from tdp_core.config import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)

ENV_HOME: Final = "ENTROPIQ_HOME"


@dataclass(frozen=True, slots=True)
class EntropiqHome:
    """Paths inside the Entropiq home folder ``root`` (absolute)."""

    root: Path

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> EntropiqHome:
        """``$ENTROPIQ_HOME`` when set and not empty, else ``~/.entropiq`` (expanded and resolved)."""
        env = os.environ if environ is None else environ
        raw = env.get(ENV_HOME, "").strip()
        root = Path(raw) if raw else Path("~/.entropiq")
        return cls(root.expanduser().absolute().resolve())

    @property
    def history_file(self) -> Path:
        """Prompt history of the shell."""
        return self.root / "history"

    @property
    def settings_file(self) -> Path:
        """Saved settings (JSON)."""
        return self.root / "settings.json"

    @property
    def models(self) -> Path:
        """Folder of locally kept model bundles."""
        return self.root / "models"

    @property
    def dev_model(self) -> Path:
        """The demo model bundle folder."""
        return self.root / "models" / "dev"

    @property
    def analyses(self) -> Path:
        """Saved analyses."""
        return self.root / "analyses"

    @property
    def mirrors(self) -> Path:
        """Lean bare mirrors of analysed URLs."""
        return self.root / "mirrors"

    def mirror_path(self, host: str, owner: str, name: str) -> Path:
        """The absolute mirror folder of a validated repository: ``mirrors/<host>/<owner>/<name>.git``."""
        return self.mirrors / host / owner / f"{name}.git"

    def szz_cache(self, slug: str) -> Path:
        """The SZZ cache folder of one repository."""
        return self.root / "cache" / "szz" / slug

    def ensure(self) -> None:
        """Create the home folder. Raises :class:`CliError` when it cannot be created."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CliError(
                f"Cannot create the Entropiq folder {self.root}: {exc.strerror or exc}",
                hint=f"Set {ENV_HOME} to a folder you can write to.",
            ) from exc


@dataclass(frozen=True, slots=True)
class Prefs:
    """User settings: plan budget in hours, rows shown by ``tdp analyze``, parallel jobs."""

    budget_hours: float = DEFAULT_SETTINGS.default_budget_hours
    top: int = 20
    jobs: int = 4


PREF_HELP: Final[dict[str, str]] = {
    "budget_hours": "Hours available for fixes when a plan is made (0 to 2000)",
    "top": "Rows shown by tdp analyze when --top is not given (1 to 500)",
    "jobs": "Parallel workers used by an analysis (1 to 64)",
}

_LIMITS: Final[dict[str, tuple[float, float]]] = {
    "budget_hours": (0.0, 2000.0),
    "top": (1, 500),
    "jobs": (1, 64),
}


def _coerce(key: str, value: object) -> float | int:
    """Validate one setting value; raises ``ValueError`` with a readable message."""
    low, high = _LIMITS[key]
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    if key == "budget_hours":
        if isinstance(value, str):
            value = float(value)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        number: float | int = float(value)
    else:
        if isinstance(value, str):
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"{key} must be a whole number")
        number = value
    if not low <= number <= high:
        raise ValueError(f"{key} must be between {low:g} and {high:g}")
    return number


def load_prefs(path: Path) -> Prefs:
    """Read settings from ``path``; a missing file gives defaults, bad values fall back per key (logged)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Prefs()
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable settings file %s: %s", path, exc)
        return Prefs()
    if not isinstance(raw, dict):
        logger.warning("Ignoring settings file %s: not a JSON object", path)
        return Prefs()
    values: dict[str, Any] = {}
    for f in fields(Prefs):
        if f.name in raw:
            try:
                values[f.name] = _coerce(f.name, raw[f.name])
            except (TypeError, ValueError) as exc:
                logger.warning("Ignoring setting %s in %s: %s", f.name, path, exc)
    return Prefs(**values)


def save_prefs(path: Path, prefs: Prefs) -> None:
    """Write ``prefs`` to ``path`` atomically (a temporary file, then a rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(asdict(prefs), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def set_pref(prefs: Prefs, key: str, raw: str) -> Prefs:
    """``prefs`` with setting ``key`` parsed from ``raw``.

    Raises :class:`UsageError` for an unknown key (the message lists the valid ones) or an invalid value
    (the message gives the range).
    """
    name = key.strip().lower().replace("-", "_")
    if name not in PREF_HELP:
        raise UsageError(f"unknown setting {key!r}; valid settings: {', '.join(PREF_HELP)}")
    try:
        value = _coerce(name, raw.strip())
    except ValueError as exc:
        message = str(exc)
        if not message.startswith(name):
            low, high = _LIMITS[name]
            message = f"{name} must be a number between {low:g} and {high:g}"
        raise UsageError(message) from exc
    changes: dict[str, Any] = {name: value}
    return replace(prefs, **changes)

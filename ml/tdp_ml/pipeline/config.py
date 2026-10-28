"""Load and validate the pipeline configs ``ml/configs/repos.yaml`` and ``ml/configs/labels.yaml``.

``repos.yaml`` lists the training repositories (``id``, ``name``, optional ``url``) and the inclusion
criteria; ``labels.yaml`` holds the overrides of :class:`tdp_core.config.Settings` used for snapshots and
labels, plus the number of snapshots kept per repository. Every problem raises :class:`ConfigError` naming
the file and the key, so a typo never silently falls back to a default.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeGuard

import yaml

from tdp_core.config import DEFAULT_SETTINGS, Settings, with_overrides
from tdp_core.security.urls import UnsafeRepoError, validate_repo_url

DEFAULT_CONFIG_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_REPOS_CONFIG: Final[Path] = DEFAULT_CONFIG_DIR / "repos.yaml"
DEFAULT_LABELS_CONFIG: Final[Path] = DEFAULT_CONFIG_DIR / "labels.yaml"

MAX_REPO_ID: Final = 9_999_999
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REPOS_KEYS: Final = frozenset({"criteria", "max_repos", "repos"})
_LABELS_KEYS: Final = frozenset({"settings", "snapshots", "szz"})
_REPO_KEYS: Final = frozenset({"id", "name", "url", "issues"})
_SNAPSHOT_KEYS: Final = frozenset({"max_per_repo"})
_SZZ_KEYS: Final = frozenset({"max_fixes"})
_SETTINGS_FIELDS: Final = {f.name: f for f in dataclasses.fields(Settings)}


class ConfigError(ValueError):
    """A pipeline config file is missing, malformed or holds an invalid value."""


def slug_for(name: str) -> str:
    """The data slug of ``"Owner/Name"``: ``"owner__name"`` (lower case, plan section 3.4).

    Raises :class:`ConfigError` unless ``name`` is ``owner/name`` with parts of letters, digits, ``.``,
    ``_`` and ``-`` that start with a letter or digit.
    """
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ConfigError(f"repository name must look like 'owner/name', got {name!r}")
    owner, repo = name.split("/")
    return f"{owner}__{repo}".lower()


@dataclass(frozen=True, slots=True)
class RepoSpec:
    """One training repository: a stable ``id`` (never reused), ``owner/name`` and its clone ``url``.

    ``issues`` False means its GitHub issues are not crawled (repositories with 5,000 or more issues): bug
    fixes then come from commit keywords only and the bug-issue criterion is waived.
    """

    id: int
    name: str
    url: str
    issues: bool = True

    @property
    def slug(self) -> str:
        return slug_for(self.name)

    @property
    def owner(self) -> str:
        return self.name.split("/")[0]

    @property
    def repo(self) -> str:
        return self.name.split("/")[1]


@dataclass(frozen=True, slots=True)
class Criteria:
    """Inclusion criteria of a training repository (all thresholds inclusive: equality passes).

    ``min_history_days`` and ``max_days_since_last_commit`` are days, ``min_source_files`` counts the
    non-test source files at the newest kept snapshot, ``min_bug_issues`` the GitHub issues with a bug label.
    """

    min_history_days: int = 1095
    min_commits: int = 1000
    min_source_files: int = 20
    min_bug_issues: int = 30
    max_days_since_last_commit: int = 365
    min_snapshots: int = 1

    def to_json(self) -> dict[str, int]:
        return {f.name: int(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Criteria:
        """Rebuild from :meth:`to_json` output; unknown keys or non-integer values raise ``ValueError``."""
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"unknown criteria keys: {', '.join(unknown)}")
        values: dict[str, int] = {}
        for key, value in data.items():
            if not _is_int(value) or value < 0:
                raise ValueError(f"criteria {key} must be an integer >= 0, got {value!r}")
            values[key] = value
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Everything the ``manifest`` command needs: repositories, criteria, settings and snapshots kept.

    ``max_repos`` caps the dataset at the first N passing repositories in file order (None: all of them);
    ``szz_max_fixes`` caps the bug fixes traced with SZZ per repository (None: no budget).
    """

    repos: tuple[RepoSpec, ...]
    criteria: Criteria
    settings: Settings
    max_snapshots: int
    max_repos: int | None = None
    szz_max_fixes: int | None = None


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _setting_value(key: str, value: object) -> object:
    """``value`` checked against the type of the default of Settings field ``key`` (``ConfigError``)."""
    default = getattr(DEFAULT_SETTINGS, key)
    if key == "history_window_days" and value is None:
        return None
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
    elif isinstance(default, int):
        if _is_int(value):
            return value
    elif isinstance(default, float):
        if _is_int(value) or isinstance(value, float):
            return float(value)
    elif isinstance(default, str):
        if isinstance(value, str):
            return value
    elif (
        isinstance(default, tuple)
        and isinstance(value, (list, tuple))
        and len(value) == len(default)
        and all(_is_int(v) for v in value)
    ):
        return tuple(value)
    raise ConfigError(f"setting {key} must have the type of its default {default!r}, got {value!r}")


def settings_from_overrides(overrides: Mapping[str, object], base: Settings = DEFAULT_SETTINGS) -> Settings:
    """``base`` with ``overrides`` applied; keys must be :class:`Settings` field names.

    A value must have the type of the field's default: an int for int fields (a bool is rejected), an int
    or float for float fields, a string for string fields, a list of two ints for ``recent_days`` (stored as
    a tuple) and an int or None for ``history_window_days``. Raises :class:`ConfigError` naming the key.
    """
    values: dict[str, object] = {}
    for key, value in overrides.items():
        if key not in _SETTINGS_FIELDS:
            raise ConfigError(f"unknown setting {key!r}")
        values[key] = _setting_value(key, value)
    return with_overrides(base, **values)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the file ({exc})") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML ({exc})") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    return data


def _check_keys(path: Path, where: str, data: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(str(k) for k in set(data) - allowed)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) in {where}: {', '.join(unknown)}")


def _repo_url(path: Path, name: str, raw: object) -> str:
    if raw is None:
        url = f"https://github.com/{name}.git"
    elif isinstance(raw, str) and raw.strip():
        url = raw.strip()
    else:
        raise ConfigError(f"{path}: repos[{name}].url must be a non-empty string")
    if url.lower().startswith("https://"):
        try:
            validate_repo_url(url, {"github.com"}, resolve_dns=False)
        except UnsafeRepoError as exc:
            raise ConfigError(f"{path}: repos[{name}].url is not allowed: {exc}") from exc
        return url
    if "://" in url or not Path(url).is_absolute():
        raise ConfigError(f"{path}: repos[{name}].url must be an https GitHub URL or an absolute local path")
    return url


def _optional_count(path: Path, where: str, value: object) -> int | None:
    """``value`` as an integer >= 1, or None when it is None (``ConfigError`` otherwise)."""
    if value is None:
        return None
    if not _is_int(value) or value < 1:
        raise ConfigError(f"{path}: {where} must be an integer >= 1 or null, got {value!r}")
    return value


def _load_repos(path: Path) -> tuple[tuple[RepoSpec, ...], Criteria, int | None]:
    data = _load_yaml(path)
    _check_keys(path, "the top level", data, _REPOS_KEYS)
    max_repos = _optional_count(path, "max_repos", data.get("max_repos"))
    raw_criteria = data.get("criteria") or {}
    if not isinstance(raw_criteria, dict):
        raise ConfigError(f"{path}: criteria must be a mapping")
    try:
        criteria = Criteria.from_json(raw_criteria)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    raw_repos = data.get("repos")
    if not isinstance(raw_repos, list) or not raw_repos:
        raise ConfigError(f"{path}: repos must be a non-empty list")
    repos: list[RepoSpec] = []
    ids: set[int] = set()
    slugs: set[str] = set()
    for position, item in enumerate(raw_repos):
        if not isinstance(item, dict):
            raise ConfigError(f"{path}: repos[{position}] must be a mapping")
        _check_keys(path, f"repos[{position}]", item, _REPO_KEYS)
        repo_id = item.get("id")
        if not _is_int(repo_id) or not 1 <= repo_id <= MAX_REPO_ID:
            raise ConfigError(f"{path}: repos[{position}].id must be an integer from 1 to {MAX_REPO_ID}")
        name = item.get("name")
        if not isinstance(name, str):
            raise ConfigError(f"{path}: repos[{position}].name must be a string like 'owner/name'")
        try:
            slug = slug_for(name)
        except ConfigError as exc:
            raise ConfigError(f"{path}: repos[{position}].name: {exc}") from exc
        issues = item.get("issues", True)
        if not isinstance(issues, bool):
            raise ConfigError(f"{path}: repos[{position}].issues must be true or false, got {issues!r}")
        if repo_id in ids:
            raise ConfigError(f"{path}: repos[{position}].id {repo_id} is used twice")
        if slug in slugs:
            raise ConfigError(f"{path}: repos[{position}].name {name} gives the slug {slug} twice")
        ids.add(repo_id)
        slugs.add(slug)
        url = _repo_url(path, name, item.get("url"))
        repos.append(RepoSpec(id=repo_id, name=name, url=url, issues=issues))
    return tuple(repos), criteria, max_repos


def _load_labels(path: Path) -> tuple[Settings, int, int | None]:
    data = _load_yaml(path)
    _check_keys(path, "the top level", data, _LABELS_KEYS)
    raw_settings = data.get("settings") or {}
    if not isinstance(raw_settings, dict):
        raise ConfigError(f"{path}: settings must be a mapping")
    try:
        settings = settings_from_overrides(raw_settings)
    except ConfigError as exc:
        raise ConfigError(f"{path}: settings: {exc}") from exc
    raw_snapshots = data.get("snapshots") or {}
    if not isinstance(raw_snapshots, dict):
        raise ConfigError(f"{path}: snapshots must be a mapping")
    _check_keys(path, "snapshots", raw_snapshots, _SNAPSHOT_KEYS)
    max_per_repo = raw_snapshots.get("max_per_repo", 8)
    if not _is_int(max_per_repo) or max_per_repo < 1:
        raise ConfigError(f"{path}: snapshots.max_per_repo must be an integer >= 1, got {max_per_repo!r}")
    raw_szz = data.get("szz") or {}
    if not isinstance(raw_szz, dict):
        raise ConfigError(f"{path}: szz must be a mapping")
    _check_keys(path, "szz", raw_szz, _SZZ_KEYS)
    szz_max_fixes = _optional_count(path, "szz.max_fixes", raw_szz.get("max_fixes"))
    return settings, max_per_repo, szz_max_fixes


def load_config(
    repos_path: Path = DEFAULT_REPOS_CONFIG, labels_path: Path = DEFAULT_LABELS_CONFIG
) -> PipelineConfig:
    """Read and validate both config files (see the module docstring).

    ``repos.yaml`` may hold only ``criteria``, ``max_repos`` (an integer >= 1, or null or absent for every
    passing repository) and ``repos``; each repository has a unique ``id`` (1 to 9,999,999), a unique slug,
    an optional ``url`` (default ``https://github.com/<name>.git``; an https URL must pass
    ``validate_repo_url`` for github.com, anything else must be an absolute local path) and an optional
    boolean ``issues`` (default true). Criteria are integers >= 0. ``labels.yaml`` may hold only
    ``settings`` (overrides of ``Settings``), ``snapshots`` (``max_per_repo`` >= 1, default 8) and ``szz``
    (``max_fixes`` >= 1, or null for no budget, the default). Raises :class:`ConfigError` naming the file
    and key.
    """
    repos, criteria, max_repos = _load_repos(repos_path)
    settings, max_snapshots, szz_max_fixes = _load_labels(labels_path)
    return PipelineConfig(
        repos=repos,
        criteria=criteria,
        settings=settings,
        max_snapshots=max_snapshots,
        max_repos=max_repos,
        szz_max_fixes=szz_max_fixes,
    )

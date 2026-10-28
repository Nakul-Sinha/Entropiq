"""Settings shared by the API service and the workers, read from the environment.

Every field is read from an environment variable of the same name in upper case (case insensitive), or from
a ``.env`` file in the current working directory. Real environment variables win over the file, and a
variable set to an empty string counts as unset, so ``GITHUB_TOKEN=`` in ``.env`` means "no token".

- ``DATABASE_URL``: SQLAlchemy URL of the Postgres database (``postgresql+asyncpg://...``). Any Postgres
  driver name works; :mod:`tdp_platform.db.session` switches to asyncpg or psycopg as needed.
- ``DATABASE_SCHEMA``: optional schema put first on the ``search_path`` of every connection. Unset in
  production (tables live in ``public``); tests use it to isolate themselves in a temporary schema.
- ``REDIS_URL``: Celery broker, result backend and progress pub/sub.
- ``REPO_CACHE_DIR``: folder of bare git mirrors (default ``$TDP_DATA_DIR/mirrors``, by default
  ``./data/mirrors``).
- ``MODEL_DIR``: folder of model bundles and ``production.json`` (default ``$TDP_MODELS_DIR``, by default
  ``./models``).
- ``GITHUB_TOKEN``, ``ANTHROPIC_API_KEY``, ``SENTRY_DSN``: optional secrets, hidden in ``repr``.
- ``ALLOWED_GIT_HOSTS``: comma list of hosts repositories may be cloned from (default
  ``github.com,gitlab.com``); entries are stripped, lower cased and de-duplicated.
- ``MAX_REPO_KB``: largest repository accepted for analysis, in kilobytes (default 1,000,000, about 1 GB).
- ``ADVISOR_ENABLED``: whether LLM advice may be requested. Defaults to true exactly when
  ``ANTHROPIC_API_KEY`` is set; an explicit value always wins (callers still need a key to call the LLM).

Settings are immutable. :func:`get_settings` caches one instance per process; tests that change the
environment call ``get_settings.cache_clear()``. Services that need extra settings subclass
:class:`PlatformSettings` in their own package.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Final

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from tdp_core import paths
from tdp_platform.db.session import validate_schema_name

DEFAULT_DATABASE_URL: Final = "postgresql+asyncpg://tdp:tdp@localhost:5432/tdp"
DEFAULT_REDIS_URL: Final = "redis://localhost:6379/0"
DEFAULT_ALLOWED_GIT_HOSTS: Final = ("github.com", "gitlab.com")
DEFAULT_MAX_REPO_KB: Final = 1_000_000


def _default_repo_cache_dir() -> Path:
    return paths.data_dir() / "mirrors"


class PlatformSettings(BaseSettings):
    """Runtime configuration of the platform (see the module docstring for every variable)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
        frozen=True,
    )

    database_url: str = Field(default=DEFAULT_DATABASE_URL, repr=False)
    database_schema: str | None = None
    redis_url: str = Field(default=DEFAULT_REDIS_URL, repr=False)
    repo_cache_dir: Path = Field(default_factory=_default_repo_cache_dir)
    model_dir: Path = Field(default_factory=paths.models_dir)
    github_token: SecretStr | None = None
    # Declared before advisor_enabled: its validator reads this value.
    anthropic_api_key: SecretStr | None = None
    sentry_dsn: SecretStr | None = None
    allowed_git_hosts: Annotated[tuple[str, ...], NoDecode] = DEFAULT_ALLOWED_GIT_HOSTS
    max_repo_kb: int = Field(default=DEFAULT_MAX_REPO_KB, gt=0)
    advisor_enabled: bool = Field(default=None, validate_default=True)

    @field_validator("database_schema", mode="after")
    @classmethod
    def _check_schema(cls, value: str | None) -> str | None:
        return None if value is None else validate_schema_name(value)

    @field_validator("allowed_git_hosts", mode="before")
    @classmethod
    def _parse_hosts(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            items: Iterable[Any] = value.split(",")
        elif isinstance(value, Iterable):
            items = value
        else:
            raise ValueError("expected a comma separated list of host names")
        hosts: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise ValueError(f"host names must be strings, got {item!r}")
            host = item.strip().lower()
            if host and host not in hosts:
                hosts.append(host)
        if not hosts:
            raise ValueError("at least one allowed git host is required")
        return tuple(hosts)

    @field_validator("advisor_enabled", mode="before")
    @classmethod
    def _default_advisor(cls, value: Any, info: ValidationInfo) -> Any:
        if value is not None:
            return value
        key = info.data.get("anthropic_api_key")
        return isinstance(key, SecretStr) and bool(key.get_secret_value())


@functools.lru_cache(maxsize=1)
def get_settings() -> PlatformSettings:
    """The process wide settings, read once from the environment (``cache_clear()`` re-reads them)."""
    return PlatformSettings()

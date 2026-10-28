"""Run the Alembic migrations of the platform database from Python.

Migrations always run on the synchronous psycopg driver (an asyncpg ``DATABASE_URL`` is switched
automatically) inside one transaction that first takes a Postgres advisory lock, so services starting
together (API and worker containers) migrate one after the other: the second waits, then finds the schema at
head. The command line equivalent is ``uv run alembic -c packages/platform/alembic.ini upgrade head``.

Alembic keeps its ``context`` and ``op`` proxies in module globals, so within one process these helpers run
one at a time (a lock serialises calls from several threads).

``alembic.ini`` and the ``alembic/`` folder live in ``packages/platform`` next to the package, not inside
it, so these helpers need a source checkout or an editable install (``FileNotFoundError`` otherwise).
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Final, TextIO

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import URL

from tdp_platform.db.session import create_sync_db_engine
from tdp_platform.settings import get_settings

PLATFORM_ROOT: Final = Path(__file__).resolve().parents[2]
ALEMBIC_INI: Final = PLATFORM_ROOT / "alembic.ini"
OFFLINE_URL: Final = "postgresql+psycopg://tdp@localhost:5432/tdp"
MIGRATION_LOCK_KEY: Final = 7_469_011_005

_ALEMBIC_LOCK = threading.Lock()


def alembic_config(*, database_url: str | URL | None = None, output_buffer: TextIO | None = None) -> Config:
    """Alembic configuration for the platform migrations.

    ``database_url`` overrides ``DATABASE_URL`` for ``alembic/env.py`` (kept in ``Config.attributes``, so a
    ``%`` in a password is not interpolated). Logging configuration from ``alembic.ini`` is skipped: the
    caller's logging stays as it is. ``output_buffer`` receives offline SQL. Raises ``FileNotFoundError``
    when ``alembic.ini`` is not available (installed wheel without the source tree).
    """
    if not ALEMBIC_INI.is_file():
        raise FileNotFoundError(f"{ALEMBIC_INI} not found; migrations need the packages/platform source tree")
    cfg = Config(str(ALEMBIC_INI), output_buffer=output_buffer)
    cfg.attributes["configure_logging"] = False
    if database_url is not None:
        cfg.attributes["database_url"] = database_url
    return cfg


def head_revision() -> str:
    """The revision id of the newest migration (single head)."""
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    if head is None:  # pragma: no cover - the initial migration always exists
        raise RuntimeError("no migrations found")
    return head


def _resolve(database_url: str | URL | None, schema: str | None) -> tuple[str | URL, str | None]:
    if database_url is not None:
        return database_url, schema
    settings = get_settings()
    return settings.database_url, settings.database_schema if schema is None else schema


def _run(database_url: str | URL | None, schema: str | None, *, upgrade: bool, revision: str) -> None:
    url, schema = _resolve(database_url, schema)
    engine = create_sync_db_engine(url, schema=schema, null_pool=True)
    try:
        with _ALEMBIC_LOCK, engine.begin() as connection:
            cfg = alembic_config(database_url=url)
            cfg.attributes["connection"] = connection
            if upgrade:
                command.upgrade(cfg, revision)
            else:
                command.downgrade(cfg, revision)
    finally:
        engine.dispose()


def upgrade(
    database_url: str | URL | None = None, revision: str = "head", *, schema: str | None = None
) -> None:
    """Migrate the database up to ``revision`` (default: the newest).

    Without ``database_url`` the settings are used (``DATABASE_URL`` and, unless ``schema`` is given,
    ``DATABASE_SCHEMA``). ``schema`` selects the schema to migrate through ``search_path``; it must exist.
    """
    _run(database_url, schema, upgrade=True, revision=revision)


def downgrade(
    database_url: str | URL | None = None, revision: str = "base", *, schema: str | None = None
) -> None:
    """Migrate the database down to ``revision`` (default: drop every table).

    Same arguments as :func:`upgrade`.
    """
    _run(database_url, schema, upgrade=False, revision=revision)


def migration_sql(*, downgrade: bool = False) -> str:
    """The SQL of all migrations (upgrade to head, or downgrade from head to base) without a database."""
    buffer = io.StringIO()
    cfg = alembic_config(database_url=OFFLINE_URL, output_buffer=buffer)
    with _ALEMBIC_LOCK:
        if downgrade:
            command.downgrade(cfg, "head:base", sql=True)
        else:
            command.upgrade(cfg, "head", sql=True)
    return buffer.getvalue()

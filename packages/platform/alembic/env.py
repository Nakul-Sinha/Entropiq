"""Alembic environment of the platform database.

The database URL comes from ``Config.attributes["database_url"]`` (set by :mod:`tdp_platform.db.migrate`) or
from the settings (``DATABASE_URL`` and ``DATABASE_SCHEMA``). Migrations run on psycopg inside one
transaction that holds an advisory lock, so concurrent migrators run one after the other. A connection
passed in ``Config.attributes["connection"]`` is used as is (Alembic's shared connection recipe).
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, text

from tdp_platform.db.migrate import MIGRATION_LOCK_KEY
from tdp_platform.db.models import Base
from tdp_platform.db.session import create_sync_db_engine, to_sync_url
from tdp_platform.settings import get_settings

config = context.config
target_metadata = Base.metadata

if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    # Keep loggers created before this point working (the default would disable them).
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def run_migrations_offline() -> None:
    """Write the migration SQL to the output buffer instead of running it."""
    url = config.attributes.get("database_url") or get_settings().database_url
    context.configure(
        url=to_sync_url(url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        # Held until the transaction ends: a concurrent migrator waits, then finds the schema at head.
        lock = text("SELECT pg_advisory_xact_lock(CAST(:key AS BIGINT))")
        connection.execute(lock, {"key": MIGRATION_LOCK_KEY})
        context.run_migrations()


def run_migrations_online() -> None:
    """Run the migrations against the database."""
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    settings = get_settings()
    url = config.attributes.get("database_url") or settings.database_url
    engine = create_sync_db_engine(url, schema=settings.database_schema, null_pool=True)
    try:
        with engine.connect() as conn:
            _run(conn)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

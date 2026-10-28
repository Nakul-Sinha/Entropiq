"""Helpers for tests that need a real Postgres database (platform, worker and API tests).

Integration tests share one database in CI. Each test session creates its own temporary schema with
:func:`temporary_schema` (migrated to head), points its engines at it (``schema=`` or ``DATABASE_SCHEMA``)
and empties the tables between tests with :func:`truncate_all`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import URL, Connection, text

from tdp_platform.db.migrate import upgrade
from tdp_platform.db.models import Base
from tdp_platform.db.session import create_sync_db_engine, validate_schema_name


@contextmanager
def temporary_schema(
    database_url: str | URL, *, prefix: str = "tdp_test", migrate: bool = True
) -> Iterator[str]:
    """Create a uniquely named schema, migrate it to head (unless ``migrate`` is False) and yield its name.

    The schema and everything in it is dropped on exit. ``prefix`` must be a valid schema name start
    (lower case letters, digits, underscores); the name is ``<prefix>_<12 hex characters>``.
    """
    name = validate_schema_name(f"{prefix}_{uuid.uuid4().hex[:12]}")
    engine = create_sync_db_engine(database_url, null_pool=True)
    quoted = engine.dialect.identifier_preparer.quote_identifier(name)
    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {quoted}"))
        try:
            if migrate:
                upgrade(database_url, schema=name)
            yield name
        finally:
            with engine.begin() as conn:
                conn.execute(text("SET LOCAL lock_timeout = '30s'"))
                conn.execute(text(f"DROP SCHEMA IF EXISTS {quoted} CASCADE"))
    finally:
        engine.dispose()


def truncate_all(connection: Connection) -> None:
    """Empty every table of the models and restart their id sequences, inside the caller's transaction.

    Tables are named unqualified, so the connection's ``search_path`` picks the schema; ``alembic_version``
    is left alone. Async callers use ``await conn.run_sync(truncate_all)``.
    """
    preparer = connection.dialect.identifier_preparer
    tables = ", ".join(preparer.quote(table.name) for table in Base.metadata.sorted_tables)
    connection.execute(text("SET LOCAL lock_timeout = '10s'"))
    connection.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))

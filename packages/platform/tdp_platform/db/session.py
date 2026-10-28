"""Engines, sessions and URL helpers for Postgres.

The application talks to Postgres through two drivers: asyncpg for the async API and worker code
(:func:`create_async_db_engine`) and psycopg 3 for synchronous code such as migrations and the parse cache
(:func:`create_sync_db_engine`). Both accept any Postgres URL (``postgres://``, ``postgresql://`` or
``postgresql+<driver>://``) and switch the driver themselves, so one ``DATABASE_URL`` serves both.

Engines are lazy: creating one never opens a connection. Every engine serialises JSON and JSONB values with
:func:`json_dumps`, so contracts dataclasses, tuples, datetimes and numpy scalars can be stored directly,
while NaN and infinities are rejected (Postgres JSONB cannot hold them).

An optional ``schema`` puts that schema alone on each connection's ``search_path``, which is how tests
isolate themselves in a temporary schema. A pooled :class:`AsyncEngine` is bound to the event loop that first
used it; with ``null_pool=True`` (no pooling) one engine can be reused across several ``asyncio.run()``
calls, which suits Celery workers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final, Literal

from sqlalchemy import URL, Engine, NullPool, create_engine, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from tdp_core.serialize import to_jsonable

ASYNC_DRIVERNAME: Final = "postgresql+asyncpg"
SYNC_DRIVERNAME: Final = "postgresql+psycopg"
SCHEMA_NAME_PATTERN: Final = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

_SURROGATE = re.compile("[\ud800-\udfff]")
# An escaped NUL, but not an escaped backslash followed by the letters "u0000".
_NUL_ESCAPE = re.compile(r"(?<!\\)(?:\\\\)*\\u0000")


def validate_schema_name(name: str) -> str:
    """Return ``name`` if it is a safe, unquoted Postgres schema name, else raise ``ValueError``.

    Allowed: 1 to 63 characters, lower case ASCII letters, digits and underscores, not starting with a digit.
    Schema names end up inside SQL text (``search_path``, ``CREATE SCHEMA``), so nothing else is accepted.
    """
    if not isinstance(name, str) or not SCHEMA_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid schema name {name!r}: use 1 to 63 of [a-z0-9_], not starting with a digit")
    return name


def _with_driver(url: str | URL, drivername: str) -> URL:
    try:
        parsed = make_url(url)
    except ArgumentError:
        # The message would repeat the URL, which may hold a password.
        raise ValueError("not a valid database URL") from None
    backend = parsed.drivername.split("+", 1)[0]
    if backend not in ("postgres", "postgresql"):
        raise ValueError(f"only Postgres URLs are supported, got driver {parsed.drivername!r}")
    return parsed.set(drivername=drivername)


def to_async_url(url: str | URL) -> URL:
    """The same Postgres URL with the asyncpg driver; host, port, user, password and query are kept.

    Raises ``ValueError`` for unparsable or non-Postgres URLs.
    """
    return _with_driver(url, ASYNC_DRIVERNAME)


def to_sync_url(url: str | URL) -> URL:
    """The same Postgres URL with the psycopg (version 3) driver; everything else is kept.

    Raises ``ValueError`` for unparsable or non-Postgres URLs.
    """
    return _with_driver(url, SYNC_DRIVERNAME)


def json_dumps(value: Any) -> str:
    """Compact JSON text for a JSONB column.

    Values go through :func:`tdp_core.serialize.to_jsonable` first (dataclasses become objects, tuples
    lists, datetimes ISO 8601 strings, numpy scalars numbers). Non ASCII text is kept as is. Raises
    ``ValueError`` for what JSONB cannot store (NaN or infinite floats, NUL characters, lone surrogates)
    and ``TypeError`` for unsupported objects; pass pydantic models as ``model.model_dump(mode="json")``.
    """
    text = json.dumps(to_jsonable(value), allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    if _SURROGATE.search(text) or ("\\u0000" in text and _NUL_ESCAPE.search(text)):
        raise ValueError("JSONB cannot store NUL characters or lone surrogates")
    return text


def connect_args(driver: Literal["asyncpg", "psycopg"], schema: str | None) -> dict[str, Any]:
    """Driver specific connect arguments setting ``search_path`` to ``schema`` (empty without a schema)."""
    if schema is None:
        return {}
    validate_schema_name(schema)
    if driver == "asyncpg":
        return {"server_settings": {"search_path": schema}}
    if driver == "psycopg":
        return {"options": f"-c search_path={schema}"}
    raise ValueError(f"unknown driver {driver!r}")


def create_async_db_engine(
    url: str | URL, *, schema: str | None = None, null_pool: bool = False, echo: bool = False
) -> AsyncEngine:
    """An asyncpg :class:`AsyncEngine` for ``url`` (any Postgres driver name), created without connecting.

    ``schema`` sets each connection's ``search_path``; ``null_pool=True`` opens a fresh connection per
    checkout, which makes the engine safe to reuse across event loops. Connections are pinged before use.
    """
    kwargs: dict[str, Any] = {"poolclass": NullPool} if null_pool else {}
    return create_async_engine(
        to_async_url(url),
        json_serializer=json_dumps,
        pool_pre_ping=True,
        echo=echo,
        connect_args=connect_args("asyncpg", schema),
        **kwargs,
    )


def create_sync_db_engine(
    url: str | URL, *, schema: str | None = None, null_pool: bool = False, echo: bool = False
) -> Engine:
    """A psycopg (version 3) :class:`Engine` for ``url``, created without connecting.

    Same options as :func:`create_async_db_engine`. Engines are not fork safe: create them in the process
    that uses them.
    """
    kwargs: dict[str, Any] = {"poolclass": NullPool} if null_pool else {}
    return create_engine(
        to_sync_url(url),
        json_serializer=json_dumps,
        pool_pre_ping=True,
        echo=echo,
        connect_args=connect_args("psycopg", schema),
        **kwargs,
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory for ``engine``; objects stay readable after commit (``expire_on_commit=False``)."""
    return async_sessionmaker(engine, expire_on_commit=False)

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy import NullPool, QueuePool
from sqlalchemy.pool import AsyncAdaptedQueuePool

from tdp_core.contracts import CostEstimate
from tdp_platform.db.session import (
    ASYNC_DRIVERNAME,
    SYNC_DRIVERNAME,
    connect_args,
    create_async_db_engine,
    create_sync_db_engine,
    json_dumps,
    make_session_factory,
    to_async_url,
    to_sync_url,
    validate_schema_name,
)

UNREACHABLE = "postgresql://u:p@nonexistent.invalid/db"


@pytest.mark.parametrize(
    "url",
    [
        "postgres://user:p%40ss@db.example:6543/tdp?sslmode=require",
        "postgresql://user:p%40ss@db.example:6543/tdp?sslmode=require",
        "postgresql+psycopg://user:p%40ss@db.example:6543/tdp?sslmode=require",
        "postgresql+psycopg2://user:p%40ss@db.example:6543/tdp?sslmode=require",
        "postgresql+asyncpg://user:p%40ss@db.example:6543/tdp?sslmode=require",
    ],
)
def test_driver_switch_keeps_everything_else(url: str) -> None:
    for convert, driver in ((to_async_url, ASYNC_DRIVERNAME), (to_sync_url, SYNC_DRIVERNAME)):
        out = convert(url)
        assert out.drivername == driver
        assert out.host == "db.example"
        assert out.port == 6543
        assert out.database == "tdp"
        assert out.username == "user"
        assert out.password == "p@ss"
        assert dict(out.query) == {"sslmode": "require"}
    assert to_sync_url(to_async_url(url)) == to_sync_url(url)


@pytest.mark.parametrize("url", ["sqlite:///x.db", "mysql://u:p@h/db", "garbage", ""])
def test_non_postgres_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        to_async_url(url)
    with pytest.raises(ValueError):
        to_sync_url(url)


def test_rejection_message_does_not_repeat_the_url() -> None:
    with pytest.raises(ValueError) as excinfo:
        to_sync_url("not a url with secret-password")
    assert "secret-password" not in str(excinfo.value)


@pytest.mark.parametrize("name", ["public", "tdp_test_0a1b2c3d4e5f", "_x", "a" * 63])
def test_valid_schema_names(name: str) -> None:
    assert validate_schema_name(name) == name


@pytest.mark.parametrize("name", ["", "Public", "with-dash", 'q"uote', "semi;colon", "1abc", "a" * 64, "a b"])
def test_invalid_schema_names(name: str) -> None:
    with pytest.raises(ValueError):
        validate_schema_name(name)


def test_connect_args() -> None:
    assert connect_args("asyncpg", "tdp_x") == {"server_settings": {"search_path": "tdp_x"}}
    assert connect_args("psycopg", "tdp_x") == {"options": "-c search_path=tdp_x"}
    assert connect_args("asyncpg", None) == {}
    assert connect_args("psycopg", None) == {}
    with pytest.raises(ValueError):
        connect_args("psycopg", "bad name")


@pytest.mark.parametrize(("null_pool", "pool_class"), [(False, AsyncAdaptedQueuePool), (True, NullPool)])
def test_async_engine_is_lazy(null_pool: bool, pool_class: type) -> None:
    engine = create_async_db_engine(UNREACHABLE, schema="tdp_x", null_pool=null_pool, echo=True)
    try:
        assert engine.url.drivername == ASYNC_DRIVERNAME
        assert engine.dialect.driver == "asyncpg"
        assert type(engine.sync_engine.pool) is pool_class
        assert engine.dialect._json_serializer is json_dumps
        assert engine.echo is True
    finally:
        engine.sync_engine.dispose()


@pytest.mark.parametrize(("null_pool", "pool_class"), [(False, QueuePool), (True, NullPool)])
def test_sync_engine_is_lazy(null_pool: bool, pool_class: type) -> None:
    engine = create_sync_db_engine(UNREACHABLE, null_pool=null_pool)
    try:
        assert engine.url.drivername == SYNC_DRIVERNAME
        assert engine.dialect.driver == "psycopg"
        assert type(engine.pool) is pool_class
        assert engine.dialect._json_serializer is json_dumps
    finally:
        engine.dispose()


@dataclass(frozen=True)
class _Point:
    x: int
    when: datetime


def test_json_dumps_is_compact_and_strict() -> None:
    assert json_dumps({"a": [1, 2], "b": None}) == '{"a":[1,2],"b":null}'
    assert json_dumps((1, "two")) == '[1,"two"]'
    assert json_dumps({"name": "ü→"}) == '{"name":"ü→"}'
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert json.loads(json_dumps(_Point(3, when))) == {"x": 3, "when": "2026-01-02T03:04:05Z"}
    cost = CostEstimate(hours=1.5, low=1.2, high=1.8, breakdown={"complexity": 90.0})
    assert json.loads(json_dumps(cost))["breakdown"] == {"complexity": 90.0}
    assert json.loads(json_dumps({"v": np.float32(0.5), "n": np.int64(3)})) == {"v": 0.5, "n": 3}
    for bad in (float("nan"), float("inf"), {"x": float("-inf")}):
        with pytest.raises(ValueError):
            json_dumps(bad)
    with pytest.raises(TypeError):
        json_dumps(object())


def test_json_dumps_rejects_text_jsonb_cannot_store() -> None:
    for bad in ({"text": "a\x00b"}, ["\ud800"], {"key\udcff": 1}):
        with pytest.raises(ValueError):
            json_dumps(bad)
    # A literal backslash followed by "u0000" is fine: only a real NUL is rejected.
    assert json.loads(json_dumps({"text": "\\u0000"})) == {"text": "\\u0000"}
    with pytest.raises(ValueError):
        json_dumps(["\\" + "\x00"])  # a backslash, then a real NUL


def test_session_factory_keeps_objects_after_commit() -> None:
    engine = create_async_db_engine(UNREACHABLE)
    try:
        factory = make_session_factory(engine)
        assert factory.kw["expire_on_commit"] is False
        assert factory.kw["bind"] is engine
    finally:
        engine.sync_engine.dispose()

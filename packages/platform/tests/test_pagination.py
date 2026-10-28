from __future__ import annotations

import base64
import json
import re
import uuid
from typing import Any

import pytest
from sqlalchemy import make_url
from sqlalchemy.sql import Select

from tdp_platform.db.pagination import (
    MAX_CURSOR_CHARS,
    SORT_KEYS,
    SORT_ORDERS,
    CursorValue,
    InvalidCursorError,
    SortKey,
    SortOrder,
    decode_cursor,
    encode_cursor,
)
from tdp_platform.db.repos import ranked_files_count, ranked_files_query

DIALECT = make_url("postgresql+asyncpg://").get_dialect()()
AID = uuid.UUID("12345678-1234-5678-1234-567812345678")
VALUES: dict[SortKey, CursorValue] = {
    "priority": 87,
    "risk": 0.10000000149011612,
    "cost": 12.5,
    "path": "pkg/ü.py",
}


def _b64(text: str | bytes) -> str:
    raw = text.encode("utf-8") if isinstance(text, str) else text
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


@pytest.mark.parametrize("sort", SORT_KEYS)
@pytest.mark.parametrize("order", SORT_ORDERS)
def test_cursor_round_trip(sort: SortKey, order: SortOrder) -> None:
    cursor = encode_cursor(sort, order, VALUES[sort], 42)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", cursor)
    value, file_id = decode_cursor(cursor, sort, order)
    assert value == VALUES[sort]
    assert type(value) is type(VALUES[sort])
    assert file_id == 42


def test_integral_float_values_decode_as_floats() -> None:
    value, _ = decode_cursor(encode_cursor("cost", "asc", 3, 1), "cost", "asc")
    assert value == 3.0 and isinstance(value, float)


def test_cursor_is_bound_to_its_sort_and_order() -> None:
    cursor = encode_cursor("priority", "desc", 50, 7)
    with pytest.raises(InvalidCursorError):
        decode_cursor(cursor, "priority", "asc")
    with pytest.raises(InvalidCursorError):
        decode_cursor(cursor, "risk", "desc")


@pytest.mark.parametrize(
    ("sort", "cursor"),
    [
        ("priority", ""),
        ("priority", "!!!"),
        ("priority", "abc="),
        ("priority", "A"),
        ("priority", _b64("not json")),
        ("priority", _b64(b"\xff\xfe\xfd")),
        ("priority", _b64('{"a": 1}')),
        ("priority", _b64('["priority","desc",1]')),
        ("priority", _b64('["priority","desc",true,1]')),
        ("priority", _b64('["priority","desc",101,1]')),
        ("priority", _b64('["priority","desc",-1,1]')),
        ("priority", _b64('["priority","desc",5.5,1]')),
        ("risk", _b64('["risk","desc",NaN,1]')),
        ("risk", _b64('["risk","desc",Infinity,1]')),
        ("risk", _b64('["risk","desc","0.5",1]')),
        ("risk", _b64('["risk","desc",false,1]')),
        ("path", _b64('["path","desc",5,1]')),
        ("path", _b64('["path","desc","a\\u0000b",1]')),
        ("path", _b64('["path","desc","\\ud800",1]')),
        ("priority", _b64('["priority","desc",5,0]')),
        ("priority", _b64('["priority","desc",5,-1]')),
        ("priority", _b64('["priority","desc",5,1.5]')),
        ("priority", _b64('["priority","desc",5,true]')),
        ("priority", _b64('["priority","desc",5,9223372036854775808]')),
        ("priority", "A" * (MAX_CURSOR_CHARS + 1)),
    ],
)
def test_malformed_cursors_are_rejected(sort: SortKey, cursor: str) -> None:
    with pytest.raises(InvalidCursorError):
        decode_cursor(cursor, sort, "desc")


def test_invalid_cursor_error_is_a_value_error() -> None:
    assert issubclass(InvalidCursorError, ValueError)


def test_encode_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        encode_cursor("risk", "desc", float("nan"), 1)


# ---------- compiled queries ----------


def _sql(stmt: Select[Any]) -> str:
    return str(stmt.compile(dialect=DIALECT))


def _params(stmt: Select[Any]) -> dict[str, Any]:
    return dict(stmt.compile(dialect=DIALECT).params)


def _query(**kw: Any) -> Select[Any]:
    args: dict[str, Any] = {
        "sort": "priority",
        "order": "desc",
        "bands": None,
        "debt_type": None,
        "path_prefix": None,
        "after": None,
        "limit": 37,
    }
    args.update(kw)
    return ranked_files_query(AID, **args)


@pytest.mark.parametrize(
    ("sort", "order", "expected"),
    [
        ("priority", "desc", "ORDER BY file_results.priority DESC, file_results.id ASC"),
        ("priority", "asc", "ORDER BY file_results.priority ASC, file_results.id ASC"),
        ("risk", "desc", "ORDER BY file_results.risk DESC, file_results.id ASC"),
        ("cost", "asc", "ORDER BY file_results.cost_high ASC, file_results.id ASC"),
        ("path", "desc", 'ORDER BY file_results.path COLLATE "C" DESC, file_results.id ASC'),
    ],
)
def test_order_by(sort: SortKey, order: SortOrder, expected: str) -> None:
    sql = _sql(_query(sort=sort, order=order))
    assert expected in sql
    assert "file_results.id >" not in sql


def test_keyset_condition_follows_the_order() -> None:
    desc = _sql(_query(after=(80, 12)))
    assert "file_results.priority < " in desc and "file_results.id > " in desc
    asc = _sql(_query(order="asc", after=(80, 12)))
    assert "file_results.priority > " in asc and "file_results.id > " in asc
    path = _sql(_query(sort="path", order="asc", after=("pkg/a.py", 3)))
    assert '(file_results.path COLLATE "C") > ' in path
    params = _params(_query(after=(80, 12)))
    assert 80 in params.values() and 12 in params.values()


def test_limit_fetches_one_extra_row() -> None:
    sql = _sql(_query())
    assert "LIMIT" in sql
    assert 38 in _params(_query()).values()


def test_filters() -> None:
    stmt = _query(bands=["High", "Critical"], debt_type="test", path_prefix="pkg/a_%")
    sql = _sql(stmt)
    assert "file_results.band IN" in sql
    assert "file_results.debt_types @> " in sql
    assert "LIKE" in sql and "ESCAPE '/'" in sql
    params = _params(stmt)
    assert ["test"] in params.values()
    assert "pkg//a/_/%" in params.values()  # "/" is the escape character, so it is escaped too


def test_empty_filters_are_ignored() -> None:
    sql = _sql(_query(bands=[], path_prefix=""))
    assert " IN " not in sql and "LIKE" not in sql


def test_only_summary_columns_are_selected() -> None:
    columns = _sql(_query()).split("FROM", 1)[0]
    for name in ("priority", "band", "risk_low", "cost_high", "debt_types", "top_reason", "blob_sha"):
        assert f"file_results.{name}" in columns
    for name in ("features", "shap", "reasons", "extra"):
        assert f"file_results.{name}" not in columns


def test_count_query_uses_the_same_filters() -> None:
    sql = _sql(ranked_files_count(AID, bands=["Low"], debt_type="admitted", path_prefix="src/"))
    assert "count(*)" in sql
    assert "file_results.band IN" in sql and "@>" in sql and "LIKE" in sql
    assert "ORDER BY" not in sql and "LIMIT" not in sql


@pytest.mark.parametrize(
    "kw",
    [
        {"limit": 0},
        {"limit": 501},
        {"limit": True},
        {"sort": "size"},
        {"order": "up"},
        {"bands": ["Severe"]},
        {"bands": "High"},
        {"debt_type": "bogus"},
        {"path_prefix": "a\x00b"},
    ],
)
def test_invalid_arguments(kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _query(**kw)


def test_count_rejects_invalid_filters() -> None:
    with pytest.raises(ValueError):
        ranked_files_count(AID, bands=["Severe"], debt_type=None, path_prefix=None)


def test_cursor_payload_is_compact_json() -> None:
    cursor = encode_cursor("path", "asc", "pkg/ü.py", 9)
    raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode("utf-8")
    assert json.loads(raw) == ["path", "asc", "pkg/ü.py", 9]
    assert " " not in raw

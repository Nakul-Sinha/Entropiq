"""JSON round-tripping for the dataclasses in :mod:`tdp_core.contracts`.

``to_jsonable`` turns dataclasses into plain dicts (tuples and sets become lists, datetimes become
ISO 8601 strings ending in ``Z``, numpy scalars become Python numbers). DataFrame fields are skipped.
``from_jsonable(cls, data)`` rebuilds a dataclass using its type hints.
"""

from __future__ import annotations

import dataclasses
import types
from datetime import UTC, datetime
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

import pandas as pd


def to_jsonable(obj: Any) -> Any:
    """Convert ``obj`` into JSON-compatible Python values. Raises ``TypeError`` for unsupported types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if isinstance(value, pd.DataFrame):
                continue
            out[f.name] = to_jsonable(value)
        return out
    if isinstance(obj, datetime):
        return _iso(obj)
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(v) for v in obj)
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def from_jsonable[T](cls: type[T], data: Any) -> T:
    """Rebuild an instance of ``cls`` (usually a contracts dataclass) from ``to_jsonable`` output."""
    return cast(T, _convert(cls, data))


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("naive datetimes are not allowed; use timezone-aware UTC")
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _convert(tp: Any, value: Any) -> Any:
    if tp is Any:
        return value
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{value!r} is not one of {args}")
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors: list[Exception] = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _convert(arg, value)
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(exc)
        raise ValueError(f"{value!r} matches none of {args}: {errors}")
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v) for v in value)
        if len(args) != len(value):
            raise ValueError(f"expected {len(args)} items, got {len(value)}")
        return tuple(_convert(a, v) for a, v in zip(args, value, strict=True))
    if origin is list:
        return [_convert(args[0], v) for v in value]
    if origin in (set, frozenset):
        return origin(_convert(args[0], v) for v in value)
    if origin is dict:
        key_t, val_t = args
        return {_convert(key_t, k): _convert(val_t, v) for k, v in value.items()}
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        hints = get_type_hints(tp)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(tp):
            if f.name in value:
                kwargs[f.name] = _convert(hints[f.name], value[f.name])
            elif hints[f.name] is pd.DataFrame:
                kwargs[f.name] = pd.DataFrame()
        return tp(**kwargs)
    if tp is datetime:
        if isinstance(value, datetime):
            return value
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError(f"naive datetime {value!r}")
        return parsed.astimezone(UTC)
    if tp is pd.DataFrame:
        return pd.DataFrame() if value is None else pd.DataFrame(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise TypeError(f"expected bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise TypeError(f"expected int, got {value!r}")
        return int(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"expected float, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise TypeError(f"expected str, got {value!r}")
        return value
    raise TypeError(f"unsupported type {tp!r}")

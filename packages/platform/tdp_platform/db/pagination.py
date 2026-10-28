"""Opaque keyset cursors for paging through ranked files.

A cursor encodes ``(sort, order, value, file_id)``: the sort key and order it was issued for, the sort
value of the last file on the page and that file's id (ties are broken by id ascending). The text is URL
safe base64 of compact JSON without padding, so it can go into a query string unchanged. A cursor is only
valid for the sort and order it was issued for; anything malformed raises :class:`InvalidCursorError`
(a ``ValueError``), which the API maps to HTTP 400.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from typing import Final, Literal

SortKey = Literal["priority", "risk", "cost", "path"]
SortOrder = Literal["asc", "desc"]
SORT_KEYS: Final[tuple[SortKey, ...]] = ("priority", "risk", "cost", "path")
SORT_ORDERS: Final[tuple[SortOrder, ...]] = ("desc", "asc")
DEFAULT_PAGE_SIZE: Final = 50
MAX_PAGE_SIZE: Final = 500
MAX_CURSOR_CHARS: Final = 8192
MAX_FILE_ID: Final = 2**63 - 1  # largest BIGINT

CursorValue = int | float | str

_CURSOR_CHARS = re.compile(r"^[A-Za-z0-9_-]+$")
_BAD_TEXT = re.compile("[\x00\ud800-\udfff]")  # Postgres text cannot hold NUL or lone surrogates


class InvalidCursorError(ValueError):
    """The cursor is malformed, tampered with, or was issued for another sort or order."""


def encode_cursor(sort: SortKey, order: SortOrder, value: CursorValue, file_id: int) -> str:
    """Opaque cursor pointing just after the file ``file_id`` whose sort value is ``value``.

    ``value`` is the priority (int), risk or ``cost_high`` (float, hours) or path (str) of that file.
    Raises ``ValueError`` for NaN or infinite values.
    """
    payload = json.dumps(
        [sort, order, value, file_id], allow_nan=False, ensure_ascii=False, separators=(",", ":")
    )
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, sort: SortKey, order: SortOrder) -> tuple[CursorValue, int]:
    """The ``(value, file_id)`` stored in ``cursor``, checked against the requested ``sort`` and ``order``.

    Values are typed by sort key: priority an int 0..100, risk and cost a finite float, path a string
    without NUL characters or lone surrogates. ``file_id`` is a positive BIGINT. Cursors longer than
    ``MAX_CURSOR_CHARS`` are rejected. Raises :class:`InvalidCursorError` for anything else.
    """
    try:
        return _decode(cursor, sort, order)
    except InvalidCursorError:
        raise
    except (ValueError, TypeError, binascii.Error) as exc:
        raise InvalidCursorError("invalid cursor") from exc


def _decode(cursor: str, sort: SortKey, order: SortOrder) -> tuple[CursorValue, int]:
    if not isinstance(cursor, str) or len(cursor) > MAX_CURSOR_CHARS or not _CURSOR_CHARS.fullmatch(cursor):
        raise InvalidCursorError("invalid cursor")
    raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, list) or len(data) != 4:
        raise InvalidCursorError("invalid cursor")
    c_sort, c_order, value, file_id = data
    if c_sort != sort or c_order != order:
        raise InvalidCursorError("cursor was issued for a different sort or order")
    if isinstance(file_id, bool) or not isinstance(file_id, int) or not 1 <= file_id <= MAX_FILE_ID:
        raise InvalidCursorError("invalid cursor")
    return _check_value(sort, value), file_id


def _check_value(sort: SortKey, value: object) -> CursorValue:
    if sort == "priority":
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            raise InvalidCursorError("invalid cursor")
        return value
    if sort in ("risk", "cost"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise InvalidCursorError("invalid cursor")
        return float(value)
    if sort == "path":
        if not isinstance(value, str) or _BAD_TEXT.search(value):
            raise InvalidCursorError("invalid cursor")
        return value
    raise InvalidCursorError("invalid cursor")

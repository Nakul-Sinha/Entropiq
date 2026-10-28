"""The synchronous parse cache against Postgres (CI only)."""

from __future__ import annotations

import dataclasses
import logging

import pytest
from sqlalchemy import Engine, func, insert, select

from tdp_core.contracts import EXTRACTOR_VERSION, CommentInfo, FileSyntax
from tdp_platform.db.models import Base
from tdp_platform.db.repos import PostgresStaticCache

pytestmark = pytest.mark.integration

SHA = "0123456789abcdef0123456789abcdef01234567"
TABLE = Base.metadata.tables["static_cache"]


def _rows(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(TABLE)).scalar_one())


def test_round_trip(sync_engine: Engine, clean_db: None, syntax_sample: FileSyntax) -> None:
    cache = PostgresStaticCache(sync_engine)
    assert cache.get(SHA) is None
    cache.put(SHA, syntax_sample)
    assert cache.get(SHA) == syntax_sample
    cache.put(SHA, dataclasses.replace(syntax_sample, path="elsewhere.py"))
    assert _rows(sync_engine) == 1
    assert cache.get(SHA) == syntax_sample  # the first entry is kept
    assert PostgresStaticCache(sync_engine, extractor_version="other").get(SHA) is None
    cache.close()  # a borrowed engine stays usable
    assert cache.get(SHA) == syntax_sample


def test_unicode_text_round_trips(sync_engine: Engine, clean_db: None, syntax_sample: FileSyntax) -> None:
    cache = PostgresStaticCache(sync_engine)
    fancy = dataclasses.replace(syntax_sample, comments=(CommentInfo(line=1, text="naïve → café ✓"),))
    cache.put(SHA, fancy)
    assert cache.get(SHA) == fancy


def test_unstorable_syntax_is_not_cached(
    sync_engine: Engine, clean_db: None, syntax_sample: FileSyntax
) -> None:
    cache = PostgresStaticCache(sync_engine)
    cache.put(SHA, dataclasses.replace(syntax_sample, comments=(CommentInfo(line=1, text="a\x00b"),)))
    assert _rows(sync_engine) == 0
    assert cache.get(SHA) is None


def test_corrupt_rows_count_as_misses(
    sync_engine: Engine, clean_db: None, caplog: pytest.LogCaptureFixture
) -> None:
    with sync_engine.begin() as conn:
        conn.execute(
            insert(TABLE).values(blob_sha=SHA, extractor_version=EXTRACTOR_VERSION, syntax={"foo": 1})
        )
    cache = PostgresStaticCache(sync_engine)
    with caplog.at_level(logging.WARNING, logger="tdp_platform.db.repos"):
        assert cache.get(SHA) is None
    assert "undecodable" in caplog.text


def test_from_url_owns_its_engine(
    pg_url: str, pg_schema: str, clean_db: None, syntax_sample: FileSyntax
) -> None:
    with PostgresStaticCache.from_url(pg_url, schema=pg_schema) as cache:
        cache.put(SHA, syntax_sample)
        assert cache.get(SHA) == syntax_sample

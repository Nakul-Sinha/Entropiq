from __future__ import annotations

import dataclasses
from typing import Any, Protocol, cast, runtime_checkable

import pytest
from sqlalchemy import Engine

from tdp_core.contracts import EXTRACTOR_VERSION, CommentInfo, FileSyntax, RawImport
from tdp_platform.db.repos import PostgresStaticCache, syntax_is_storable

SHA = "0123456789abcdef0123456789abcdef01234567"


@runtime_checkable
class StaticCache(Protocol):
    """Copy of the feature assembly cache protocol."""

    def get(self, blob_sha: str) -> FileSyntax | None: ...

    def put(self, blob_sha: str, syntax: FileSyntax) -> None: ...


class ExplodingEngine:
    """Stands in for an engine that must never be used."""

    def connect(self) -> Any:
        raise AssertionError("the engine must not be used")

    def begin(self) -> Any:
        raise AssertionError("the engine must not be used")

    def dispose(self) -> None:
        raise AssertionError("a borrowed engine must not be disposed")


@pytest.fixture
def cache() -> PostgresStaticCache:
    return PostgresStaticCache(cast(Engine, ExplodingEngine()))


def test_implements_the_static_cache_protocol(cache: PostgresStaticCache) -> None:
    assert isinstance(cache, StaticCache)
    assert cache.extractor_version == EXTRACTOR_VERSION


@pytest.mark.parametrize("sha", ["", "abc", SHA.upper(), "g" * 40, SHA + "0"])
def test_malformed_shas_are_rejected_before_any_query(
    cache: PostgresStaticCache, syntax_sample: FileSyntax, sha: str
) -> None:
    with pytest.raises(ValueError):
        cache.get(sha)
    with pytest.raises(ValueError):
        cache.put(sha, syntax_sample)


def test_syntax_is_storable(syntax_sample: FileSyntax) -> None:
    assert syntax_sample.comments and syntax_sample.functions and syntax_sample.imports
    assert syntax_is_storable(syntax_sample)
    nul_comment = dataclasses.replace(syntax_sample, comments=(CommentInfo(line=1, text="a\x00b"),))
    assert not syntax_is_storable(nul_comment)
    surrogate_token = dataclasses.replace(syntax_sample, tokens=(*syntax_sample.tokens, "\ud800"))
    assert not syntax_is_storable(surrogate_token)
    bad_import = dataclasses.replace(
        syntax_sample, imports=(RawImport(module="m", names=("x\udcff",), level=0, line=1),)
    )
    assert not syntax_is_storable(bad_import)
    assert not syntax_is_storable(dataclasses.replace(syntax_sample, path="a\x00.py"))


def test_unstorable_syntax_is_skipped_without_a_query(
    cache: PostgresStaticCache, syntax_sample: FileSyntax
) -> None:
    unstorable = dataclasses.replace(syntax_sample, comments=(CommentInfo(line=1, text="\x00"),))
    cache.put(SHA, unstorable)  # would raise AssertionError if the engine were touched


def test_borrowed_engine_is_not_disposed(cache: PostgresStaticCache) -> None:
    with cache as entered:
        assert entered is cache
    cache.close()


def test_extractor_version_must_not_be_empty() -> None:
    with pytest.raises(ValueError):
        PostgresStaticCache(cast(Engine, ExplodingEngine()), extractor_version="")


def test_from_url_creates_a_lazy_owned_engine() -> None:
    with PostgresStaticCache.from_url(
        "postgresql+asyncpg://u:p@nonexistent.invalid/db", schema="tdp_x", extractor_version="7"
    ) as owned:
        assert owned.extractor_version == "7"
        engine = owned._engine
        assert engine.url.drivername == "postgresql+psycopg"

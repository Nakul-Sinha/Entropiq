"""Migrations against a real Postgres (CI only): round trip, drift against the models, CLI, concurrency."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect

from tdp_platform.db.migrate import ALEMBIC_INI, downgrade, upgrade
from tdp_platform.db.models import Base
from tdp_platform.db.session import create_sync_db_engine
from tdp_platform.db.testing import temporary_schema

pytestmark = pytest.mark.integration

TABLES = set(Base.metadata.tables)


def _tables(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def _revision(engine: Engine) -> str | None:
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def test_upgrade_matches_the_models_and_downgrade_removes_everything(pg_url: str) -> None:
    with temporary_schema(pg_url, prefix="tdp_platform_mig", migrate=False) as schema:
        engine = create_sync_db_engine(pg_url, schema=schema, null_pool=True)
        try:
            assert _tables(engine) == set()
            upgrade(pg_url, schema=schema)
            assert _tables(engine) == TABLES | {"alembic_version"}
            assert _revision(engine) == "0001"
            with engine.connect() as conn:
                diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)
            assert diffs == [], diffs
            upgrade(pg_url, schema=schema)  # already at head: nothing to do
            downgrade(pg_url, schema=schema)
            assert _tables(engine) == {"alembic_version"}
            assert _revision(engine) is None
            upgrade(pg_url, schema=schema)
            assert _tables(engine) == TABLES | {"alembic_version"}
        finally:
            engine.dispose()


def _env(pg_url: str, schema: str) -> dict[str, str]:
    return {**os.environ, "DATABASE_URL": pg_url, "DATABASE_SCHEMA": schema}


def _alembic(tmp_path: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_alembic_cli_upgrades_and_downgrades(pg_url: str, tmp_path: Path) -> None:
    with temporary_schema(pg_url, prefix="tdp_platform_cli", migrate=False) as schema:
        env = _env(pg_url, schema)
        up = _alembic(tmp_path, env, "upgrade", "head")
        assert up.returncode == 0, up.stderr
        current = _alembic(tmp_path, env, "current")
        assert current.returncode == 0, current.stderr
        assert "0001 (head)" in current.stdout
        down = _alembic(tmp_path, env, "downgrade", "base")
        assert down.returncode == 0, down.stderr
        current = _alembic(tmp_path, env, "current")
        assert current.returncode == 0, current.stderr
        assert "0001" not in current.stdout


def test_concurrent_processes_migrate_one_after_the_other(pg_url: str, tmp_path: Path) -> None:
    with temporary_schema(pg_url, prefix="tdp_platform_lock", migrate=False) as schema:
        code = "from tdp_platform.db.migrate import upgrade; upgrade()"
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=tmp_path,
                env=_env(pg_url, schema),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        outputs = [proc.communicate(timeout=180) for proc in procs]
        assert [proc.returncode for proc in procs] == [0, 0], outputs
        engine = create_sync_db_engine(pg_url, schema=schema, null_pool=True)
        try:
            assert _tables(engine) == TABLES | {"alembic_version"}
            assert _revision(engine) == "0001"
        finally:
            engine.dispose()


def test_concurrent_threads_are_serialised(pg_url: str) -> None:
    with temporary_schema(pg_url, prefix="tdp_platform_thr", migrate=False) as schema:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(upgrade, pg_url, schema=schema) for _ in range(2)]
            for future in futures:
                future.result(timeout=180)
        engine = create_sync_db_engine(pg_url, schema=schema, null_pool=True)
        try:
            assert _revision(engine) == "0001"
        finally:
            engine.dispose()

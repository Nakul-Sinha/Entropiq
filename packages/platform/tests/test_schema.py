from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast, get_args

import pytest
from sqlalchemy import Connection, make_url
from sqlalchemy.schema import CreateIndex, CreateTable

from tdp_platform.db.migrate import ALEMBIC_INI, alembic_config, head_revision, migration_sql
from tdp_platform.db.models import (
    ADVICE_STATUSES,
    ANALYSIS_STATUSES,
    ARTIFACT_KINDS,
    FEEDBACK_VERDICTS,
    AdviceStatus,
    Analysis,
    AnalysisStatus,
    ArtifactKind,
    Base,
    StoredAdvice,
    Verdict,
)
from tdp_platform.db.testing import truncate_all

DIALECT = make_url("postgresql+psycopg://").get_dialect()()
TS = "TIMESTAMP WITH TIME ZONE"
GOLDEN: dict[str, set[str]] = {
    "repositories": {
        "id UUID DEFAULT gen_random_uuid() NOT NULL",
        "url TEXT NOT NULL",
        "host TEXT NOT NULL",
        "owner TEXT NOT NULL",
        "name TEXT NOT NULL",
        f"created_at {TS} DEFAULT now() NOT NULL",
        "CONSTRAINT repositories_pkey PRIMARY KEY (id)",
        "CONSTRAINT repositories_url_key UNIQUE (url)",
    },
    "model_versions": {
        "version TEXT NOT NULL",
        "metrics JSONB DEFAULT '{}' NOT NULL",
        f"promoted_at {TS} DEFAULT now() NOT NULL",
        "CONSTRAINT model_versions_pkey PRIMARY KEY (version)",
    },
    "analyses": {
        "id UUID DEFAULT gen_random_uuid() NOT NULL",
        "repository_id UUID NOT NULL",
        "commit_sha CHAR(40) NOT NULL",
        "model_version TEXT NOT NULL",
        "budget_hours REAL DEFAULT 40 NOT NULL",
        "status TEXT NOT NULL",
        "stage TEXT",
        "progress SMALLINT DEFAULT 0 NOT NULL",
        "error TEXT",
        "summary JSONB",
        "stats JSONB",
        f"created_at {TS} DEFAULT now() NOT NULL",
        f"started_at {TS}",
        f"finished_at {TS}",
        "CONSTRAINT analyses_pkey PRIMARY KEY (id)",
        "CONSTRAINT analyses_repository_id_commit_sha_model_version_key "
        "UNIQUE (repository_id, commit_sha, model_version)",
        "CONSTRAINT analyses_status_check CHECK (status IN ('queued','running','succeeded','failed'))",
        "CONSTRAINT analyses_repository_id_fkey FOREIGN KEY(repository_id) REFERENCES repositories (id) "
        "ON DELETE CASCADE",
        "CONSTRAINT analyses_model_version_fkey FOREIGN KEY(model_version) "
        "REFERENCES model_versions (version)",
    },
    "file_results": {
        "id BIGSERIAL NOT NULL",
        "analysis_id UUID NOT NULL",
        "path TEXT NOT NULL",
        "blob_sha CHAR(40) NOT NULL",
        "priority SMALLINT NOT NULL",
        "band TEXT NOT NULL",
        "risk REAL NOT NULL",
        "risk_low REAL NOT NULL",
        "risk_high REAL NOT NULL",
        "score REAL NOT NULL",
        "cost_low REAL NOT NULL",
        "cost_high REAL NOT NULL",
        "debt_types TEXT[] DEFAULT '{}' NOT NULL",
        "top_reason TEXT DEFAULT '' NOT NULL",
        "features JSONB NOT NULL",
        "shap JSONB NOT NULL",
        "reasons JSONB NOT NULL",
        "extra JSONB NOT NULL",
        "CONSTRAINT file_results_pkey PRIMARY KEY (id)",
        "CONSTRAINT file_results_analysis_id_path_key UNIQUE (analysis_id, path)",
        "CONSTRAINT file_results_analysis_id_fkey FOREIGN KEY(analysis_id) REFERENCES analyses (id) "
        "ON DELETE CASCADE",
    },
    "analysis_artifacts": {
        "analysis_id UUID NOT NULL",
        "kind TEXT NOT NULL",
        "body JSONB NOT NULL",
        "CONSTRAINT analysis_artifacts_pkey PRIMARY KEY (analysis_id, kind)",
        "CONSTRAINT analysis_artifacts_kind_check CHECK (kind IN ('graph','treemap'))",
        "CONSTRAINT analysis_artifacts_analysis_id_fkey FOREIGN KEY(analysis_id) REFERENCES analyses (id) "
        "ON DELETE CASCADE",
    },
    "plans": {
        "analysis_id UUID NOT NULL",
        "budget_hours REAL NOT NULL",
        "body JSONB NOT NULL",
        f"created_at {TS} DEFAULT now() NOT NULL",
        "CONSTRAINT plans_pkey PRIMARY KEY (analysis_id, budget_hours)",
        "CONSTRAINT plans_analysis_id_fkey FOREIGN KEY(analysis_id) REFERENCES analyses (id) "
        "ON DELETE CASCADE",
    },
    "advice": {
        "cache_key TEXT NOT NULL",
        "file_result_id BIGINT",
        "model TEXT NOT NULL",
        "status TEXT NOT NULL",
        "body JSONB",
        "error TEXT",
        f"created_at {TS} DEFAULT now() NOT NULL",
        "CONSTRAINT advice_pkey PRIMARY KEY (cache_key)",
        "CONSTRAINT advice_status_check CHECK (status IN ('queued','ready','failed'))",
        "CONSTRAINT advice_file_result_id_fkey FOREIGN KEY(file_result_id) REFERENCES file_results (id) "
        "ON DELETE CASCADE",
    },
    "feedback": {
        "id BIGSERIAL NOT NULL",
        "file_result_id BIGINT",
        "verdict TEXT NOT NULL",
        "comment TEXT",
        "client_id TEXT",
        f"created_at {TS} DEFAULT now() NOT NULL",
        "CONSTRAINT feedback_pkey PRIMARY KEY (id)",
        "CONSTRAINT feedback_verdict_check CHECK (verdict IN ('agree','not_a_problem','should_be_higher'))",
        "CONSTRAINT feedback_file_result_id_fkey FOREIGN KEY(file_result_id) REFERENCES file_results (id) "
        "ON DELETE CASCADE",
    },
    "static_cache": {
        "blob_sha CHAR(40) NOT NULL",
        "extractor_version TEXT NOT NULL",
        "syntax JSONB NOT NULL",
        f"created_at {TS} DEFAULT now() NOT NULL",
        "CONSTRAINT static_cache_pkey PRIMARY KEY (blob_sha, extractor_version)",
    },
}
GOLDEN_INDEXES = {
    "CREATE INDEX file_results_rank ON file_results (analysis_id, priority DESC, id)",
    "CREATE INDEX file_results_types ON file_results USING gin (debt_types)",
}


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _split_top_level(body: str) -> set[str]:
    parts: list[str] = []
    depth = 0
    quoted = False
    current: list[str] = []
    for ch in body:
        if ch == "'":
            quoted = not quoted
        elif not quoted and ch == "(":
            depth += 1
        elif not quoted and ch == ")":
            depth -= 1
        elif not quoted and ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return {_collapse(p) for p in parts if p.strip()}


def _table_parts(statement: str) -> tuple[str, set[str]]:
    match = re.match(r"^CREATE TABLE (\w+) \(", statement.strip())
    assert match, statement
    body = statement[statement.index("(") + 1 : statement.rindex(")")]
    return match.group(1), _split_top_level(body)


def _model_ddl() -> tuple[dict[str, set[str]], set[str]]:
    dialect = DIALECT
    tables: dict[str, set[str]] = {}
    indexes: set[str] = set()
    for table in Base.metadata.sorted_tables:
        name, parts = _table_parts(str(CreateTable(table).compile(dialect=dialect)))
        tables[name] = parts
        indexes.update(_collapse(str(CreateIndex(ix).compile(dialect=dialect))) for ix in table.indexes)
    return tables, indexes


def _migration_ddl(sql: str) -> tuple[dict[str, set[str]], set[str]]:
    lines = [line for line in sql.splitlines() if not line.lstrip().startswith("--")]
    tables: dict[str, set[str]] = {}
    indexes: set[str] = set()
    for statement in "\n".join(lines).split(";"):
        text = statement.strip()
        if text.startswith("CREATE TABLE"):
            name, parts = _table_parts(text)
            tables[name] = parts
        elif text.startswith("CREATE INDEX"):
            indexes.add(_collapse(text))
    tables.pop("alembic_version", None)
    return tables, indexes


def test_every_table_is_mapped() -> None:
    assert set(Base.metadata.tables) == set(GOLDEN)


def test_models_match_the_planned_ddl() -> None:
    tables, indexes = _model_ddl()
    for name, expected in GOLDEN.items():
        assert tables[name] == expected, name
    assert indexes == GOLDEN_INDEXES


def test_migration_matches_the_models() -> None:
    model_tables, model_indexes = _model_ddl()
    migration_tables, migration_indexes = _migration_ddl(migration_sql())
    assert migration_tables == model_tables
    assert migration_indexes == model_indexes == GOLDEN_INDEXES


def test_offline_upgrade_sql_is_complete() -> None:
    sql = migration_sql()
    assert sql.lstrip().startswith("BEGIN;")
    assert "INSERT INTO alembic_version (version_num) VALUES ('0001')" in sql
    assert sql.rstrip().endswith("COMMIT;")


def test_offline_downgrade_drops_everything() -> None:
    sql = migration_sql(downgrade=True)
    for name in GOLDEN:
        assert f"DROP TABLE {name};" in sql
    assert "DELETE FROM alembic_version" in sql
    assert (
        sql.index("DROP TABLE feedback;")
        < sql.index("DROP TABLE file_results;")
        < sql.index("DROP TABLE analyses;")
    )


def test_single_head() -> None:
    assert head_revision() == "0001"


def test_alembic_config_does_not_interpolate_urls() -> None:
    cfg = alembic_config(database_url="postgresql://u:p%40ss@h/db")
    assert cfg.attributes["database_url"] == "postgresql://u:p%40ss@h/db"
    assert cfg.attributes["configure_logging"] is False


def test_alembic_config_needs_the_ini_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("tdp_platform.db.migrate.ALEMBIC_INI", tmp_path / "missing.ini")
    with pytest.raises(FileNotFoundError):
        alembic_config()


def test_constants_match_the_api_contract() -> None:
    assert ANALYSIS_STATUSES == get_args(AnalysisStatus) == ("queued", "running", "succeeded", "failed")
    assert ADVICE_STATUSES == get_args(AdviceStatus) == ("queued", "ready", "failed")
    assert FEEDBACK_VERDICTS == get_args(Verdict) == ("agree", "not_a_problem", "should_be_higher")
    assert ARTIFACT_KINDS == get_args(ArtifactKind) == ("graph", "treemap")


def test_nullable_json_columns_store_sql_null() -> None:
    for column in (Analysis.__table__.c.summary, Analysis.__table__.c.stats, StoredAdvice.__table__.c.body):
        assert column.type.none_as_null is True  # type: ignore[attr-defined]


class _RecordingConnection:
    def __init__(self) -> None:
        self.dialect = DIALECT
        self.statements: list[str] = []

    def execute(self, statement: Any) -> None:
        self.statements.append(str(statement))


def test_truncate_all_empties_every_model_table() -> None:
    conn = _RecordingConnection()
    truncate_all(cast(Connection, conn))
    assert conn.statements[0] == "SET LOCAL lock_timeout = '10s'"
    truncate = conn.statements[1]
    assert truncate.startswith("TRUNCATE TABLE ") and truncate.endswith(" RESTART IDENTITY CASCADE")
    named = set(
        truncate.removeprefix("TRUNCATE TABLE ").removesuffix(" RESTART IDENTITY CASCADE").split(", ")
    )
    assert named == set(GOLDEN)


def _alembic(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "DATABASE_SCHEMA")}
    return subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_cli_lists_the_head(tmp_path: Path) -> None:
    proc = _alembic(tmp_path, "heads")
    assert proc.returncode == 0, proc.stderr
    assert "0001 (head)" in proc.stdout


def test_cli_offline_upgrade_runs_env_py(tmp_path: Path) -> None:
    proc = _alembic(tmp_path, "upgrade", "head", "--sql")
    assert proc.returncode == 0, proc.stderr
    assert "CREATE TABLE file_results" in proc.stdout
    assert "COMMIT" in proc.stdout

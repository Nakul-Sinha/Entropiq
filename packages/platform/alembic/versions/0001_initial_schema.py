"""Initial schema: repositories, analyses, file results, artifacts, plans, advice, feedback, parse cache.

Revision ID: 0001
Revises:
Create Date: 2026-10-28

Written by hand to mirror tdp_platform.db.models column by column (migrations never import the models, so
they stay frozen). Constraint names are the ones Postgres would choose itself.
"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _tz() -> postgresql.TIMESTAMP:
    return postgresql.TIMESTAMP(timezone=True)


def _uuid() -> postgresql.UUID[uuid.UUID]:
    return postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    """Create every table and index."""
    op.create_table(
        "repositories",
        sa.Column("id", _uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("host", sa.Text(), nullable=False),
        sa.Column("owner", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("created_at", _tz(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("repositories_pkey")),
        sa.UniqueConstraint("url", name=op.f("repositories_url_key")),
    )
    op.create_table(
        "model_versions",
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("promoted_at", _tz(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("version", name=op.f("model_versions_pkey")),
    )
    op.create_table(
        "analyses",
        sa.Column("id", _uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("repository_id", _uuid(), nullable=False),
        sa.Column("commit_sha", sa.CHAR(40), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("budget_hours", sa.REAL(), server_default=sa.text("40"), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=True),
        sa.Column("progress", sa.SmallInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("summary", postgresql.JSONB(), nullable=True),
        sa.Column("stats", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", _tz(), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", _tz(), nullable=True),
        sa.Column("finished_at", _tz(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("analyses_pkey")),
        sa.UniqueConstraint(
            "repository_id",
            "commit_sha",
            "model_version",
            name=op.f("analyses_repository_id_commit_sha_model_version_key"),
        ),
        sa.CheckConstraint(
            "status IN ('queued','running','succeeded','failed')", name=op.f("analyses_status_check")
        ),
        sa.ForeignKeyConstraint(
            ["repository_id"],
            ["repositories.id"],
            name=op.f("analyses_repository_id_fkey"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["model_version"], ["model_versions.version"], name=op.f("analyses_model_version_fkey")
        ),
    )
    op.create_table(
        "file_results",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("analysis_id", _uuid(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("blob_sha", sa.CHAR(40), nullable=False),
        sa.Column("priority", sa.SmallInteger(), nullable=False),
        sa.Column("band", sa.Text(), nullable=False),
        sa.Column("risk", sa.REAL(), nullable=False),
        sa.Column("risk_low", sa.REAL(), nullable=False),
        sa.Column("risk_high", sa.REAL(), nullable=False),
        sa.Column("score", sa.REAL(), nullable=False),
        sa.Column("cost_low", sa.REAL(), nullable=False),
        sa.Column("cost_high", sa.REAL(), nullable=False),
        sa.Column("debt_types", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("top_reason", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.Column("shap", postgresql.JSONB(), nullable=False),
        sa.Column("reasons", postgresql.JSONB(), nullable=False),
        sa.Column("extra", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("file_results_pkey")),
        sa.UniqueConstraint("analysis_id", "path", name=op.f("file_results_analysis_id_path_key")),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["analyses.id"],
            name=op.f("file_results_analysis_id_fkey"),
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        op.f("file_results_rank"), "file_results", ["analysis_id", sa.text("priority DESC"), "id"]
    )
    op.create_index(op.f("file_results_types"), "file_results", ["debt_types"], postgresql_using="gin")
    op.create_table(
        "analysis_artifacts",
        sa.Column("analysis_id", _uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("body", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("analysis_id", "kind", name=op.f("analysis_artifacts_pkey")),
        sa.CheckConstraint("kind IN ('graph','treemap')", name=op.f("analysis_artifacts_kind_check")),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["analyses.id"],
            name=op.f("analysis_artifacts_analysis_id_fkey"),
            ondelete="CASCADE",
        ),
    )
    op.create_table(
        "plans",
        sa.Column("analysis_id", _uuid(), nullable=False),
        sa.Column("budget_hours", sa.REAL(), nullable=False),
        sa.Column("body", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", _tz(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("analysis_id", "budget_hours", name=op.f("plans_pkey")),
        sa.ForeignKeyConstraint(
            ["analysis_id"], ["analyses.id"], name=op.f("plans_analysis_id_fkey"), ondelete="CASCADE"
        ),
    )
    op.create_table(
        "advice",
        sa.Column("cache_key", sa.Text(), nullable=False),
        sa.Column("file_result_id", sa.BigInteger(), nullable=True),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("body", postgresql.JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", _tz(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("cache_key", name=op.f("advice_pkey")),
        sa.CheckConstraint("status IN ('queued','ready','failed')", name=op.f("advice_status_check")),
        sa.ForeignKeyConstraint(
            ["file_result_id"],
            ["file_results.id"],
            name=op.f("advice_file_result_id_fkey"),
            ondelete="CASCADE",
        ),
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("file_result_id", sa.BigInteger(), nullable=True),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("client_id", sa.Text(), nullable=True),
        sa.Column("created_at", _tz(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("feedback_pkey")),
        sa.CheckConstraint(
            "verdict IN ('agree','not_a_problem','should_be_higher')", name=op.f("feedback_verdict_check")
        ),
        sa.ForeignKeyConstraint(
            ["file_result_id"],
            ["file_results.id"],
            name=op.f("feedback_file_result_id_fkey"),
            ondelete="CASCADE",
        ),
    )
    op.create_table(
        "static_cache",
        sa.Column("blob_sha", sa.CHAR(40), nullable=False),
        sa.Column("extractor_version", sa.Text(), nullable=False),
        sa.Column("syntax", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", _tz(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("blob_sha", "extractor_version", name=op.f("static_cache_pkey")),
    )


def downgrade() -> None:
    """Drop every table (dependents first)."""
    for table in (
        "feedback",
        "advice",
        "plans",
        "analysis_artifacts",
        "file_results",
        "analyses",
        "model_versions",
        "repositories",
        "static_cache",
    ):
        op.drop_table(table)

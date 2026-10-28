"""SQLAlchemy 2 models of the platform database (one class per table).

The schema follows the DDL of the implementation plan exactly. Constraint names follow Postgres's own
defaults (``<table>_pkey``, ``<table>_<columns>_key``, ``<table>_<columns>_fkey``, ``<table>_<name>_check``)
through a naming convention, so the models, the Alembic migrations and the plan's raw DDL all produce the
same schema and later migrations can address constraints by name.

Units and conventions: timestamps are ``TIMESTAMPTZ`` (timezone aware UTC datetimes in Python), budgets and
costs are hours, ``priority`` and ``progress`` are 0..100, ``risk`` values are probabilities 0..1. JSON
shapes of ``file_results`` are defined in :mod:`tdp_platform.db.rows`.

This module deliberately imports nothing from ``tdp_core``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Final, Literal

from sqlalchemy import (
    CHAR,
    REAL,
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    MetaData,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION: Final[dict[str, str]] = {
    "pk": "%(table_name)s_pkey",
    "uq": "%(table_name)s_%(column_0_N_name)s_key",
    "fk": "%(table_name)s_%(column_0_N_name)s_fkey",
    "ck": "%(table_name)s_%(constraint_name)s_check",
    "ix": "%(table_name)s_%(column_0_N_name)s_idx",
}

AnalysisStatus = Literal["queued", "running", "succeeded", "failed"]
ANALYSIS_STATUSES: Final[tuple[AnalysisStatus, ...]] = ("queued", "running", "succeeded", "failed")
AdviceStatus = Literal["queued", "ready", "failed"]
ADVICE_STATUSES: Final[tuple[AdviceStatus, ...]] = ("queued", "ready", "failed")
Verdict = Literal["agree", "not_a_problem", "should_be_higher"]
FEEDBACK_VERDICTS: Final[tuple[Verdict, ...]] = ("agree", "not_a_problem", "should_be_higher")
ArtifactKind = Literal["graph", "treemap"]
ARTIFACT_KINDS: Final[tuple[ArtifactKind, ...]] = ("graph", "treemap")


def _in_check(column: str, values: tuple[str, ...]) -> str:
    """CHECK text ``column IN ('a','b')`` written exactly like the plan's DDL."""
    return f"{column} IN ({','.join(repr(v) for v in values)})"


def _tz() -> TIMESTAMP:
    return TIMESTAMP(timezone=True)


class Base(DeclarativeBase):
    """Declarative base whose metadata names constraints like Postgres does."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Repository(Base):
    """A git repository, identified by its canonical clone URL."""

    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    url: Mapped[str] = mapped_column(Text, unique=True)
    host: Mapped[str] = mapped_column(Text)
    owner: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(_tz(), server_default=func.now())


class ModelVersion(Base):
    """A model bundle version that analyses were scored with; ``metrics`` comes from its metadata."""

    __tablename__ = "model_versions"

    version: Mapped[str] = mapped_column(Text, primary_key=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'"))
    promoted_at: Mapped[datetime] = mapped_column(_tz(), server_default=func.now())


class Analysis(Base):
    """One analysis of a repository at a commit with a model version (unique per triple).

    ``status`` moves queued -> running -> succeeded or failed; ``progress`` is 0..100 and ``stage`` the
    current stage name. ``budget_hours`` is the plan budget in hours. ``summary`` and ``stats`` are JSON
    objects set on success. ``repository`` is never lazy loaded: repositories load it eagerly.
    """

    __tablename__ = "analyses"
    __table_args__ = (
        UniqueConstraint("repository_id", "commit_sha", "model_version"),
        CheckConstraint(_in_check("status", ANALYSIS_STATUSES), name="status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE")
    )
    commit_sha: Mapped[str] = mapped_column(CHAR(40))
    model_version: Mapped[str] = mapped_column(Text, ForeignKey("model_versions.version"))
    budget_hours: Mapped[float] = mapped_column(REAL, server_default=text("40"))
    status: Mapped[AnalysisStatus] = mapped_column(Text)
    stage: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[int] = mapped_column(SmallInteger, server_default=text("0"))
    error: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(_tz(), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(_tz())
    finished_at: Mapped[datetime | None] = mapped_column(_tz())

    repository: Mapped[Repository] = relationship(lazy="raise")


class FileResult(Base):
    """One ranked file of an analysis.

    ``priority`` 0..100, ``band`` Critical/High/Medium/Low, ``risk*`` probabilities, ``score`` the raw ranker
    margin, ``cost_low``/``cost_high`` fix cost in hours. JSON columns: ``features`` (feature -> value),
    ``shap`` (``by_family``, ``by_feature``), ``reasons`` (``reasons``, ``mitigations``) and ``extra`` (a
    serialised ``FileExtra``); see :mod:`tdp_platform.db.rows`.
    """

    __tablename__ = "file_results"
    __table_args__ = (UniqueConstraint("analysis_id", "path"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE")
    )
    path: Mapped[str] = mapped_column(Text)
    blob_sha: Mapped[str] = mapped_column(CHAR(40))
    priority: Mapped[int] = mapped_column(SmallInteger)
    band: Mapped[str] = mapped_column(Text)
    risk: Mapped[float] = mapped_column(REAL)
    risk_low: Mapped[float] = mapped_column(REAL)
    risk_high: Mapped[float] = mapped_column(REAL)
    score: Mapped[float] = mapped_column(REAL)
    cost_low: Mapped[float] = mapped_column(REAL)
    cost_high: Mapped[float] = mapped_column(REAL)
    debt_types: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    top_reason: Mapped[str] = mapped_column(Text, server_default=text("''"))
    features: Mapped[dict[str, Any]] = mapped_column(JSONB)
    shap: Mapped[dict[str, Any]] = mapped_column(JSONB)
    reasons: Mapped[dict[str, Any]] = mapped_column(JSONB)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB)


Index("file_results_rank", FileResult.analysis_id, FileResult.priority.desc(), FileResult.id)
Index("file_results_types", FileResult.debt_types, postgresql_using="gin")


class AnalysisArtifact(Base):
    """A JSON artifact of an analysis: the dependency ``graph`` or the ``treemap``."""

    __tablename__ = "analysis_artifacts"
    __table_args__ = (CheckConstraint(_in_check("kind", ARTIFACT_KINDS), name="kind"),)

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), primary_key=True
    )
    kind: Mapped[ArtifactKind] = mapped_column(Text, primary_key=True)
    body: Mapped[dict[str, Any]] = mapped_column(JSONB)


class StoredPlan(Base):
    """A fix plan of an analysis for one budget in hours (stored as REAL)."""

    __tablename__ = "plans"

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), primary_key=True
    )
    budget_hours: Mapped[float] = mapped_column(REAL, primary_key=True)
    body: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(_tz(), server_default=func.now())


class StoredAdvice(Base):
    """LLM advice for a file, keyed by a cache key; ``status`` is queued, ready or failed."""

    __tablename__ = "advice"
    __table_args__ = (CheckConstraint(_in_check("status", ADVICE_STATUSES), name="status"),)

    cache_key: Mapped[str] = mapped_column(Text, primary_key=True)
    file_result_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("file_results.id", ondelete="CASCADE")
    )
    model: Mapped[str] = mapped_column(Text)
    status: Mapped[AdviceStatus] = mapped_column(Text)
    body: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(_tz(), server_default=func.now())


class Feedback(Base):
    """A user's verdict on a ranked file."""

    __tablename__ = "feedback"
    __table_args__ = (CheckConstraint(_in_check("verdict", FEEDBACK_VERDICTS), name="verdict"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    file_result_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("file_results.id", ondelete="CASCADE")
    )
    verdict: Mapped[Verdict] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)
    client_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(_tz(), server_default=func.now())


class StaticCacheEntry(Base):
    """A parsed ``FileSyntax`` (JSON) keyed by git blob sha and extractor version."""

    __tablename__ = "static_cache"

    blob_sha: Mapped[str] = mapped_column(CHAR(40), primary_key=True)
    extractor_version: Mapped[str] = mapped_column(Text, primary_key=True)
    syntax: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(_tz(), server_default=func.now())

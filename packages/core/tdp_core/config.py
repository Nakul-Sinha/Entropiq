"""Global settings shared by every stage.

Values are the defaults agreed in IMPLEMENTATION_PLAN.md section 3.1.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Settings:
    horizon_days: int = 365  # label window H
    snapshot_step_days: int = 182  # training snapshots every ~6 months
    min_history_days: int = 365  # first snapshot needs this much history
    history_window_days: int | None = 1095  # inference: mine last 3 years; training passes None
    max_commits: int = 20_000  # inference cap
    half_life_days: float = 90.0  # recency decay for churn/commit counts
    recent_days: tuple[int, int] = (90, 180)
    bulk_commit_files: int = 50  # > this many files -> bulk commit
    minor_author_share: float = 0.05
    owner_active_days: int = 90
    new_file_days: int = 90
    cochange_min_support: int = 5
    cochange_min_confidence: float = 0.5
    clone_k: int = 50  # tokens per k-gram
    clone_window: int = 10  # winnowing window
    gamma: float = 0.25  # churn weight in pain
    max_file_bytes: int = 1_000_000
    parse_timeout_s: float = 5.0
    analysis_timeout_s: float = 900.0
    inference_szz_max_fixes: int = 300
    seed: int = 42
    default_budget_hours: float = 40.0
    min_plan_priority: int = 50
    advisor_model: str = "claude-sonnet-5-5"
    advisor_max_excerpt_lines: int = 300


DEFAULT_SETTINGS = Settings()


def with_overrides(s: Settings, **kw: object) -> Settings:
    """Return a copy of ``s`` with the given fields replaced (unknown names raise ``TypeError``)."""
    return replace(s, **kw)  # type: ignore[arg-type]

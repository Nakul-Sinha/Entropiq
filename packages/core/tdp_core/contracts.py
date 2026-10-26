"""Shared data contracts (IMPLEMENTATION_PLAN.md sections 3.2 and 3.3).

Every step imports these types. Changing anything here requires updating the plan in the same PR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd
import pyarrow as pa

EXTRACTOR_VERSION = "1"  # bump whenever any static metric or FileSyntax field changes


# ---------- parsing ----------
@dataclass(frozen=True, slots=True)
class FunctionInfo:
    name: str  # qualified: "func", "Class.method", "outer.inner"
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    cyclomatic: int  # >= 1
    cognitive: int  # >= 0
    max_nesting: int  # >= 0
    params: int  # excluding self/cls


@dataclass(frozen=True, slots=True)
class CommentInfo:
    line: int  # 1-based
    text: str  # comment text without the leading '#', stripped


@dataclass(frozen=True, slots=True)
class RawImport:
    module: str  # "a.b" for `import a.b` / `from a.b import x`; "" for `from . import x`
    names: tuple[str, ...]  # imported names for `from` imports, () otherwise
    level: int  # 0 absolute, 1 for ".", 2 for "..", ...
    line: int


@dataclass(frozen=True, slots=True)
class FileSyntax:
    path: str
    language: str  # "python"
    sloc: int
    module_cc: int  # 1 + decision points in module-level code outside functions and classes
    functions: tuple[FunctionInfo, ...]
    classes: int
    comments: tuple[CommentInfo, ...]
    imports: tuple[RawImport, ...]
    tokens: tuple[str, ...]  # normalised: identifiers -> "ID", literals -> "LIT", others -> node type
    token_lines: tuple[int, ...]  # same length as tokens: 1-based line of each token
    parse_error: bool


@dataclass(frozen=True, slots=True)
class CloneStats:
    dup_ratio: float  # duplicated tokens / total tokens, 0..1
    dup_tokens: int
    clone_groups: int  # number of distinct fingerprints shared with another location
    dup_lines: int  # distinct lines covered by duplicated tokens


# ---------- time ----------
@dataclass(frozen=True, slots=True)
class Snapshot:
    repo: str  # slug, e.g. "psf__requests"
    index: int  # 0-based snapshot number
    time: datetime  # T (UTC)
    sha: str  # last non-merge commit with author_time <= T on the analysed branch


# ---------- features / labels ----------
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "S": (
        "sloc",
        "cc_sum",
        "cc_mean",
        "cc_max",
        "cog_max",
        "nesting_max",
        "n_functions",
        "n_classes",
        "longest_function",
        "params_max",
        "dup_ratio",
    ),
    "E": (
        "commits_total",
        "commits_90d",
        "commits_180d",
        "commits_decayed",
        "churn_total",
        "churn_decayed",
        "rel_churn_decayed",
        "age_days",
        "staleness_days",
        "cc_trend",
        "burstiness",
        "is_new",
    ),
    "O": ("n_authors", "owner_share", "minor_authors", "author_entropy", "owner_active"),
    "D": ("prior_bugfix_commits", "prior_latent_bugs"),
    "I": (
        "fan_in",
        "fan_out",
        "transitive_dependents",
        "pagerank",
        "betweenness",
        "scc_size",
        "cochange_partners",
        "cochange_max_conf",
        "hidden_coupling",
    ),
    "T": ("has_test_file", "test_importers", "test_cochange_rate", "pkg_test_ratio"),
    "Q": ("satd_count", "satd_density", "satd_age_days"),
}
FAMILY_ORDER: tuple[str, ...] = ("S", "E", "O", "D", "I", "T", "Q")
FAMILY_NAMES: dict[str, str] = {
    "S": "Static structure",
    "E": "Evolution",
    "O": "Ownership",
    "D": "Defect history",
    "I": "Impact",
    "T": "Tests",
    "Q": "Self-admitted debt",
}
FEATURE_COLUMNS: tuple[str, ...] = tuple(f for fam in FAMILY_ORDER for f in FEATURE_FAMILIES[fam])
BINARY_FEATURES: frozenset[str] = frozenset({"is_new", "owner_active", "has_test_file"})
PCT_COLUMNS: tuple[str, ...] = tuple(f"pct_{f}" for f in FEATURE_COLUMNS if f not in BINARY_FEATURES)
FEATURE_INPUT_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + PCT_COLUMNS  # model input order
INDEX_COLUMNS: tuple[str, ...] = ("repo", "snapshot_idx", "snapshot_time", "path")
LABEL_COLUMNS: tuple[str, ...] = ("latent_defects", "future_churn", "pain", "grade", "has_latent_defect")


def family_of(column: str) -> str:
    """Family code for a feature or pct_ column. Raises KeyError for unknown columns."""
    base = column[4:] if column.startswith("pct_") else column
    for fam, cols in FEATURE_FAMILIES.items():
        if base in cols:
            return fam
    raise KeyError(column)


# ---------- scoring ----------
Band = Literal["Critical", "High", "Medium", "Low"]
BANDS: tuple[tuple[Band, int], ...] = (("Critical", 90), ("High", 75), ("Medium", 50), ("Low", 0))


def band_for(priority: int) -> Band:
    """Band for a 0..100 priority: Critical >= 90, High >= 75, Medium >= 50, else Low."""
    for name, lo in BANDS:
        if priority >= lo:
            return name
    return "Low"


DEBT_TYPES: tuple[str, ...] = (
    "complexity",
    "duplication",
    "test",
    "architecture",
    "knowledge",
    "admitted",
    "defect_hotspot",
)
WHATIF_ACTIONS: tuple[str, ...] = (
    "add_tests",
    "split_longest_function",
    "remove_duplication",
    "resolve_satd",
    "break_cycle",
)


@dataclass(frozen=True, slots=True)
class Reason:
    family: str  # S/E/O/D/I/T/Q
    feature: str  # base feature name (never pct_)
    value: float  # raw value of the feature
    percentile: float  # 0..100 within the repo
    shap: float  # contribution (family total for the reason)
    text: str  # rendered sentence


@dataclass(frozen=True, slots=True)
class FileScore:
    path: str
    score: float  # raw ranker margin
    priority: int  # 0..100
    band: Band
    risk: float  # calibrated probability 0..1
    risk_low: float
    risk_high: float
    reasons: tuple[Reason, ...]
    mitigations: tuple[Reason, ...]
    debt_types: tuple[str, ...]  # subset of DEBT_TYPES, max 2
    shap_by_family: dict[str, float]  # keys: FAMILY_ORDER


@dataclass(frozen=True, slots=True)
class CostEstimate:
    hours: float
    low: float  # 0.8 * hours
    high: float  # 1.2 * hours
    breakdown: dict[str, float]  # minutes per issue kind


@dataclass(frozen=True, slots=True)
class PlanStep:
    order: int  # 1-based
    path: str
    benefit: float
    cost: CostEstimate
    roi: float  # benefit / cost.hours
    group: int  # SCC group id; files fixed together share it
    reason: str


@dataclass(frozen=True, slots=True)
class Plan:
    budget_hours: float
    steps: tuple[PlanStep, ...]
    total_benefit: float
    total_hours: float


# ---------- analysis output ----------
@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str  # path
    priority: int
    band: Band
    sloc: int
    dependents: int  # transitive dependents


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str  # importer (or co-change file a)
    target: str  # imported (or co-change file b)
    kind: Literal["import", "hidden_coupling"]
    weight: float  # 1.0 for imports, confidence for coupling


@dataclass(frozen=True, slots=True)
class GraphData:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


@dataclass(frozen=True, slots=True)
class FileExtra:
    blob_sha: str
    functions: tuple[FunctionInfo, ...]
    dependents: tuple[str, ...]  # direct importers (non-test)
    dependencies: tuple[str, ...]  # direct imports (in-repo)
    test_importers: tuple[str, ...]
    commit_activity: tuple[tuple[str, int], ...]  # ("YYYY-MM", commits) for the last 24 months
    cost: CostEstimate


@dataclass(frozen=True)
class AnalysisResult:
    sha: str
    analyzed_at: datetime
    model_version: str
    extractor_version: str
    files: tuple[FileScore, ...]  # sorted by priority desc, then path
    features: pd.DataFrame  # index = path, columns = FEATURE_INPUT_COLUMNS
    extras: dict[str, FileExtra]
    graph: GraphData
    plan: Plan
    stats: dict[str, Any] = field(default_factory=dict)  # n_files, skipped, timings_s per stage


# ---------- Arrow / Parquet schemas (section 3.3) ----------
COMMITS_SCHEMA = pa.schema(
    [
        ("sha", pa.string()),
        ("author_name", pa.string()),  # mailmap applied
        ("author_email", pa.string()),  # mailmap applied, lowercased
        ("author_time", pa.timestamp("s", tz="UTC")),
        ("parents", pa.list_(pa.string())),
        ("subject", pa.string()),
        ("message", pa.string()),  # full message (subject + body)
        ("files_touched", pa.int32()),
        ("is_bulk", pa.bool_()),
    ]
)  # merge commits are NOT included

FILE_CHANGES_SCHEMA = pa.schema(
    [
        ("sha", pa.string()),
        ("path", pa.string()),
        ("old_path", pa.string()),  # null unless renamed
        ("added", pa.int32()),  # binary: 0
        ("deleted", pa.int32()),  # binary: 0
        ("is_binary", pa.bool_()),
    ]
)

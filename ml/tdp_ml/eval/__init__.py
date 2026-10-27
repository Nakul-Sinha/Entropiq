"""Evaluation harness: metrics, leakage-safe splits, protocol runs, statistics and result tables.

Typical use (baselines, the ranker and ablations all follow it)::

    from tdp_ml.eval import run_protocol, summarize, summary_markdown

    run = run_protocol(df, "lopo", lambda train, test: test["pct_churn_decayed"] * test["pct_cc_sum"],
                       model="hotspot")
    print(summary_markdown(summarize(run.per_query)))

Modules: :mod:`~tdp_ml.eval.metrics` (per-query metrics and :func:`evaluate`), :mod:`~tdp_ml.eval.splits`
(temporal, forward-chaining and LOPO splits), :mod:`~tdp_ml.eval.harness` (protocol runner),
:mod:`~tdp_ml.eval.stats` (bootstrap summaries, Wilcoxon with Holm, Cliff's delta),
:mod:`~tdp_ml.eval.report` (per-query files and Markdown tables) and :mod:`~tdp_ml.eval.synthetic`
(deterministic test data in the dataset layout).
"""

from __future__ import annotations

from tdp_ml.eval.harness import (
    PROTOCOLS,
    FitPredict,
    ProtocolName,
    ProtocolRun,
    Split,
    protocol_splits,
    run_protocol,
)
from tdp_ml.eval.metrics import (
    ECE_BINS,
    EVAL_INPUT_COLUMNS,
    GROUP_COLUMNS,
    HIGHER_IS_BETTER,
    LOC_FRACTION,
    METRIC_LABELS,
    METRICS,
    MIN_GRADE,
    NDCG_KS,
    PRECISION_K,
    PROBABILITY_METRICS,
    QUERY_COLUMNS,
    RANKING_METRICS,
    RUN_COLUMNS,
    FloatArray,
    IntArray,
    auc_pr,
    brier,
    defect_density,
    ece,
    effort_curve,
    evaluate,
    ndcg_at_k,
    popt,
    precision_at_k,
    rank_order,
    recall_at_loc,
    reliability_table,
    spearman,
)
from tdp_ml.eval.report import (
    comparison_markdown,
    load_per_query,
    per_query_path,
    save_per_query,
    summary_markdown,
)
from tdp_ml.eval.splits import (
    forward_folds,
    lopo_splits,
    temporal_split,
)
from tdp_ml.eval.stats import (
    CLIFFS_THRESHOLDS,
    SUMMARY_COLUMNS,
    TEST_COLUMNS,
    Alternative,
    cliffs_delta,
    cliffs_magnitude,
    holm,
    summarize,
    wilcoxon_holm,
)
from tdp_ml.eval.synthetic import (
    SYNTHETIC_COLUMNS,
    SYNTHETIC_START,
    make_synthetic_dataset,
)

__all__ = [
    "CLIFFS_THRESHOLDS",
    "ECE_BINS",
    "EVAL_INPUT_COLUMNS",
    "GROUP_COLUMNS",
    "HIGHER_IS_BETTER",
    "LOC_FRACTION",
    "METRICS",
    "METRIC_LABELS",
    "MIN_GRADE",
    "NDCG_KS",
    "PRECISION_K",
    "PROBABILITY_METRICS",
    "PROTOCOLS",
    "QUERY_COLUMNS",
    "RANKING_METRICS",
    "RUN_COLUMNS",
    "SUMMARY_COLUMNS",
    "SYNTHETIC_COLUMNS",
    "SYNTHETIC_START",
    "TEST_COLUMNS",
    "Alternative",
    "FitPredict",
    "FloatArray",
    "IntArray",
    "ProtocolName",
    "ProtocolRun",
    "Split",
    "auc_pr",
    "brier",
    "cliffs_delta",
    "cliffs_magnitude",
    "comparison_markdown",
    "defect_density",
    "ece",
    "effort_curve",
    "evaluate",
    "forward_folds",
    "holm",
    "load_per_query",
    "lopo_splits",
    "make_synthetic_dataset",
    "ndcg_at_k",
    "per_query_path",
    "popt",
    "precision_at_k",
    "protocol_splits",
    "rank_order",
    "recall_at_loc",
    "reliability_table",
    "run_protocol",
    "save_per_query",
    "spearman",
    "summarize",
    "summary_markdown",
    "temporal_split",
    "wilcoxon_holm",
]

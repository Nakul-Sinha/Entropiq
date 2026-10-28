/** Domain constants mirrored from tdp_core.contracts and the analysis pipeline, with UI labels. */
import type { FeedbackVerdict, WhatIfAction } from "@/lib/api/types";

export const DEBT_TYPES = [
  "complexity",
  "duplication",
  "test",
  "architecture",
  "knowledge",
  "admitted",
  "defect_hotspot",
] as const;
export type DebtType = (typeof DEBT_TYPES)[number];

export const DEBT_TYPE_LABELS: Record<DebtType, string> = {
  complexity: "Complexity",
  duplication: "Duplication",
  test: "Missing tests",
  architecture: "Architecture",
  knowledge: "Knowledge concentration",
  admitted: "Self-admitted debt",
  defect_hotspot: "Defect hotspot",
};

/** The label of a debt type; unknown types are shown as they are. */
export function debtTypeLabel(type: string): string {
  return (DEBT_TYPE_LABELS as Record<string, string>)[type] ?? type;
}

export const WHATIF_ACTIONS = [
  "add_tests",
  "split_longest_function",
  "remove_duplication",
  "resolve_satd",
  "break_cycle",
] as const satisfies readonly WhatIfAction[];

export const WHATIF_ACTION_LABELS: Record<WhatIfAction, string> = {
  add_tests: "Add tests",
  split_longest_function: "Split the longest function",
  remove_duplication: "Remove duplication",
  resolve_satd: "Resolve TODO and FIXME comments",
  break_cycle: "Break the dependency cycle",
};

export const FAMILY_ORDER = ["S", "E", "O", "D", "I", "T", "Q"] as const;
export type Family = (typeof FAMILY_ORDER)[number];

export const FAMILY_NAMES: Record<Family, string> = {
  S: "Static structure",
  E: "Evolution",
  O: "Ownership",
  D: "Defect history",
  I: "Impact",
  T: "Tests",
  Q: "Self-admitted debt",
};

export const FEATURE_FAMILIES = {
  S: [
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
  ],
  E: [
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
  ],
  O: ["n_authors", "owner_share", "minor_authors", "author_entropy", "owner_active"],
  D: ["prior_bugfix_commits", "prior_latent_bugs"],
  I: [
    "fan_in",
    "fan_out",
    "transitive_dependents",
    "pagerank",
    "betweenness",
    "scc_size",
    "cochange_partners",
    "cochange_max_conf",
    "hidden_coupling",
  ],
  T: ["has_test_file", "test_importers", "test_cochange_rate", "pkg_test_ratio"],
  Q: ["satd_count", "satd_density", "satd_age_days"],
} as const satisfies Record<Family, readonly string[]>;

/** The 46 raw feature columns in model order (families in FAMILY_ORDER). */
export const FEATURE_COLUMNS: readonly string[] = FAMILY_ORDER.flatMap((f) => FEATURE_FAMILIES[f]);

/** Features that are 0 or 1 and have no pct_ copy. */
export const BINARY_FEATURES: ReadonlySet<string> = new Set([
  "is_new",
  "owner_active",
  "has_test_file",
]);

/** Pipeline stages and their share of the whole analysis (percent when the stage finishes). */
export const ANALYSIS_STAGES: readonly { stage: string; pct: number; label: string }[] = [
  { stage: "mine", pct: 25, label: "Mining history" },
  { stage: "parse", pct: 50, label: "Parsing files" },
  { stage: "graphs", pct: 60, label: "Building graphs" },
  { stage: "features", pct: 70, label: "Computing features" },
  { stage: "score", pct: 80, label: "Scoring files" },
  { stage: "explain", pct: 90, label: "Explaining scores" },
  { stage: "plan", pct: 95, label: "Planning fixes" },
];

export const FEEDBACK_VERDICTS: readonly { value: FeedbackVerdict; label: string }[] = [
  { value: "agree", label: "Agree" },
  { value: "not_a_problem", label: "Not a problem" },
  { value: "should_be_higher", label: "Should be higher" },
];

/** Shown next to every piece of LLM advice. */
export const ADVICE_DISCLAIMER = "AI-generated advice, review before use";

/** Range and default of the plan page's budget slider, in hours. */
export const PLAN_BUDGET = { min: 1, max: 400, default: 40 } as const;

/** Budgets the API accepts, in hours. */
export const API_BUDGET_RANGE = { min: 0.5, max: 2000 } as const;

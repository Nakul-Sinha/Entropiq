/**
 * Deterministic fixture data for the mock backend: one analysed repository, "inventory-service",
 * with 40 Python files. Everything is built once at module load from the tables below and a seeded
 * PRNG (mulberry32(42)); nothing depends on the clock except the simulated running analyses.
 */
import type {
  Advice,
  Analysis,
  AnalysisSummary,
  FileDetail,
  FileSummary,
  GraphOut,
  PlanOut,
  PlanStepOut,
  Reason,
  TreemapNode,
  TrendOut,
  WhatIfAction,
  WhatIfOut,
} from "@/lib/api/types";
import { bandFor } from "@/lib/bands";
import {
  ANALYSIS_STAGES,
  BINARY_FEATURES,
  DEBT_TYPES,
  FAMILY_ORDER,
  FEATURE_COLUMNS,
  type DebtType,
  type Family,
} from "@/lib/domain";

import {
  MOCK_ANALYSIS_ID,
  MOCK_COMMIT_SHA,
  MOCK_FAILED_ANALYSIS_ID,
  MOCK_FIRST_FILE_ID,
  MOCK_MODEL_VERSION,
  MOCK_REPO_URL,
  MOCK_REPOSITORY_ID,
} from "./ids";
import { mockRunningAnalysisId, runningStartedAt, runningState } from "./progress";
import { mockState } from "./state";

export * from "./ids";

// ---------- helpers ----------

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(42);
const roundTo = (value: number, digits: number) => {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
};
const round1 = (v: number) => roundTo(v, 1);
const round2 = (v: number) => roundTo(v, 2);
const round3 = (v: number) => roundTo(v, 3);
const round4 = (v: number) => roundTo(v, 4);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
const plural = (n: number) => (n === 1 ? "" : "s");

// ---------- the repository ----------

const ROOT = "src/inventory/";
const N_FILES = 40;
const mod = (name: string) => `${ROOT}${name}.py`;

/** Rank order (highest priority first): module path under src/inventory/, SLOC, debt types. */
const TABLE: readonly (readonly [string, number, readonly DebtType[]])[] = [
  ["core/pricing", 1240, ["complexity", "defect_hotspot"]],
  ["sync/legacy_adapter", 980, ["duplication", "test"]],
  ["core/allocation", 860, ["complexity", "test"]],
  ["api/routes", 910, ["architecture", "complexity"]],
  ["db/queries", 720, ["architecture"]],
  ["core/reservations", 640, ["architecture"]],
  ["sync/importer", 590, ["duplication", "admitted"]],
  ["core/stock", 610, ["architecture"]],
  ["reports/daily", 480, ["knowledge", "test"]],
  ["core/rules", 530, ["admitted", "complexity"]],
  ["db/models", 450, ["architecture"]],
  ["core/forecast", 410, ["test", "knowledge"]],
  ["notify/webhooks", 300, ["test"]],
  ["api/auth", 280, ["knowledge"]],
  ["sync/scheduler", 260, ["test"]],
  ["reports/export_pdf", 340, ["duplication"]],
  ["api/schemas", 390, []],
  ["sync/exporter", 310, ["duplication"]],
  ["db/migrations", 220, ["admitted"]],
  ["app", 180, []],
  ["config", 160, []],
  ["sync/csv_reader", 200, []],
  ["notify/email", 150, []],
  ["reports/charts", 240, []],
  ["utils/money", 120, []],
  ["utils/retry", 90, []],
  ["cli", 210, ["test"]],
  ["plugins", 110, []],
  ["api/errors", 80, []],
  ["db/session", 70, []],
  ["utils/dates", 100, []],
  ["utils/text", 85, []],
  ["__init__", 12, []],
  ["api/__init__", 6, []],
  ["core/__init__", 8, []],
  ["db/__init__", 5, []],
  ["sync/__init__", 4, []],
  ["notify/__init__", 3, []],
  ["reports/__init__", 3, []],
  ["utils/__init__", 2, []],
];

/** importer -> imported modules (68 edges). core/stock and core/reservations form the only cycle. */
const IMPORTS: Readonly<Record<string, readonly string[]>> = {
  app: ["config", "api/routes", "db/session", "plugins"],
  cli: ["app", "sync/importer", "sync/exporter", "reports/daily", "config"],
  "api/routes": [
    "api/schemas",
    "api/auth",
    "api/errors",
    "core/pricing",
    "core/stock",
    "core/reservations",
    "db/session",
  ],
  "api/auth": ["config", "api/errors"],
  "api/schemas": ["utils/money", "utils/dates"],
  "core/pricing": ["core/rules", "db/queries", "utils/money", "core/stock"],
  "core/stock": ["db/queries", "db/models", "core/reservations"],
  "core/reservations": ["db/models", "core/stock", "notify/webhooks"],
  "core/allocation": ["core/stock", "core/rules", "core/forecast", "utils/retry"],
  "core/forecast": ["db/queries", "utils/dates"],
  "core/rules": ["config", "utils/money"],
  "db/queries": ["db/models", "db/session"],
  "db/models": ["db/session", "utils/dates"],
  "db/migrations": ["db/models", "db/session"],
  "db/session": ["config"],
  "sync/importer": ["sync/csv_reader", "sync/legacy_adapter", "core/stock", "utils/retry"],
  "sync/exporter": ["db/queries", "utils/text"],
  "sync/legacy_adapter": ["db/models", "utils/text", "utils/dates"],
  "sync/scheduler": ["sync/importer", "sync/exporter", "utils/retry"],
  "notify/email": ["config", "utils/text"],
  "notify/webhooks": ["utils/retry", "config"],
  "reports/daily": ["db/queries", "reports/charts", "reports/export_pdf", "utils/dates"],
  "reports/export_pdf": ["reports/charts", "utils/text"],
  plugins: ["config"],
};

/** Co-change pairs with no import between them: (a, b, confidence). */
const HIDDEN: readonly (readonly [string, string, number])[] = [
  ["core/pricing", "sync/legacy_adapter", 0.72],
  ["api/routes", "db/migrations", 0.61],
  ["core/allocation", "reports/daily", 0.55],
];

/** Test files (not ranked) and the modules they import. */
const TEST_IMPORTS: Readonly<Record<string, readonly string[]>> = {
  "tests/core/test_pricing.py": ["core/pricing"],
  "tests/core/test_stock.py": ["core/stock", "core/reservations"],
  "tests/core/test_reservations.py": ["core/reservations"],
  "tests/api/test_routes.py": ["api/routes"],
  "tests/api/test_auth.py": ["api/auth"],
  "tests/db/test_queries.py": ["db/queries"],
  "tests/sync/test_csv_reader.py": ["sync/csv_reader"],
  "tests/utils/test_money.py": ["utils/money"],
  "tests/utils/test_dates.py": ["utils/dates"],
  "tests/test_config.py": ["config"],
};

const CYCLE: readonly string[] = [mod("core/stock"), mod("core/reservations")];

// ---------- graph structure ----------

const IMPORT_EDGES: readonly (readonly [string, string])[] = Object.entries(IMPORTS)
  .flatMap(([src, targets]) => targets.map((t) => [mod(src), mod(t)] as const))
  .sort((a, b) => byCodePoint(a[0], b[0]) || byCodePoint(a[1], b[1]));
const HIDDEN_EDGES = HIDDEN.map(([a, b, w]) => [mod(a), mod(b), w] as const);

const dependenciesOf = (path: string) =>
  IMPORT_EDGES.filter(([s]) => s === path)
    .map(([, t]) => t)
    .sort(byCodePoint);
const dependentsOf = (path: string) =>
  IMPORT_EDGES.filter(([, t]) => t === path)
    .map(([s]) => s)
    .sort(byCodePoint);

/** Files that import `path` directly or indirectly (BFS over reversed import edges). */
function transitiveDependents(path: string): number {
  const seen = new Set<string>([path]);
  const queue = [path];
  while (queue.length > 0) {
    const current = queue.shift() as string;
    for (const importer of dependentsOf(current)) {
      if (!seen.has(importer)) {
        seen.add(importer);
        queue.push(importer);
      }
    }
  }
  return seen.size - 1;
}

const testImportersOf = (path: string) =>
  Object.entries(TEST_IMPORTS)
    .filter(([, targets]) => targets.some((t) => mod(t) === path))
    .map(([test]) => test)
    .sort(byCodePoint);

// ---------- scores and costs ----------

/**
 * Priority of rank i out of n = 40: floor((200 * lower + n) / (2 * n)) with lower = 39 - i, the
 * P3.0 percentile formula. Gives 98, 95, 93, 90, 88, ..., 50 at i = 19, 48 at i = 20, ..., 0.
 */
const priorityOf = (rank: number) => Math.floor((200 * (N_FILES - 1 - rank) + N_FILES) / 80);

/** Calibrated risk grows with the square of priority: 0.03 + 0.62 * (p / 100)^2. */
const riskOf = (priority: number) => round3(0.03 + 0.62 * (priority / 100) ** 2);

/** Effort in hours, one decimal: 0.5 + sloc / 120 + priority / 40. Low and high are 0.8x, 1.2x. */
const hoursOf = (sloc: number, priority: number) => round1(0.5 + sloc / 120 + priority / 40);

const FUNCTION_VERBS = ["apply", "build", "check", "compute", "load", "merge", "parse", "update"];
const FUNCTION_NOUNS = ["batch", "discount", "entry", "order", "price", "record", "rule", "stock"];
const MONTHS: readonly string[] = Array.from({ length: 24 }, (_, k) => {
  const index = 2024 * 12 + 10 + k; // 2024-11 .. 2026-10
  return `${Math.floor(index / 12)}-${String((index % 12) + 1).padStart(2, "0")}`;
});

type Built = {
  detail: Omit<FileDetail, "reasons" | "mitigations" | "shap_by_family" | "top_reason" | "plan_step">;
  hours: number;
};

function buildFile(rank: number): Built {
  const [name, sloc, debt] = TABLE[rank];
  const path = mod(name);
  const has = (t: DebtType) => debt.includes(t);
  const priority = priorityOf(rank);
  const risk = riskOf(priority);
  const spread = 0.04 + 0.12 * risk;
  const hours = hoursOf(sloc, priority);

  // Functions: n = clamp(round(sloc / 120), 1, 8) (none below 20 SLOC), the first is the most complex.
  const nFunctions = sloc < 20 ? 0 : clamp(Math.round(sloc / 120), 1, 8);
  const ccMax = has("complexity")
    ? 22 + Math.round((N_FILES - 1 - rank) / 3)
    : 2 + Math.floor(sloc / 150) + Math.floor(rand() * 3);
  const segment = nFunctions > 0 ? Math.floor(sloc / nFunctions) : 0;
  const functions = Array.from({ length: nFunctions }, (_, k) => {
    const start = k * segment + 3;
    const length = Math.max(3, Math.floor(segment * (k === 0 ? 0.8 : 0.4 + 0.3 * rand())));
    const verb = FUNCTION_VERBS[Math.floor(rand() * FUNCTION_VERBS.length)];
    const noun = FUNCTION_NOUNS[(rank + k) % FUNCTION_NOUNS.length];
    return {
      name: `${verb}_${noun}${k === 0 ? "" : `_${k}`}`,
      start_line: start,
      end_line: start + length - 1,
      cyclomatic: k === 0 ? ccMax : Math.max(1, Math.round(ccMax * (0.55 - 0.06 * k))),
    };
  });

  const activity = MONTHS.map((month) => ({
    month,
    commits: Math.floor(rand() * (1 + priority / 12)),
  }));
  const counts = activity.map((a) => a.commits);
  const sum = (xs: number[]) => xs.reduce((s, x) => s + x, 0);
  const lastActive = counts.findLastIndex((c) => c > 0);
  const commitsTotal = sum(counts) + Math.floor(rand() * 20);
  const decayed = round2(sum(counts.map((c, m) => c * 0.5 ** ((23 - m) / 6))));
  const churnPerCommit = 10 + sloc / 40;

  const dependents = dependentsOf(path);
  const dependencies = dependenciesOf(path);
  const hidden = HIDDEN_EDGES.filter(([a, b]) => a === path || b === path);
  const tests = testImportersOf(path);
  const knowledge = has("knowledge");
  const nAuthors = knowledge ? 2 : 2 + Math.floor(rand() * 8);
  const bugfixes = has("defect_hotspot")
    ? 14 + Math.floor(rand() * 4)
    : Math.floor(rand() * (1 + priority / 20));
  const satd = has("admitted") ? 3 + Math.floor(rand() * 6) : 0;
  const ccSum = sum(functions.map((f) => f.cyclomatic));

  const raw: Record<string, number> = {
    sloc,
    cc_sum: ccSum,
    cc_mean: nFunctions > 0 ? round2(ccSum / nFunctions) : 0,
    cc_max: nFunctions > 0 ? ccMax : 0,
    cog_max: nFunctions > 0 ? Math.round(ccMax * 1.5) : 0,
    nesting_max: nFunctions > 0 ? Math.min(7, 1 + Math.floor(ccMax / 6)) : 0,
    n_functions: nFunctions,
    n_classes: Math.floor(sloc / 250),
    longest_function: Math.max(0, ...functions.map((f) => f.end_line - f.start_line + 1)),
    params_max: nFunctions > 0 ? 2 + Math.floor(rand() * 4) : 0,
    dup_ratio: has("duplication") ? round3(0.18 + 0.1 * rand()) : round3(0.04 * rand()),
    commits_total: commitsTotal,
    commits_90d: sum(counts.slice(-3)),
    commits_180d: sum(counts.slice(-6)),
    commits_decayed: decayed,
    churn_total: Math.round(commitsTotal * churnPerCommit),
    churn_decayed: round2(decayed * churnPerCommit),
    rel_churn_decayed: round4((decayed * churnPerCommit) / Math.max(sloc, 1)),
    age_days: 400 + Math.floor(rand() * 900),
    staleness_days: (lastActive === -1 ? 720 : (23 - lastActive) * 30) + Math.floor(rand() * 20),
    cc_trend: round3((rand() - 0.3) * (has("complexity") ? 2 : 0.5)),
    burstiness: round3(rand() * 0.9 - 0.1),
    is_new: 0,
    n_authors: nAuthors,
    owner_share: knowledge ? round3(0.9 + 0.08 * rand()) : round3(0.35 + 0.4 * rand()),
    minor_authors: knowledge ? 1 : Math.max(0, nAuthors - 2 - Math.floor(rand() * 2)),
    author_entropy: knowledge ? round3(0.3 + 0.2 * rand()) : round3(1 + 1.5 * rand()),
    owner_active: knowledge ? 0 : rand() < 0.8 ? 1 : 0,
    prior_bugfix_commits: bugfixes,
    prior_latent_bugs: Math.floor(bugfixes / 3),
    fan_in: dependents.length,
    fan_out: dependencies.length,
    transitive_dependents: transitiveDependents(path),
    pagerank: round4(0.01 + 0.004 * transitiveDependents(path) + 0.003 * dependents.length),
    betweenness: round4(Math.min(1, 0.01 * dependents.length * dependencies.length)),
    scc_size: CYCLE.includes(path) ? 2 : 1,
    cochange_partners: 1 + Math.floor(rand() * 6) + hidden.length,
    cochange_max_conf: hidden.length > 0 ? Math.max(...hidden.map((h) => h[2])) : round3(0.2 + 0.3 * rand()),
    hidden_coupling: hidden.length,
    has_test_file: tests.length > 0 ? 1 : 0,
    test_importers: tests.length,
    test_cochange_rate: tests.length > 0 ? round3(0.2 + 0.4 * rand()) : 0,
    pkg_test_ratio: 0, // filled in below, once every file is known
    satd_count: satd,
    satd_density: round3((satd * 100) / sloc),
    satd_age_days: satd > 0 ? 200 + Math.floor(rand() * 500) : 0,
  };

  return {
    hours,
    detail: {
      id: MOCK_FIRST_FILE_ID + rank,
      path,
      priority,
      band: bandFor(priority),
      risk,
      risk_low: round3(Math.max(0, risk - spread)),
      risk_high: round3(Math.min(1, risk + spread)),
      cost_low: round2(0.8 * hours),
      cost_high: round2(1.2 * hours),
      debt_types: [...debt],
      features: raw,
      functions,
      dependents,
      dependencies,
      test_importers: tests,
      commit_activity: activity,
    },
  };
}

const BUILT: readonly Built[] = TABLE.map((_, rank) => buildFile(rank));

// pkg_test_ratio: share of the files in the same directory that a test imports.
const dirOf = (path: string) => path.slice(0, path.lastIndexOf("/"));
for (const { detail } of BUILT) {
  const siblings = BUILT.filter((b) => dirOf(b.detail.path) === dirOf(detail.path));
  const tested = siblings.filter((b) => b.detail.test_importers.length > 0).length;
  detail.features.pkg_test_ratio = round3(tested / siblings.length);
}

// pct_ columns: round(100 * lower / n), lower = files with a strictly smaller value.
const percentileOf = (feature: string, value: number) =>
  Math.round((100 * BUILT.filter((b) => b.detail.features[feature] < value).length) / N_FILES);
for (const { detail } of BUILT) {
  const raw = detail.features;
  const features: Record<string, number> = {};
  for (const column of FEATURE_COLUMNS) features[column] = raw[column];
  for (const column of FEATURE_COLUMNS) {
    if (!BINARY_FEATURES.has(column)) features[`pct_${column}`] = percentileOf(column, raw[column]);
  }
  detail.features = features;
}

// ---------- explanations ----------

const REASON_FEATURE: Record<DebtType, readonly [Family, string]> = {
  complexity: ["S", "cc_max"],
  duplication: ["S", "dup_ratio"],
  test: ["T", "has_test_file"],
  architecture: ["I", "transitive_dependents"],
  knowledge: ["O", "owner_share"],
  admitted: ["Q", "satd_count"],
  defect_hotspot: ["D", "prior_bugfix_commits"],
};

/** One ASCII line in the style of tdp_core/model/reasons.yaml. */
function reasonText(feature: string, value: number, percentile: number): string {
  const top = `(top ${Math.max(1, 100 - percentile)}% in repo)`;
  switch (feature) {
    case "cc_max":
      return `Complex function: its most complex function has a cyclomatic complexity of ${value} ${top}`;
    case "dup_ratio":
      return `Duplicated code: ${Math.round(value * 100)}% of its lines also appear elsewhere ${top}`;
    case "has_test_file":
      return "No tests: no test file imports it";
    case "transitive_dependents":
      return `Widely depended on: ${value} file${plural(value)} depend on it directly or indirectly ${top}`;
    case "owner_share":
      return `Knowledge concentration: one author wrote ${Math.round(value * 100)}% of it ${top}`;
    case "satd_count":
      return `Self-admitted debt: ${value} TODO or FIXME comment${plural(value)} ${top}`;
    case "prior_bugfix_commits":
      return `Defect hotspot: ${value} bug-fix commit${plural(value)} touched it ${top}`;
    case "commits_180d":
      return `Recent changes: ${value} commit${plural(value)} in the last 6 months ${top}`;
    default:
      return `Covered by tests: ${value} test file${plural(value)} import it`;
  }
}

function makeReason(family: Family, feature: string, features: Record<string, number>, shap: number): Reason {
  const value = features[feature];
  const percentile = percentileOf(feature, value);
  return { family, feature, value, percentile, shap, text: reasonText(feature, value, percentile) };
}

// ---------- plan ----------

type Candidate = { file: Built; benefit: number; roi: number };

/**
 * The greedy plan for a budget: files with priority >= 50, benefit = risk * (1 + transitive
 * dependents / 10), ordered by benefit per hour (ties by path), each taken when it still fits.
 * When both cycle files are chosen the second moves right after the first and shares its group.
 */
export function mockPlan(budgetHours: number): PlanOut {
  const candidates: Candidate[] = BUILT.filter((b) => b.detail.priority >= 50)
    .map((file) => {
      const benefit = round3(file.detail.risk * (1 + file.detail.features.transitive_dependents / 10));
      return { file, benefit, roi: round3(benefit / file.hours) };
    })
    .sort((a, b) => b.roi - a.roi || byCodePoint(a.file.detail.path, b.file.detail.path));
  const budgetTenths = Math.floor(budgetHours * 10 + 1e-9);
  let usedTenths = 0;
  const chosen: Candidate[] = [];
  for (const candidate of candidates) {
    const tenths = Math.round(candidate.file.hours * 10);
    if (usedTenths + tenths <= budgetTenths) {
      chosen.push(candidate);
      usedTenths += tenths;
    }
  }
  const cycleIdx = chosen
    .map((c, i) => (CYCLE.includes(c.file.detail.path) ? i : -1))
    .filter((i) => i >= 0);
  if (cycleIdx.length === 2) {
    const [partner] = chosen.splice(cycleIdx[1], 1);
    chosen.splice(cycleIdx[0] + 1, 0, partner);
  }
  let group = 0;
  const steps: PlanStepOut[] = chosen.map((c, i) => {
    const path = c.file.detail.path;
    const inCycle = cycleIdx.length === 2 && CYCLE.includes(path);
    const other = CYCLE.find((p) => p !== path) as string;
    const joinsPrevious = inCycle && i > 0 && CYCLE.includes(chosen[i - 1].file.detail.path);
    if (!joinsPrevious) group += 1;
    return {
      order: i + 1,
      path,
      file_id: c.file.detail.id,
      benefit: c.benefit,
      roi: c.roi,
      cost_hours: c.file.hours,
      cost_low: c.file.detail.cost_low,
      cost_high: c.file.detail.cost_high,
      group,
      reason: inCycle
        ? `fix together with ${other} (dependency cycle)`
        : "best return on effort among the remaining files",
    };
  });
  return {
    budget_hours: budgetHours,
    total_hours: round1(usedTenths / 10),
    total_benefit: round3(chosen.reduce((s, c) => s + c.benefit, 0)),
    steps,
  };
}

const DEFAULT_PLAN = mockPlan(40);

// ---------- files ----------

/** Every file in rank order (priority desc, then path). */
export const mockFileDetails: readonly FileDetail[] = BUILT.map(({ detail }) => {
  const priority = detail.priority;
  const base = 0.15 + priority / 40;
  const picks = detail.debt_types.map((t) => REASON_FEATURE[t as DebtType]);
  picks.push(["E", "commits_180d"]);
  const reasons = picks.map(([family, feature], k) =>
    makeReason(family, feature, detail.features, round3(base * [1, 0.6, 0.35][k])),
  );
  const mitigations =
    detail.test_importers.length > 0
      ? [makeReason("T", "test_importers", detail.features, -round3(0.05 + 0.1 * rand()))]
      : [];
  const shapByFamily: Record<string, number> = {};
  for (const family of FAMILY_ORDER) {
    const own = [...reasons, ...mitigations].filter((r) => r.family === family);
    shapByFamily[family] =
      own.length > 0 ? round3(own.reduce((s, r) => s + r.shap, 0)) : round3((rand() - 0.5) * 0.06);
  }
  const step = DEFAULT_PLAN.steps.find((s) => s.file_id === detail.id);
  return {
    ...detail,
    top_reason: reasons[0]?.text ?? "",
    reasons,
    mitigations,
    shap_by_family: shapByFamily,
    plan_step: step ? step.order : null,
  };
});

/** The summary rows of every file, in rank order. */
export const mockFileSummaries: readonly FileSummary[] = mockFileDetails.map((f) => ({
  id: f.id,
  path: f.path,
  priority: f.priority,
  band: f.band,
  risk: f.risk,
  risk_low: f.risk_low,
  risk_high: f.risk_high,
  cost_low: f.cost_low,
  cost_high: f.cost_high,
  debt_types: f.debt_types,
  top_reason: f.top_reason,
}));

const hoursById = new Map(BUILT.map((b) => [b.detail.id, b.hours]));

/** The file with this id, or undefined. */
export function findMockFile(fileId: number): FileDetail | undefined {
  return mockFileDetails.find((f) => f.id === fileId);
}

/** Estimated hours of a file (the midpoint of its cost range). */
export function mockFileHours(fileId: number): number {
  return hoursById.get(fileId) ?? 0;
}

// ---------- analysis ----------

const countBand = (band: string) => mockFileDetails.filter((f) => f.band === band).length;

export const mockSummary: AnalysisSummary = {
  files: N_FILES,
  critical: countBand("Critical"),
  high: countBand("High"),
  medium: countBand("Medium"),
  low: countBand("Low"),
  critical_hours: round1(
    mockFileDetails.filter((f) => f.band === "Critical").reduce((s, f) => s + mockFileHours(f.id), 0),
  ),
  debt_types: Object.fromEntries(
    DEBT_TYPES.map((t) => [t, mockFileDetails.filter((f) => f.debt_types.includes(t)).length]),
  ),
  priority_histogram: Array.from(
    { length: 10 },
    (_, bucket) =>
      mockFileDetails.filter((f) => Math.min(Math.floor(f.priority / 10), 9) === bucket).length,
  ),
};

export const mockAnalysis: Analysis = {
  id: MOCK_ANALYSIS_ID,
  repository_id: MOCK_REPOSITORY_ID,
  repo_url: MOCK_REPO_URL,
  commit_sha: MOCK_COMMIT_SHA,
  model_version: MOCK_MODEL_VERSION,
  status: "succeeded",
  stage: "done",
  progress: 100,
  error: null,
  budget_hours: 40,
  created_at: "2026-10-27T09:15:00Z",
  finished_at: "2026-10-27T09:16:12Z",
  summary: mockSummary,
};

export const mockFailedAnalysis: Analysis = {
  id: MOCK_FAILED_ANALYSIS_ID,
  repository_id: "5c7e9a1b-2d3f-4a5b-8c6d-7e8f9a0b1c2d",
  repo_url: "https://github.com/example/huge-monorepo",
  commit_sha: "4e2d9c1b7a5f3e8d6c0b2a4f6e8d0c2b4a6f8e0d",
  model_version: MOCK_MODEL_VERSION,
  status: "failed",
  stage: "mine",
  progress: 25,
  error: "The repository is larger than the 1 GB limit.",
  budget_hours: 40,
  created_at: "2026-10-27T10:02:05Z",
  finished_at: "2026-10-27T10:02:41Z",
  summary: null,
};

/** Progress messages of a simulated run, worded like the real pipeline's. */
export const MOCK_STAGES: readonly { stage: string; pct: number; message: string }[] = [
  "Mined 382 commits and found 26 bug fixes",
  "Parsed the source files",
  "Built the import and co-change graphs",
  "Computed the file features",
  `Scored ${N_FILES} files`,
  `Explained ${N_FILES} files`,
  `Planned ${DEFAULT_PLAN.steps.length} fix${DEFAULT_PLAN.steps.length === 1 ? "" : "es"} within 40 hours`,
].map((message, k) => ({ ...ANALYSIS_STAGES[k], message }));

/**
 * The Analysis of a simulated run id at `nowMs`, or undefined for any other id. Finished runs
 * carry the same summary (and serve the same files) as the demo analysis.
 */
export function analysisForRunning(id: string, nowMs: number): Analysis | undefined {
  const startedAt = runningStartedAt(id);
  if (startedAt === null) return undefined;
  const state = runningState(startedAt, nowMs);
  const submitted = mockState.submitted.get(id);
  return {
    id,
    repository_id: MOCK_REPOSITORY_ID,
    repo_url: submitted?.repoUrl ?? MOCK_REPO_URL,
    commit_sha: "0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e",
    model_version: MOCK_MODEL_VERSION,
    status: state.status,
    stage: state.stage,
    progress: state.progress,
    error: null,
    budget_hours: submitted?.budgetHours ?? 40,
    created_at: new Date(startedAt).toISOString(),
    finished_at: state.finishedAtMs === null ? null : new Date(state.finishedAtMs).toISOString(),
    summary: state.status === "succeeded" ? mockSummary : null,
  };
}

// ---------- graph and treemap ----------

export const mockGraph: GraphOut = {
  nodes: mockFileDetails.map((f) => ({
    id: f.path,
    priority: f.priority,
    band: f.band,
    sloc: f.features.sloc,
    dependents: f.features.transitive_dependents,
  })),
  edges: [
    ...IMPORT_EDGES.map(([source, target]) => ({
      source,
      target,
      kind: "import" as const,
      weight: 1,
    })),
    ...HIDDEN_EDGES.map(([source, target, weight]) => ({
      source,
      target,
      kind: "hidden_coupling" as const,
      weight,
    })),
  ],
};

function buildTreemap(): TreemapNode {
  type Dir = { name: string; path: string; dirs: Map<string, Dir>; leaves: TreemapNode[] };
  const root: Dir = { name: "inventory-service", path: "", dirs: new Map(), leaves: [] };
  for (const f of mockFileDetails) {
    const parts = f.path.split("/");
    let dir = root;
    for (const part of parts.slice(0, -1)) {
      let next = dir.dirs.get(part);
      if (!next) {
        next = { name: part, path: dir.path ? `${dir.path}/${part}` : part, dirs: new Map(), leaves: [] };
        dir.dirs.set(part, next);
      }
      dir = next;
    }
    dir.leaves.push({
      name: parts[parts.length - 1],
      path: f.path,
      sloc: f.features.sloc,
      priority: f.priority,
      band: f.band,
    });
  }
  const toNode = (dir: Dir): TreemapNode => ({
    name: dir.name,
    path: dir.path,
    children: [...[...dir.dirs.values()].map(toNode), ...dir.leaves].sort((a, b) =>
      byCodePoint(a.name, b.name),
    ),
  });
  return toNode(root);
}

export const mockTreemap: TreemapNode = buildTreemap();

// ---------- trend ----------

const EARLIER_TREND: readonly (readonly [string, number, number, number])[] = [
  ["2026-05-12T09:00:00Z", 6, 8, 52.4],
  ["2026-06-09T09:00:00Z", 6, 7, 49.0],
  ["2026-07-14T09:00:00Z", 5, 7, 44.1],
  ["2026-08-11T09:00:00Z", 5, 6, 40.3],
  ["2026-09-15T09:00:00Z", 4, 7, 38.9],
];
const EARLIER_SHAS = [
  "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d",
  "2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e",
  "3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f",
  "4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f70",
  "5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f7081",
];

/** Six analyses of MOCK_REPOSITORY_ID, oldest first; the last is the demo analysis. */
export const mockTrend: TrendOut = {
  points: [
    ...EARLIER_TREND.map(([createdAt, critical, high, criticalHours], k) => ({
      analysis_id: mockRunningAnalysisId(Date.parse(createdAt)),
      commit_sha: EARLIER_SHAS[k],
      created_at: createdAt,
      critical,
      high,
      critical_hours: criticalHours,
    })),
    {
      analysis_id: MOCK_ANALYSIS_ID,
      commit_sha: MOCK_COMMIT_SHA,
      created_at: mockAnalysis.created_at,
      critical: mockSummary.critical,
      high: mockSummary.high,
      critical_hours: mockSummary.critical_hours,
    },
  ],
};

// ---------- advice and what-if ----------

/** Ready advice for a file: three steps with effort ranges, tests to add and risks. */
export function mockAdviceFor(file: FileDetail): Advice {
  const fn = file.functions[0]?.name ?? "the module body";
  const hours = mockFileHours(file.id);
  const step = (share: number): [number, number] => [round1(hours * share * 0.8), round1(hours * share * 1.2)];
  return {
    summary: `Reduce the maintenance risk of ${file.path} by covering it with tests, then splitting ${fn} into smaller functions.`,
    steps: [
      { order: 1, action: `Add characterisation tests for ${fn}`, targets: [file.path], effort_hours: step(0.4) },
      { order: 2, action: `Split ${fn} into smaller functions`, targets: [`${file.path}::${fn}`], effort_hours: step(0.4) },
      { order: 3, action: "Remove dead branches and inline single-use helpers", targets: [file.path], effort_hours: step(0.2) },
    ],
    tests_to_add: [`test_${fn}_handles_empty_input`, `test_${fn}_keeps_current_results`],
    risks: [
      `${file.dependents.length} module${plural(file.dependents.length)} import this file, so run their tests too.`,
    ],
    confidence: "medium",
    dropped_steps: 0,
  };
}

/** Columns each action changes (tdp_core.planning.whatif ACTION_COLUMNS). */
const ACTION_COLUMNS: Record<WhatIfAction, readonly string[]> = {
  add_tests: ["has_test_file", "test_importers", "test_cochange_rate"],
  split_longest_function: ["cc_max", "cog_max", "nesting_max", "longest_function"],
  remove_duplication: ["dup_ratio"],
  resolve_satd: ["satd_count", "satd_density", "satd_age_days"],
  break_cycle: ["scc_size", "hidden_coupling"],
};

/** Priority points each action removes when it applies to the file. */
const ACTION_DELTAS: Record<WhatIfAction, number> = {
  add_tests: 12,
  split_longest_function: 9,
  remove_duplication: 7,
  resolve_satd: 4,
  break_cycle: 10,
};

const MEDIAN_TEST_COCHANGE = (() => {
  const rates = mockFileDetails
    .filter((f) => f.features.test_importers > 0)
    .map((f) => f.features.test_cochange_rate)
    .sort((a, b) => a - b);
  const mid = Math.floor(rates.length / 2);
  return rates.length % 2 ? rates[mid] : round3((rates[mid - 1] + rates[mid]) / 2);
})();

function applies(action: WhatIfAction, features: Record<string, number>): boolean {
  switch (action) {
    case "add_tests":
      return features.has_test_file === 0;
    case "split_longest_function":
      return features.longest_function > 0;
    case "remove_duplication":
      return features.dup_ratio > 0.05;
    case "resolve_satd":
      return features.satd_count > 0;
    case "break_cycle":
      return features.scc_size > 1;
  }
}

/** New value of a column after an action (the improving direction only, as in P4.2). */
function target(column: string, v: number): number {
  switch (column) {
    case "has_test_file":
      return 1;
    case "test_importers":
      return Math.max(v, 1);
    case "test_cochange_rate":
      return Math.max(v, MEDIAN_TEST_COCHANGE);
    case "cc_max":
      return Math.min(v, Math.max(10, v / 2));
    case "cog_max":
    case "longest_function":
      return v / 2;
    case "nesting_max":
      return Math.max(0, v - 1);
    case "scc_size":
      return Math.min(v, 1);
    default:
      return 0;
  }
}

/** The before and after scores of a file when the actions are taken (duplicates count once). */
export function mockWhatIf(file: FileDetail, actions: readonly WhatIfAction[]): WhatIfOut {
  const chosen = [...new Set(actions)];
  const delta = chosen
    .filter((a) => applies(a, file.features))
    .reduce((s, a) => s + ACTION_DELTAS[a], 0);
  const priority = Math.max(0, file.priority - delta);
  const columns = new Set(chosen.flatMap((a) => ACTION_COLUMNS[a]));
  const changed: Record<string, [number, number]> = {};
  for (const column of FEATURE_COLUMNS) {
    if (!columns.has(column)) continue;
    const old = file.features[column];
    const next = round4(target(column, old));
    if (next !== old) changed[column] = [old, next];
  }
  return {
    before: { priority: file.priority, band: file.band, risk: file.risk },
    after: {
      priority,
      band: bandFor(priority),
      risk: round3(Math.max(0.01, (file.risk * (priority + 5)) / (file.priority + 5))),
    },
    changed_features: changed,
  };
}

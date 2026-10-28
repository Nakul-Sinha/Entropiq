// @vitest-environment node
import { describe, expect, it } from "vitest";

import type { TreemapNode } from "@/lib/api/types";
import { bandFor, BANDS } from "@/lib/bands";
import { BINARY_FEATURES, DEBT_TYPES, FAMILY_ORDER, FEATURE_COLUMNS } from "@/lib/domain";

import * as fixtures from "./fixtures";
import {
  MOCK_FIRST_FILE_ID,
  mockFileDetails,
  mockFileHours,
  mockFileSummaries,
  mockGraph,
  mockPlan,
  mockSummary,
  mockTreemap,
  mockTrend,
  mockWhatIf,
} from "./fixtures";
import { runningStartedAt } from "./progress";

const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
const imports = mockGraph.edges.filter((e) => e.kind === "import");
const hidden = mockGraph.edges.filter((e) => e.kind === "hidden_coupling");

describe("files", () => {
  it("has 40 files with consecutive ids in rank order", () => {
    expect(mockFileDetails).toHaveLength(40);
    expect(mockFileDetails.map((f) => f.id)).toEqual(
      Array.from({ length: 40 }, (_, i) => MOCK_FIRST_FILE_ID + i),
    );
    const sorted = [...mockFileDetails].sort(
      (a, b) => b.priority - a.priority || byCodePoint(a.path, b.path),
    );
    expect(sorted.map((f) => f.id)).toEqual(mockFileDetails.map((f) => f.id));
  });

  it("uses the percentile priority formula and tdp_core bands", () => {
    mockFileDetails.forEach((f, i) => {
      expect(f.priority).toBe(Math.floor((200 * (39 - i) + 40) / 80));
      expect(f.band).toBe(bandFor(f.priority));
    });
    expect(mockFileDetails.slice(0, 4).map((f) => f.priority)).toEqual([98, 95, 93, 90]);
    const counts = BANDS.map((b) => mockFileDetails.filter((f) => f.band === b).length);
    expect(counts).toEqual([4, 6, 10, 20]);
  });

  it("keeps ranges ordered and debt types known", () => {
    for (const f of mockFileDetails) {
      expect(f.risk_low).toBeLessThanOrEqual(f.risk);
      expect(f.risk).toBeLessThanOrEqual(f.risk_high);
      expect(f.risk_low).toBeGreaterThanOrEqual(0);
      expect(f.risk_high).toBeLessThanOrEqual(1);
      expect(f.cost_low).toBeLessThanOrEqual(f.cost_high);
      expect(f.debt_types.length).toBeLessThanOrEqual(2);
      for (const t of f.debt_types) expect(DEBT_TYPES).toContain(t);
      expect(f.top_reason).toBe(f.reasons[0]?.text ?? "");
      expect(f.reasons.length).toBeGreaterThanOrEqual(1);
      expect(f.reasons.length).toBeLessThanOrEqual(3);
      for (const r of f.reasons) {
        expect(r.shap).toBeGreaterThan(0);
        expect(r.percentile).toBeGreaterThanOrEqual(0);
        expect(r.percentile).toBeLessThanOrEqual(100);
      }
      expect(f.mitigations.length).toBeLessThanOrEqual(1);
      for (const m of f.mitigations) expect(m.shap).toBeLessThan(0);
      expect(Object.keys(f.shap_by_family).sort()).toEqual([...FAMILY_ORDER].sort());
    }
  });

  it("summaries are the detail's summary fields", () => {
    expect(mockFileSummaries.map((s) => s.id)).toEqual(mockFileDetails.map((f) => f.id));
    mockFileSummaries.forEach((s, i) => expect(mockFileDetails[i]).toMatchObject(s));
  });

  it("has all 89 model input features", () => {
    const expected = [
      ...FEATURE_COLUMNS,
      ...FEATURE_COLUMNS.filter((c) => !BINARY_FEATURES.has(c)).map((c) => `pct_${c}`),
    ];
    expect(expected).toHaveLength(89);
    for (const f of mockFileDetails) {
      expect(Object.keys(f.features)).toEqual(expected);
      for (const value of Object.values(f.features)) expect(Number.isFinite(value)).toBe(true);
    }
  });

  it("derives graph and test features from the edges", () => {
    for (const f of mockFileDetails) {
      const into = imports.filter((e) => e.target === f.path).map((e) => e.source);
      const out = imports.filter((e) => e.source === f.path).map((e) => e.target);
      expect(f.features.fan_in).toBe(into.length);
      expect(f.features.fan_out).toBe(out.length);
      expect(f.dependents).toEqual([...into].sort(byCodePoint));
      expect(f.dependencies).toEqual([...out].sort(byCodePoint));
      const seen = new Set([f.path]);
      const queue = [f.path];
      while (queue.length) {
        const current = queue.shift() as string;
        for (const e of imports) {
          if (e.target === current && !seen.has(e.source)) {
            seen.add(e.source);
            queue.push(e.source);
          }
        }
      }
      expect(f.features.transitive_dependents).toBe(seen.size - 1);
      const inCycle = f.path.endsWith("core/stock.py") || f.path.endsWith("core/reservations.py");
      expect(f.features.scc_size).toBe(inCycle ? 2 : 1);
      expect(f.features.hidden_coupling).toBe(
        hidden.filter((e) => e.source === f.path || e.target === f.path).length,
      );
      expect(f.features.test_importers).toBe(f.test_importers.length);
      expect(f.features.has_test_file).toBe(f.test_importers.length > 0 ? 1 : 0);
      if (f.debt_types.includes("test")) expect(f.test_importers).toEqual([]);
      expect(f.features.satd_count > 0).toBe(f.debt_types.includes("admitted"));
      if (f.debt_types.includes("duplication")) {
        expect(f.features.dup_ratio).toBeGreaterThanOrEqual(0.15);
      }
    }
  });

  it("lists functions inside the file, the first being the most complex", () => {
    for (const f of mockFileDetails) {
      const sloc = f.features.sloc;
      const expected = sloc < 20 ? 0 : Math.min(8, Math.max(1, Math.round(sloc / 120)));
      expect(f.functions).toHaveLength(expected);
      let previousEnd = 0;
      for (const fn of f.functions) {
        expect(fn.start_line).toBeGreaterThan(previousEnd);
        expect(fn.end_line).toBeGreaterThanOrEqual(fn.start_line);
        expect(fn.end_line).toBeLessThanOrEqual(sloc + 3);
        expect(fn.cyclomatic).toBeGreaterThanOrEqual(1);
        previousEnd = fn.end_line;
      }
      if (f.functions.length > 0) expect(f.functions[0].cyclomatic).toBe(f.features.cc_max);
    }
  });

  it("has 24 consecutive months of commit activity", () => {
    for (const f of mockFileDetails) {
      expect(f.commit_activity).toHaveLength(24);
      expect(f.commit_activity[0].month).toBe("2024-11");
      expect(f.commit_activity[23].month).toBe("2026-10");
      f.commit_activity.forEach(({ month, commits }, k) => {
        const index = 2024 * 12 + 10 + k;
        expect(month).toBe(
          `${Math.floor(index / 12)}-${String((index % 12) + 1).padStart(2, "0")}`,
        );
        expect(Number.isInteger(commits) && commits >= 0).toBe(true);
      });
    }
  });
});

describe("summary", () => {
  it("matches the files", () => {
    expect(mockSummary).toMatchObject({ files: 40, critical: 4, high: 6, medium: 10, low: 20 });
    expect(mockSummary.priority_histogram).toHaveLength(10);
    expect(mockSummary.priority_histogram.reduce((s, n) => s + n, 0)).toBe(40);
    expect(Object.keys(mockSummary.debt_types)).toEqual([...DEBT_TYPES]);
    for (const t of DEBT_TYPES) {
      expect(mockSummary.debt_types[t]).toBe(
        mockFileDetails.filter((f) => f.debt_types.includes(t)).length,
      );
    }
    const criticalHours = mockFileDetails
      .filter((f) => f.band === "Critical")
      .reduce((s, f) => s + mockFileHours(f.id), 0);
    expect(mockSummary.critical_hours).toBeCloseTo(criticalHours, 1);
  });
});

describe("graph and treemap", () => {
  it("has a node per file and 68 import plus 3 hidden edges", () => {
    const paths = mockFileDetails.map((f) => f.path);
    expect(mockGraph.nodes.map((n) => n.id)).toEqual(paths);
    expect(imports).toHaveLength(68);
    expect(hidden).toHaveLength(3);
    for (const e of mockGraph.edges) {
      expect(paths).toContain(e.source);
      expect(paths).toContain(e.target);
    }
    for (const e of hidden) {
      const linked = imports.some(
        (i) =>
          (i.source === e.source && i.target === e.target) ||
          (i.source === e.target && i.target === e.source),
      );
      expect(linked).toBe(false);
    }
  });

  it("has a leaf per file with its sloc, priority and band", () => {
    const leaves: TreemapNode[] = [];
    const walk = (node: TreemapNode) => {
      if (node.children) node.children.forEach(walk);
      else leaves.push(node);
    };
    walk(mockTreemap);
    expect(mockTreemap.name).toBe("inventory-service");
    expect(leaves.map((l) => l.path).sort(byCodePoint)).toEqual(
      mockFileDetails.map((f) => f.path).sort(byCodePoint),
    );
    for (const leaf of leaves) {
      const file = mockFileDetails.find((f) => f.path === leaf.path);
      expect(leaf).toMatchObject({
        sloc: file?.features.sloc,
        priority: file?.priority,
        band: file?.band,
      });
    }
  });
});

describe("plan and what-if", () => {
  it("plans steps that match the files", () => {
    const plan = mockPlan(40);
    expect(plan.steps.length).toBeGreaterThan(0);
    expect(plan.total_hours).toBeLessThanOrEqual(40);
    let lastGroup = 0;
    for (const step of plan.steps) {
      const file = mockFileDetails.find((f) => f.id === step.file_id);
      expect(file?.path).toBe(step.path);
      expect(file?.priority).toBeGreaterThanOrEqual(50);
      expect(step.cost_low).toBe(file?.cost_low);
      expect(step.cost_high).toBe(file?.cost_high);
      expect(step.group === lastGroup || step.group === lastGroup + 1).toBe(true);
      lastGroup = step.group;
    }
    for (const f of mockFileDetails) {
      const step = plan.steps.find((s) => s.file_id === f.id);
      expect(f.plan_step).toBe(step ? step.order : null);
    }
    expect(mockPlan(0.5).steps).toEqual([]);
  });

  it("keeps the dependency cycle together when both files fit", () => {
    const plan = mockPlan(2000);
    const i = plan.steps.findIndex((s) => s.path.endsWith("core/stock.py"));
    const j = plan.steps.findIndex((s) => s.path.endsWith("core/reservations.py"));
    expect(Math.abs(i - j)).toBe(1);
    expect(plan.steps[i].group).toBe(plan.steps[j].group);
    expect(plan.steps[i].reason).toContain("dependency cycle");
  });

  it("lowers the priority only through applicable actions", () => {
    const file = mockFileDetails[0];
    const result = mockWhatIf(file, ["split_longest_function", "add_tests"]);
    expect(result.before).toEqual({ priority: file.priority, band: file.band, risk: file.risk });
    expect(result.after.priority).toBeLessThan(file.priority);
    expect(result.after.band).toBe(bandFor(result.after.priority));
    expect(Object.keys(result.changed_features)).toEqual(
      FEATURE_COLUMNS.filter((c) => c in result.changed_features),
    );
    // break_cycle does not apply outside a cycle; like P4.2 it still zeroes hidden coupling.
    const loose = mockFileDetails.find(
      (f) => f.features.scc_size === 1 && f.features.hidden_coupling === 0,
    );
    expect(loose).toBeDefined();
    if (!loose) return;
    const noCycle = mockWhatIf(loose, ["break_cycle"]);
    expect(noCycle.after.priority).toBe(loose.priority);
    expect(noCycle.changed_features).toEqual({});
    expect(mockWhatIf(file, ["break_cycle"]).changed_features).toEqual({ hidden_coupling: [1, 0] });
  });
});

describe("trend and strings", () => {
  it("ends the trend with the demo analysis", () => {
    expect(mockTrend.points).toHaveLength(6);
    const last = mockTrend.points[5];
    expect(last).toMatchObject({
      analysis_id: fixtures.MOCK_ANALYSIS_ID,
      critical: mockSummary.critical,
      high: mockSummary.high,
      critical_hours: mockSummary.critical_hours,
    });
    for (const point of mockTrend.points.slice(0, 5)) {
      expect(runningStartedAt(point.analysis_id)).toBe(Date.parse(point.created_at));
    }
  });

  it("never uses en or em dashes", () => {
    const text = JSON.stringify([
      mockFileDetails,
      mockGraph,
      mockTreemap,
      mockTrend,
      mockPlan(40),
      fixtures.mockAnalysis,
      fixtures.mockFailedAnalysis,
      fixtures.MOCK_STAGES,
      mockFileDetails.map((f) => fixtures.mockAdviceFor(f)),
    ]);
    expect(text).not.toMatch(/[–—]/);
  });
});

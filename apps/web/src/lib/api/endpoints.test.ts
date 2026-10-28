// @vitest-environment node
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { FEATURE_COLUMNS } from "@/lib/domain";
import { mockFetch } from "@/mocks/fetch";
import {
  MOCK_ADVICE_DISABLED_FILE_ID,
  MOCK_ADVICE_FAILED_FILE_ID,
  MOCK_ANALYSIS_ID,
  MOCK_FAILED_ANALYSIS_ID,
  MOCK_FIRST_FILE_ID,
  MOCK_REPO_URL,
  MOCK_REPOSITORY_ID,
  mockAnalysis,
  mockFileDetails,
  mockGraph,
  mockSummary,
  mockTreemap,
  mockTrend,
} from "@/mocks/fixtures";
import { server } from "@/mocks/node";
import { mockRunningAnalysisId, mockTiming, runningStartedAt } from "@/mocks/progress";

import { apiBaseUrl, createApiClient } from "./client";
import {
  createAnalysis,
  getAdvice,
  getAnalysis,
  getFile,
  getGraph,
  getPlan,
  getTreemap,
  getTrend,
  listFiles,
  requestAdvice,
  runWhatIf,
  sendFeedback,
} from "./endpoints";
import { ApiError, isApiError } from "./errors";
import type { FileListQuery, FileSummary, WhatIfAction } from "./types";

const ID = MOCK_ANALYSIS_ID;

async function apiError(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof ApiError) return error;
    throw error;
  }
  throw new Error("expected an ApiError");
}

async function allPages(query: FileListQuery): Promise<FileSummary[]> {
  const items: FileSummary[] = [];
  let cursor: string | undefined;
  for (let page = 0; page < 100; page++) {
    const result = await listFiles(ID, { ...query, cursor });
    items.push(...result.items);
    if (result.next_cursor === null) return items;
    cursor = result.next_cursor;
  }
  throw new Error("too many pages");
}

describe("analyses", () => {
  it("reuses the demo analysis for its repository URL", async () => {
    expect(await createAnalysis({ repo_url: `${MOCK_REPO_URL}.git` })).toEqual({
      id: MOCK_ANALYSIS_ID,
      status: "succeeded",
      reused: true,
    });
  });

  it("starts a simulated run for another URL", async () => {
    const accepted = await createAnalysis({
      repo_url: "https://gitlab.com/group/project",
      budget_hours: 12,
    });
    expect(accepted.status).toBe("queued");
    expect(accepted.reused).toBe(false);
    expect(runningStartedAt(accepted.id)).not.toBeNull();
    const analysis = await getAnalysis(accepted.id);
    expect(analysis.repo_url).toBe("https://gitlab.com/group/project");
    expect(analysis.budget_hours).toBe(12);
  });

  it("rejects an invalid URL or budget with a 422 problem", async () => {
    const error = await apiError(createAnalysis({ repo_url: "http://x" }));
    expect(error.status).toBe(422);
    expect(error.problem.title).toBe("Invalid repository URL");
    expect(error.message).toBe("Use an https URL.");
    const budget = await apiError(
      createAnalysis({ repo_url: "https://github.com/a/b", budget_hours: 0.1 }),
    );
    expect(budget.status).toBe(422);
  });

  it("returns the demo, failed and simulated analyses", async () => {
    expect(await getAnalysis(ID)).toEqual(mockAnalysis);
    const failed = await getAnalysis(MOCK_FAILED_ANALYSIS_ID);
    expect(failed.status).toBe("failed");
    expect(failed.summary).toBeNull();

    const finished = await getAnalysis(mockRunningAnalysisId(Date.now() - 100 * mockTiming.stepMs));
    expect(finished.status).toBe("succeeded");
    expect(finished.progress).toBe(100);
    expect(finished.summary).toEqual(mockSummary);

    const fresh = await getAnalysis(mockRunningAnalysisId(Date.now()));
    expect(fresh).toMatchObject({ status: "queued", stage: null, progress: 0, summary: null });
  });

  it("throws an ApiError 404 for an unknown analysis", async () => {
    const error = await apiError(getAnalysis("missing"));
    expect(isApiError(error, 404)).toBe(true);
    expect(error.message).toBe("Analysis missing does not exist.");
  });
});

describe("files", () => {
  it("lists every file by default", async () => {
    const page = await listFiles(ID);
    expect(page.items).toHaveLength(40);
    expect(page.total).toBe(40);
    expect(page.next_cursor).toBeNull();
    expect(page.items.map((f) => f.id)).toEqual(mockFileDetails.map((f) => f.id));
  });

  it("pages through the list without gaps or overlap", async () => {
    const first = await listFiles(ID, { limit: 15 });
    expect(first.items).toHaveLength(15);
    expect(first.total).toBe(40);
    expect(first.next_cursor).not.toBeNull();
    const items = await allPages({ limit: 15 });
    expect(items.map((f) => f.id)).toEqual(mockFileDetails.map((f) => f.id));
  });

  it("filters by band, debt type and path prefix", async () => {
    const bands = await listFiles(ID, { band: "Critical,High" });
    expect(bands.total).toBe(10);
    expect(new Set(bands.items.map((f) => f.band))).toEqual(new Set(["Critical", "High"]));
    const tests = await listFiles(ID, { debt_type: "test" });
    expect(tests.total).toBe(7);
    expect(tests.items.every((f) => f.debt_types.includes("test"))).toBe(true);
    const core = await listFiles(ID, { path_prefix: "src/inventory/core/" });
    expect(core.total).toBeGreaterThan(0);
    expect(core.items.every((f) => f.path.startsWith("src/inventory/core/"))).toBe(true);
  });

  it("sorts by path and by risk", async () => {
    const byPath = (await listFiles(ID, { sort: "path", order: "asc" })).items.map((f) => f.path);
    expect(byPath).toEqual([...byPath].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)));
    const byRisk = (await listFiles(ID, { sort: "risk", order: "asc" })).items.map((f) => f.risk);
    expect(byRisk).toEqual([...byRisk].sort((a, b) => a - b));
    const byCost = (await listFiles(ID, { sort: "cost" })).items.map((f) => f.cost_high);
    expect(byCost).toEqual([...byCost].sort((a, b) => b - a));
  });

  it("rejects invalid queries", async () => {
    expect((await apiError(listFiles(ID, { band: "Severe" }))).status).toBe(422);
    expect((await apiError(listFiles(ID, { limit: 0 }))).status).toBe(422);
    expect((await apiError(listFiles(ID, { limit: 501 }))).status).toBe(422);
    expect((await apiError(listFiles(ID, { debt_type: "style" }))).status).toBe(422);
    const cursor = await apiError(listFiles(ID, { cursor: "not-a-cursor" }));
    expect(cursor.status).toBe(400);
    expect(cursor.problem.title).toBe("Invalid cursor");
  });

  it("refuses data of an unfinished or failed analysis", async () => {
    const running = await apiError(listFiles(mockRunningAnalysisId(Date.now())));
    expect(running.status).toBe(409);
    expect(running.message).toBe("The analysis has not finished.");
    const failed = await apiError(getGraph(MOCK_FAILED_ANALYSIS_ID));
    expect(failed.status).toBe(409);
    expect(failed.message).toBe("The analysis failed.");
  });

  it("returns a file detail or a 404", async () => {
    expect(await getFile(ID, MOCK_FIRST_FILE_ID)).toEqual(mockFileDetails[0]);
    expect((await apiError(getFile(ID, 99))).status).toBe(404);
  });
});

describe("graph, treemap and plan", () => {
  it("returns the fixtures", async () => {
    expect(await getGraph(ID)).toEqual(mockGraph);
    expect(await getTreemap(ID)).toEqual(mockTreemap);
  });

  it("plans within the budget", async () => {
    for (const budget of [undefined, 40, 8, 2000]) {
      const plan = await getPlan(ID, budget);
      expect(plan.budget_hours).toBe(budget ?? 40);
      expect(plan.total_hours).toBeLessThanOrEqual(budget ?? 40);
      expect(plan.steps.map((s) => s.order)).toEqual(plan.steps.map((_, i) => i + 1));
    }
    expect((await apiError(getPlan(ID, 0.1))).status).toBe(422);
  });

  it("simulates what-if actions", async () => {
    const result = await runWhatIf(ID, {
      file_id: MOCK_FIRST_FILE_ID,
      actions: ["add_tests", "split_longest_function"],
    });
    expect(result.after.priority).toBeLessThanOrEqual(result.before.priority);
    expect(Object.keys(result.changed_features).length).toBeGreaterThan(0);
    for (const [name, [before, after]] of Object.entries(result.changed_features)) {
      expect(FEATURE_COLUMNS).toContain(name);
      expect(before).not.toBe(after);
    }
    const unknown = await apiError(
      runWhatIf(ID, { file_id: MOCK_FIRST_FILE_ID, actions: ["rewrite" as WhatIfAction] }),
    );
    expect(unknown.status).toBe(422);
    expect((await apiError(runWhatIf(ID, { file_id: 5, actions: ["add_tests"] }))).status).toBe(
      404,
    );
  });
});

describe("advice and feedback", () => {
  it("queues advice and makes it ready on the second poll", async () => {
    const fileId = MOCK_FIRST_FILE_ID;
    expect((await getAdvice(fileId)).status).toBe("none");
    expect(await requestAdvice(fileId)).toEqual({ status: "queued" });
    expect(await getAdvice(fileId)).toEqual({
      status: "queued",
      advice: null,
      model: null,
      created_at: null,
    });
    const ready = await getAdvice(fileId);
    expect(ready.status).toBe("ready");
    expect(ready.advice?.steps).toHaveLength(3);
    expect(ready.advice?.summary).toContain(mockFileDetails[0].path);
    expect(await requestAdvice(fileId)).toEqual({ status: "ready" });
  });

  it("reports disabled and failed advice", async () => {
    expect(await requestAdvice(MOCK_ADVICE_DISABLED_FILE_ID)).toEqual({ status: "disabled" });
    expect((await getAdvice(MOCK_ADVICE_DISABLED_FILE_ID)).status).toBe("disabled");
    await requestAdvice(MOCK_ADVICE_FAILED_FILE_ID);
    const failed = await getAdvice(MOCK_ADVICE_FAILED_FILE_ID);
    expect(failed.status).toBe("failed");
    expect(failed.advice).toBeNull();
    expect((await apiError(getAdvice(42))).status).toBe(404);
  });

  it("numbers feedback and validates the verdict", async () => {
    expect(await sendFeedback({ file_id: MOCK_FIRST_FILE_ID, verdict: "agree" })).toEqual({
      id: 1,
    });
    expect(
      await sendFeedback({ file_id: MOCK_FIRST_FILE_ID, verdict: "not_a_problem", comment: "ok" }),
    ).toEqual({ id: 2 });
    const bad = await apiError(
      sendFeedback({ file_id: MOCK_FIRST_FILE_ID, verdict: "meh" as "agree" }),
    );
    expect(bad.status).toBe(422);
  });

  it("returns the trend of the demo repository", async () => {
    const trend = await getTrend(MOCK_REPOSITORY_ID);
    expect(trend).toEqual(mockTrend);
    expect(trend.points).toHaveLength(6);
    expect((await apiError(getTrend("other"))).status).toBe(404);
  });
});

describe("client", () => {
  it("uses API_URL without trailing slashes on the server", () => {
    expect(apiBaseUrl()).toBe("http://localhost:8000");
    vi.stubEnv("API_URL", "http://api:8000//");
    expect(apiBaseUrl()).toBe("http://api:8000");
    vi.unstubAllEnvs();
  });

  it("turns an unexpected server error into an ApiError", async () => {
    server.use(http.get("*/api/v1/analyses/:id", () => new HttpResponse("boom", { status: 500 })));
    const error = await apiError(getAnalysis(ID));
    expect(error.status).toBe(500);
    expect(error.problem.detail).toBe("");
  });

  it("answers in-process with the mock fetch", async () => {
    const client = createApiClient({ fetch: mockFetch });
    expect(await getAnalysis(ID, client)).toEqual(mockAnalysis);
  });
});

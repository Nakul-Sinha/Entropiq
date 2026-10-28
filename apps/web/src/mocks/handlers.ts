/**
 * MSW request handlers for every endpoint of the HTTP API, backed by the fixtures. Tests run them
 * in an MSW node server; mock mode (NEXT_PUBLIC_USE_MOCKS=1) answers in-process through
 * mockFetch. The progress stream is a plain text/event-stream response (MSW's sse() helper needs a
 * global EventSource, which Node and jsdom lack).
 */
import { http, HttpResponse, type HttpHandler } from "msw";

import type { Analysis, FileSummary, WhatIfAction } from "@/lib/api/types";
import { BANDS } from "@/lib/bands";
import { API_BUDGET_RANGE, DEBT_TYPES, FEEDBACK_VERDICTS, WHATIF_ACTIONS } from "@/lib/domain";
import { canonicalRepoUrl, repoUrlError } from "@/lib/repo-url";

import {
  analysisForRunning,
  findMockFile,
  mockAdviceFor,
  mockAnalysis,
  mockFailedAnalysis,
  mockFileSummaries,
  mockGraph,
  mockPlan,
  MOCK_STAGES,
  mockTreemap,
  mockTrend,
  mockWhatIf,
} from "./fixtures";
import {
  MOCK_ADVICE_DISABLED_FILE_ID,
  MOCK_ADVICE_FAILED_FILE_ID,
  MOCK_ANALYSIS_ID,
  MOCK_FAILED_ANALYSIS_ID,
  MOCK_REPO_URL,
  MOCK_REPOSITORY_ID,
} from "./ids";
import { MOCK_STEPS, mockRunningAnalysisId, mockTiming, runningStartedAt } from "./progress";
import { mockState } from "./state";

const API = "*/api/v1";
const ADVICE_MODEL = "claude-sonnet-5-5";
const ADVICE_CREATED_AT = "2026-10-27T09:20:00Z";
const DEFAULT_BUDGET = 40;

/** An RFC 9457 problem response. */
export function problem(status: number, title: string, detail: string): Response {
  return HttpResponse.json(
    { type: "about:blank", title, status, detail },
    { status, headers: { "Content-Type": "application/problem+json" } },
  );
}

const invalid = (detail: string) => problem(422, "Invalid request", detail);

/** The analysis with this id at the current time, or a 404 problem. */
function resolveAnalysis(id: string): Analysis | Response {
  if (id === MOCK_ANALYSIS_ID) return mockAnalysis;
  if (id === MOCK_FAILED_ANALYSIS_ID) return mockFailedAnalysis;
  return (
    analysisForRunning(id, Date.now()) ??
    problem(404, "Not found", `Analysis ${id} does not exist.`)
  );
}

/** The analysis when it has succeeded; otherwise a 404 or 409 problem. */
function succeededAnalysis(id: string): Analysis | Response {
  const analysis = resolveAnalysis(id);
  if (analysis instanceof Response) return analysis;
  if (analysis.status === "failed") return problem(409, "Conflict", "The analysis failed.");
  if (analysis.status !== "succeeded") {
    return problem(409, "Conflict", "The analysis has not finished.");
  }
  return analysis;
}

/** The parsed JSON body, or undefined when it is missing or malformed. */
async function jsonBody(request: Request): Promise<unknown> {
  try {
    return await request.json();
  } catch {
    return undefined;
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const fileIdParam = (value: string | readonly string[] | undefined): number =>
  /^\d+$/.test(String(value)) ? Number(value) : NaN;

// ---------- progress stream ----------

const encoder = new TextEncoder();
const frame = (event: string, data: unknown) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

/**
 * The SSE progress stream of an analysis: a ": connected" comment, then for a simulated run the
 * current stage (when one has finished) and every later stage on its schedule, then `done`.
 * Finished and failed analyses get `done` at once.
 */
function progressStream(analysis: Analysis): ReadableStream<Uint8Array> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (text: string) => controller.enqueue(encoder.encode(text));
      send(": connected\n\n");
      const startedAt = runningStartedAt(analysis.id);
      if (startedAt === null || analysis.status === "succeeded" || analysis.status === "failed") {
        send(frame("done", { status: analysis.status === "failed" ? "failed" : "succeeded" }));
        controller.close();
        return;
      }
      const elapsedSteps = () =>
        mockTiming.stepMs <= 0
          ? MOCK_STEPS
          : Math.floor((Date.now() - startedAt) / mockTiming.stepMs);
      const sendStep = (k: number) => {
        if (k < MOCK_STEPS) send(frame("progress", MOCK_STAGES[k - 1]));
        else send(frame("done", { status: "succeeded" }));
      };
      const completed = Math.min(MOCK_STEPS, Math.max(0, elapsedSteps()));
      if (completed >= 1) sendStep(completed);
      if (completed >= MOCK_STEPS) {
        controller.close();
        return;
      }
      let next = completed + 1;
      const schedule = () => {
        const due = startedAt + next * Math.max(0, mockTiming.stepMs);
        timer = setTimeout(
          () => {
            while (next <= MOCK_STEPS && elapsedSteps() >= next) {
              sendStep(next);
              next += 1;
            }
            if (next > MOCK_STEPS) {
              timer = undefined;
              controller.close();
              return;
            }
            schedule();
          },
          Math.max(0, due - Date.now()),
        );
      };
      schedule();
    },
    cancel() {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
    },
  });
}

// ---------- file list ----------

const SORTS = ["priority", "risk", "cost", "path"] as const;
type Sort = (typeof SORTS)[number];
const MAX_LIMIT = 500;
const DEFAULT_LIMIT = 50;

const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

function compareFiles(sort: Sort, a: FileSummary, b: FileSummary): number {
  switch (sort) {
    case "priority":
      return a.priority - b.priority;
    case "risk":
      return a.risk - b.risk;
    case "cost":
      return a.cost_high - b.cost_high;
    case "path":
      return byCodePoint(a.path, b.path);
  }
}

function encodeCursor(offset: number): string {
  return btoa(JSON.stringify({ offset }));
}

function decodeCursor(cursor: string): number | null {
  try {
    const value: unknown = JSON.parse(atob(cursor));
    if (isRecord(value) && Number.isInteger(value.offset) && (value.offset as number) >= 0) {
      return value.offset as number;
    }
  } catch {
    // fall through
  }
  return null;
}

function listFiles(search: URLSearchParams): Response {
  const sort = search.get("sort") ?? "priority";
  if (!(SORTS as readonly string[]).includes(sort)) {
    return invalid(`sort must be one of ${SORTS.join(", ")}.`);
  }
  const order = search.get("order") ?? "desc";
  if (order !== "desc" && order !== "asc") return invalid("order must be desc or asc.");
  const bands = (search.get("band") ?? "")
    .split(",")
    .map((b) => b.trim())
    .filter(Boolean);
  const unknownBand = bands.find((b) => !(BANDS as readonly string[]).includes(b));
  if (unknownBand !== undefined) {
    return invalid(`band must be a comma list of ${BANDS.join(", ")}.`);
  }
  const debtType = search.get("debt_type") || null;
  if (debtType !== null && !(DEBT_TYPES as readonly string[]).includes(debtType)) {
    return invalid(`debt_type must be one of ${DEBT_TYPES.join(", ")}.`);
  }
  const limitText = search.get("limit");
  const limit = limitText === null ? DEFAULT_LIMIT : Number(limitText);
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_LIMIT) {
    return invalid(`limit must be an integer from 1 to ${MAX_LIMIT}.`);
  }
  const cursor = search.get("cursor");
  const offset = cursor === null || cursor === "" ? 0 : decodeCursor(cursor);
  if (offset === null) return problem(400, "Invalid cursor", "The cursor is not valid.");
  const pathPrefix = search.get("path_prefix") ?? "";

  const direction = order === "desc" ? -1 : 1;
  const matching = mockFileSummaries
    .filter((f) => bands.length === 0 || bands.includes(f.band))
    .filter((f) => debtType === null || f.debt_types.includes(debtType))
    .filter((f) => f.path.startsWith(pathPrefix))
    .sort((a, b) => direction * compareFiles(sort as Sort, a, b) || a.id - b.id);
  const items = matching.slice(offset, offset + limit);
  const end = offset + items.length;
  return HttpResponse.json({
    items,
    next_cursor: end < matching.length ? encodeCursor(end) : null,
    total: matching.length,
  });
}

// ---------- handlers ----------

export const handlers: HttpHandler[] = [
  http.post(`${API}/analyses`, async ({ request }) => {
    const body = await jsonBody(request);
    if (!isRecord(body) || typeof body.repo_url !== "string") {
      return problem(422, "Invalid repository URL", "Send a JSON body with repo_url.");
    }
    const error = repoUrlError(body.repo_url);
    if (error !== null) return problem(422, "Invalid repository URL", error);
    const budget = body.budget_hours;
    if (
      budget !== undefined &&
      budget !== null &&
      (typeof budget !== "number" ||
        !Number.isFinite(budget) ||
        budget < API_BUDGET_RANGE.min ||
        budget > API_BUDGET_RANGE.max)
    ) {
      return invalid(
        `budget_hours must be a number from ${API_BUDGET_RANGE.min} to ${API_BUDGET_RANGE.max}.`,
      );
    }
    const repoUrl = canonicalRepoUrl(body.repo_url);
    if (repoUrl === MOCK_REPO_URL) {
      return HttpResponse.json(
        { id: MOCK_ANALYSIS_ID, status: "succeeded", reused: true },
        { status: 202 },
      );
    }
    const id = mockRunningAnalysisId(Date.now());
    mockState.submitted.set(id, {
      repoUrl,
      budgetHours: typeof budget === "number" ? budget : DEFAULT_BUDGET,
    });
    return HttpResponse.json({ id, status: "queued", reused: false }, { status: 202 });
  }),

  http.get(`${API}/analyses/:id`, ({ params }) => {
    const analysis = resolveAnalysis(String(params.id));
    return analysis instanceof Response ? analysis : HttpResponse.json(analysis);
  }),

  http.get(`${API}/analyses/:id/events`, ({ params }) => {
    const analysis = resolveAnalysis(String(params.id));
    if (analysis instanceof Response) return analysis;
    return new HttpResponse(progressStream(analysis), {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }),

  http.get(`${API}/analyses/:id/files`, ({ params, request }) => {
    const analysis = succeededAnalysis(String(params.id));
    if (analysis instanceof Response) return analysis;
    return listFiles(new URL(request.url).searchParams);
  }),

  http.get(`${API}/analyses/:id/files/:fileId`, ({ params }) => {
    const analysis = succeededAnalysis(String(params.id));
    if (analysis instanceof Response) return analysis;
    const file = findMockFile(fileIdParam(params.fileId));
    if (!file) return problem(404, "Not found", `File ${String(params.fileId)} does not exist.`);
    return HttpResponse.json(file);
  }),

  http.get(`${API}/analyses/:id/graph`, ({ params }) => {
    const analysis = succeededAnalysis(String(params.id));
    return analysis instanceof Response ? analysis : HttpResponse.json(mockGraph);
  }),

  http.get(`${API}/analyses/:id/treemap`, ({ params }) => {
    const analysis = succeededAnalysis(String(params.id));
    return analysis instanceof Response ? analysis : HttpResponse.json(mockTreemap);
  }),

  http.get(`${API}/analyses/:id/plan`, ({ params, request }) => {
    const analysis = succeededAnalysis(String(params.id));
    if (analysis instanceof Response) return analysis;
    const text = new URL(request.url).searchParams.get("budget_hours");
    const budget = text === null || text === "" ? DEFAULT_BUDGET : Number(text);
    if (
      !Number.isFinite(budget) ||
      budget < API_BUDGET_RANGE.min ||
      budget > API_BUDGET_RANGE.max
    ) {
      return invalid(
        `budget_hours must be a number from ${API_BUDGET_RANGE.min} to ${API_BUDGET_RANGE.max}.`,
      );
    }
    return HttpResponse.json(mockPlan(budget));
  }),

  http.post(`${API}/analyses/:id/whatif`, async ({ params, request }) => {
    const analysis = succeededAnalysis(String(params.id));
    if (analysis instanceof Response) return analysis;
    const body = await jsonBody(request);
    if (!isRecord(body) || !Number.isInteger(body.file_id)) {
      return invalid("Send a JSON body with an integer file_id and a list of actions.");
    }
    const file = findMockFile(body.file_id as number);
    if (!file) return problem(404, "Not found", `File ${String(body.file_id)} does not exist.`);
    const actions = body.actions;
    if (
      !Array.isArray(actions) ||
      actions.length === 0 ||
      !actions.every((a) => (WHATIF_ACTIONS as readonly unknown[]).includes(a))
    ) {
      return invalid(`actions must be a non-empty list of ${WHATIF_ACTIONS.join(", ")}.`);
    }
    return HttpResponse.json(mockWhatIf(file, actions as WhatIfAction[]));
  }),

  http.post(`${API}/files/:fileId/advice`, ({ params }) => {
    const fileId = fileIdParam(params.fileId);
    if (!findMockFile(fileId)) {
      return problem(404, "Not found", `File ${String(params.fileId)} does not exist.`);
    }
    if (fileId === MOCK_ADVICE_DISABLED_FILE_ID) {
      return HttpResponse.json({ status: "disabled" }, { status: 202 });
    }
    const state = mockState.advice.get(fileId);
    const ready = state !== undefined && state.polls >= 2 && fileId !== MOCK_ADVICE_FAILED_FILE_ID;
    if (!ready) mockState.advice.set(fileId, { polls: 0 });
    return HttpResponse.json({ status: ready ? "ready" : "queued" }, { status: 202 });
  }),

  http.get(`${API}/files/:fileId/advice`, ({ params }) => {
    const fileId = fileIdParam(params.fileId);
    const file = findMockFile(fileId);
    if (!file) return problem(404, "Not found", `File ${String(params.fileId)} does not exist.`);
    const empty = { advice: null, model: null, created_at: null };
    if (fileId === MOCK_ADVICE_DISABLED_FILE_ID) {
      return HttpResponse.json({ status: "disabled", ...empty });
    }
    const state = mockState.advice.get(fileId);
    if (!state) return HttpResponse.json({ status: "none", ...empty });
    state.polls += 1;
    if (fileId === MOCK_ADVICE_FAILED_FILE_ID) {
      return HttpResponse.json({
        status: "failed",
        advice: null,
        model: ADVICE_MODEL,
        created_at: ADVICE_CREATED_AT,
      });
    }
    if (state.polls < 2) return HttpResponse.json({ status: "queued", ...empty });
    return HttpResponse.json({
      status: "ready",
      advice: mockAdviceFor(file),
      model: ADVICE_MODEL,
      created_at: ADVICE_CREATED_AT,
    });
  }),

  http.post(`${API}/feedback`, async ({ request }) => {
    const body = await jsonBody(request);
    if (!isRecord(body) || !Number.isInteger(body.file_id)) {
      return invalid("Send a JSON body with an integer file_id and a verdict.");
    }
    if (!findMockFile(body.file_id as number)) {
      return problem(404, "Not found", `File ${String(body.file_id)} does not exist.`);
    }
    if (!FEEDBACK_VERDICTS.some((v) => v.value === body.verdict)) {
      return invalid(`verdict must be one of ${FEEDBACK_VERDICTS.map((v) => v.value).join(", ")}.`);
    }
    if (body.comment !== undefined && body.comment !== null && typeof body.comment !== "string") {
      return invalid("comment must be a string.");
    }
    const id = mockState.nextFeedbackId;
    mockState.nextFeedbackId += 1;
    return HttpResponse.json({ id }, { status: 201 });
  }),

  http.get(`${API}/repositories/:repoId/trend`, ({ params }) => {
    if (params.repoId !== MOCK_REPOSITORY_ID) {
      return problem(404, "Not found", `Repository ${String(params.repoId)} does not exist.`);
    }
    return HttpResponse.json(mockTrend);
  }),

  http.get("*/healthz", () => HttpResponse.json({ status: "ok" })),
  http.get("*/readyz", () => HttpResponse.json({ db: "ok", redis: "ok" })),
  http.get(
    "*/metrics",
    () =>
      new HttpResponse(
        "# HELP tdp_analysis_total Analyses by status\n# TYPE tdp_analysis_total counter\n",
        { headers: { "Content-Type": "text/plain; version=0.0.4" } },
      ),
  ),
];

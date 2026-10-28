/**
 * One function per API endpoint. Each returns the parsed success body or throws an ApiError
 * carrying the RFC 9457 problem. `client` defaults to the shared `api` client.
 */
import { api, type ApiClient } from "./client";
import { ApiError, toProblem } from "./errors";
import type {
  AdviceAccepted,
  AdviceOut,
  Analysis,
  AnalysisAccepted,
  AnalysisCreate,
  FeedbackCreated,
  FeedbackIn,
  FileDetail,
  FileListQuery,
  FilePage,
  GraphOut,
  PlanOut,
  TreemapNode,
  TrendOut,
  WhatIfOut,
  WhatIfRequest,
} from "./types";

type Result = { data?: unknown; error?: unknown; response: Response };

/**
 * The success body, typed by the caller's declared return type. openapi-fetch widens tuples such
 * as `effort_hours: [number, number]` to arrays, so the body is cast to the type derived in
 * types.ts (the JSON is the same).
 */
function unwrap<T>({ data, error, response }: Result): T {
  if (error !== undefined || !response.ok) throw new ApiError(toProblem(error, response));
  return data as T;
}

/** POST /analyses: starts an analysis, or reuses a finished one for the same commit (202). */
export async function createAnalysis(
  body: AnalysisCreate,
  client: ApiClient = api,
): Promise<AnalysisAccepted> {
  return unwrap(await client.POST("/api/v1/analyses", { body }));
}

/** GET /analyses/{id}. Throws ApiError 404 for an unknown id. */
export async function getAnalysis(id: string, client: ApiClient = api): Promise<Analysis> {
  return unwrap(await client.GET("/api/v1/analyses/{id}", { params: { path: { id } } }));
}

/** GET /analyses/{id}/files: one page of the ranked file list (409 before success). */
export async function listFiles(
  id: string,
  query: FileListQuery = {},
  client: ApiClient = api,
): Promise<FilePage> {
  return unwrap(
    await client.GET("/api/v1/analyses/{id}/files", { params: { path: { id }, query } }),
  );
}

/** GET /analyses/{id}/files/{file_id}. */
export async function getFile(
  id: string,
  fileId: number,
  client: ApiClient = api,
): Promise<FileDetail> {
  return unwrap(
    await client.GET("/api/v1/analyses/{id}/files/{file_id}", {
      params: { path: { id, file_id: fileId } },
    }),
  );
}

/** GET /analyses/{id}/graph: import edges and hidden coupling edges. */
export async function getGraph(id: string, client: ApiClient = api): Promise<GraphOut> {
  return unwrap(await client.GET("/api/v1/analyses/{id}/graph", { params: { path: { id } } }));
}

/** GET /analyses/{id}/treemap: directory tree whose leaves carry sloc, priority and band. */
export async function getTreemap(id: string, client: ApiClient = api): Promise<TreemapNode> {
  return unwrap(await client.GET("/api/v1/analyses/{id}/treemap", { params: { path: { id } } }));
}

/** GET /analyses/{id}/plan. `budgetHours` is 0.5..2000; the API defaults to 40 when omitted. */
export async function getPlan(
  id: string,
  budgetHours?: number,
  client: ApiClient = api,
): Promise<PlanOut> {
  const query = budgetHours === undefined ? {} : { budget_hours: budgetHours };
  return unwrap(
    await client.GET("/api/v1/analyses/{id}/plan", { params: { path: { id }, query } }),
  );
}

/** POST /analyses/{id}/whatif: the priority, band and risk after the chosen actions. */
export async function runWhatIf(
  id: string,
  body: WhatIfRequest,
  client: ApiClient = api,
): Promise<WhatIfOut> {
  return unwrap(
    await client.POST("/api/v1/analyses/{id}/whatif", { params: { path: { id } }, body }),
  );
}

/** POST /files/{file_id}/advice: queues AI advice (202 queued, ready or disabled). */
export async function requestAdvice(
  fileId: number,
  client: ApiClient = api,
): Promise<AdviceAccepted> {
  return unwrap(
    await client.POST("/api/v1/files/{file_id}/advice", {
      params: { path: { file_id: fileId } },
    }),
  );
}

/** GET /files/{file_id}/advice: the advice state; poll while it is "queued". */
export async function getAdvice(fileId: number, client: ApiClient = api): Promise<AdviceOut> {
  return unwrap(
    await client.GET("/api/v1/files/{file_id}/advice", { params: { path: { file_id: fileId } } }),
  );
}

/** POST /feedback (201). */
export async function sendFeedback(
  body: FeedbackIn,
  client: ApiClient = api,
): Promise<FeedbackCreated> {
  return unwrap(await client.POST("/api/v1/feedback", { body }));
}

/** GET /repositories/{repo_id}/trend: one point per finished analysis, oldest first. */
export async function getTrend(repoId: string, client: ApiClient = api): Promise<TrendOut> {
  return unwrap(
    await client.GET("/api/v1/repositories/{repo_id}/trend", {
      params: { path: { repo_id: repoId } },
    }),
  );
}

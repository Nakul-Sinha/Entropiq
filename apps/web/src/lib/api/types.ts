/**
 * Domain types derived from `schema.ts` by path, method and status code.
 *
 * App code imports these, never `components["schemas"]` names (FastAPI chooses those), so running
 * `npm run gen:api` against the real OpenAPI document is a drop-in replacement and any contract
 * drift shows up as type errors here.
 */
import type { paths } from "./schema";

export type { Problem } from "./errors";

type Method = "get" | "post";
type Operation<P extends keyof paths, M extends Method> = NonNullable<paths[P][M]>;

/** The JSON body of response `S` of `M P`. */
export type ResponseBody<P extends keyof paths, M extends Method, S extends number> =
  Operation<P, M> extends { responses: infer R }
    ? S extends keyof R
      ? R[S] extends { content: { "application/json": infer B } }
        ? B
        : never
      : never
    : never;

/** The JSON request body of `M P`. */
export type RequestBody<P extends keyof paths, M extends Method> =
  Operation<P, M> extends { requestBody: { content: { "application/json": infer B } } }
    ? B
    : Operation<P, M> extends { requestBody?: { content: { "application/json": infer B } } }
      ? B
      : never;

/** The query parameters of `M P` (an object whose keys are all optional). */
export type QueryOf<P extends keyof paths, M extends Method> =
  Operation<P, M> extends { parameters: { query?: infer Q } } ? NonNullable<Q> : never;

export type Analysis = ResponseBody<"/api/v1/analyses/{id}", "get", 200>;
export type AnalysisStatus = Analysis["status"];
export type AnalysisSummary = NonNullable<Analysis["summary"]>;
export type AnalysisCreate = RequestBody<"/api/v1/analyses", "post">;
export type AnalysisAccepted = ResponseBody<"/api/v1/analyses", "post", 202>;

export type FilePage = ResponseBody<"/api/v1/analyses/{id}/files", "get", 200>;
export type FileSummary = FilePage["items"][number];
export type FileListQuery = QueryOf<"/api/v1/analyses/{id}/files", "get">;
export type FileDetail = ResponseBody<"/api/v1/analyses/{id}/files/{file_id}", "get", 200>;
export type Reason = FileDetail["reasons"][number];
export type FunctionRow = FileDetail["functions"][number];
export type MonthActivity = FileDetail["commit_activity"][number];
export type Band = FileSummary["band"];

export type GraphOut = ResponseBody<"/api/v1/analyses/{id}/graph", "get", 200>;
export type GraphNodeOut = GraphOut["nodes"][number];
export type GraphEdgeOut = GraphOut["edges"][number];
export type TreemapNode = ResponseBody<"/api/v1/analyses/{id}/treemap", "get", 200>;

export type PlanOut = ResponseBody<"/api/v1/analyses/{id}/plan", "get", 200>;
export type PlanStepOut = PlanOut["steps"][number];

export type WhatIfRequest = RequestBody<"/api/v1/analyses/{id}/whatif", "post">;
export type WhatIfAction = WhatIfRequest["actions"][number];
export type WhatIfOut = ResponseBody<"/api/v1/analyses/{id}/whatif", "post", 200>;
export type WhatIfPoint = WhatIfOut["before"];

export type AdviceAccepted = ResponseBody<"/api/v1/files/{file_id}/advice", "post", 202>;
export type AdviceOut = ResponseBody<"/api/v1/files/{file_id}/advice", "get", 200>;
export type AdviceStatus = AdviceOut["status"];
export type Advice = NonNullable<AdviceOut["advice"]>;

export type FeedbackIn = RequestBody<"/api/v1/feedback", "post">;
export type FeedbackVerdict = FeedbackIn["verdict"];
export type FeedbackCreated = ResponseBody<"/api/v1/feedback", "post", 201>;

export type TrendOut = ResponseBody<"/api/v1/repositories/{repo_id}/trend", "get", 200>;
export type TrendPoint = TrendOut["points"][number];

/** Data of an SSE `progress` event (not part of OpenAPI). `pct` is 0..100 of the whole analysis. */
export type ProgressEventData = { stage: string; pct: number; message: string };
/** Data of the final SSE `done` event. */
export type DoneEventData = { status: AnalysisStatus };

// Compile-time guard: a renamed path, method or status code in schema.ts turns a derived type into
// `never`, which fails this check instead of silently typing values as `never`.
type IsNever<T> = [T] extends [never] ? true : false;
type AllDefined<T extends readonly unknown[]> = true extends {
  [K in keyof T]: IsNever<T[K]>;
}[number]
  ? false
  : true;
type Assert<T extends true> = T;
// eslint-disable-next-line @typescript-eslint/no-unused-vars
type DerivedTypesAreDefined = Assert<
  AllDefined<
    [
      Analysis,
      AnalysisStatus,
      AnalysisSummary,
      AnalysisCreate,
      AnalysisAccepted,
      FilePage,
      FileSummary,
      FileListQuery,
      FileDetail,
      Reason,
      FunctionRow,
      MonthActivity,
      Band,
      GraphOut,
      GraphNodeOut,
      GraphEdgeOut,
      TreemapNode,
      PlanOut,
      PlanStepOut,
      WhatIfRequest,
      WhatIfAction,
      WhatIfOut,
      WhatIfPoint,
      AdviceAccepted,
      AdviceOut,
      AdviceStatus,
      Advice,
      FeedbackIn,
      FeedbackVerdict,
      FeedbackCreated,
      TrendOut,
      TrendPoint,
    ]
  >
>;

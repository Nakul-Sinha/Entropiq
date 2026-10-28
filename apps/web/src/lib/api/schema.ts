/**
 * Hand-written from the HTTP API contract in the exact layout openapi-typescript 7 emits.
 * `npm run gen:api` replaces it once contracts/openapi.json exists.
 */

export interface paths {
  "/api/v1/analyses": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    get?: never;
    put?: never;
    /** Create Analysis */
    post: operations["create_analysis"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/analyses/{id}": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Get Analysis */
    get: operations["get_analysis"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/analyses/{id}/events": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Stream Analysis Events */
    get: operations["stream_analysis_events"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/analyses/{id}/files": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** List Files */
    get: operations["list_files"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/analyses/{id}/files/{file_id}": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Get File */
    get: operations["get_file"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/analyses/{id}/graph": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Get Graph */
    get: operations["get_graph"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/analyses/{id}/treemap": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Get Treemap */
    get: operations["get_treemap"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/analyses/{id}/plan": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Get Plan */
    get: operations["get_plan"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/analyses/{id}/whatif": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    get?: never;
    put?: never;
    /** Run Whatif */
    post: operations["run_whatif"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/files/{file_id}/advice": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Get Advice */
    get: operations["get_advice"];
    put?: never;
    /** Request Advice */
    post: operations["request_advice"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/feedback": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    get?: never;
    put?: never;
    /** Create Feedback */
    post: operations["create_feedback"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/v1/repositories/{repo_id}/trend": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Get Trend */
    get: operations["get_trend"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/healthz": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Healthz */
    get: operations["healthz"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/readyz": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Readyz */
    get: operations["readyz"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/metrics": {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    /** Metrics */
    get: operations["metrics"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
}
export type webhooks = Record<string, never>;
export interface components {
  schemas: {
    /** Problem */
    Problem: { type: string; title: string; status: number; detail: string };
    /** AnalysisCreate */
    AnalysisCreate: { repo_url: string; ref?: string | null; budget_hours?: number | null };
    /** AnalysisAccepted */
    AnalysisAccepted: {
      id: string;
      status: "queued" | "running" | "succeeded" | "failed";
      reused: boolean;
    };
    /** Analysis */
    Analysis: {
      id: string;
      repository_id: string;
      repo_url: string;
      commit_sha: string;
      model_version: string;
      status: "queued" | "running" | "succeeded" | "failed";
      stage: string | null;
      progress: number;
      error: string | null;
      budget_hours: number;
      created_at: string;
      finished_at: string | null;
      summary: components["schemas"]["AnalysisSummary"] | null;
    };
    /** AnalysisSummary */
    AnalysisSummary: {
      files: number;
      critical: number;
      high: number;
      medium: number;
      low: number;
      critical_hours: number;
      debt_types: { [key: string]: number };
      /** @description Ten buckets of width 10 (the last one also holds 100) */ priority_histogram: number[];
    };
    /** Reason */
    Reason: {
      family: string;
      feature: string;
      value: number;
      percentile: number;
      shap: number;
      text: string;
    };
    /** FileSummary */
    FileSummary: {
      id: number;
      path: string;
      priority: number;
      band: "Critical" | "High" | "Medium" | "Low";
      risk: number;
      risk_low: number;
      risk_high: number;
      cost_low: number;
      cost_high: number;
      debt_types: string[];
      top_reason: string;
    };
    /** FilePage */
    FilePage: {
      items: components["schemas"]["FileSummary"][];
      next_cursor: string | null;
      total: number;
    };
    /** FunctionRow */
    FunctionRow: { name: string; start_line: number; end_line: number; cyclomatic: number };
    /** MonthActivity */
    MonthActivity: { month: string; commits: number };
    /** FileDetail */
    FileDetail: {
      id: number;
      path: string;
      priority: number;
      band: "Critical" | "High" | "Medium" | "Low";
      risk: number;
      risk_low: number;
      risk_high: number;
      cost_low: number;
      cost_high: number;
      debt_types: string[];
      top_reason: string;
      reasons: components["schemas"]["Reason"][];
      mitigations: components["schemas"]["Reason"][];
      shap_by_family: { [key: string]: number };
      features: { [key: string]: number };
      functions: components["schemas"]["FunctionRow"][];
      dependents: string[];
      dependencies: string[];
      test_importers: string[];
      commit_activity: components["schemas"]["MonthActivity"][];
      plan_step: number | null;
    };
    /** GraphNode */
    GraphNode: {
      id: string;
      priority: number;
      band: "Critical" | "High" | "Medium" | "Low";
      sloc: number;
      dependents: number;
    };
    /** GraphEdge */
    GraphEdge: {
      source: string;
      target: string;
      kind: "import" | "hidden_coupling";
      weight: number;
    };
    /** Graph */
    Graph: {
      nodes: components["schemas"]["GraphNode"][];
      edges: components["schemas"]["GraphEdge"][];
    };
    /** TreemapNode */
    TreemapNode: {
      name: string;
      path: string;
      children?: components["schemas"]["TreemapNode"][];
      sloc?: number;
      priority?: number;
      band?: "Critical" | "High" | "Medium" | "Low";
    };
    /** PlanStep */
    PlanStep: {
      order: number;
      path: string;
      file_id: number;
      benefit: number;
      roi: number;
      cost_hours: number;
      cost_low: number;
      cost_high: number;
      group: number;
      reason: string;
    };
    /** Plan */
    Plan: {
      budget_hours: number;
      total_hours: number;
      total_benefit: number;
      steps: components["schemas"]["PlanStep"][];
    };
    /** WhatIfRequest */
    WhatIfRequest: {
      file_id: number;
      actions: (
        | "add_tests"
        | "split_longest_function"
        | "remove_duplication"
        | "resolve_satd"
        | "break_cycle"
      )[];
    };
    /** WhatIfPoint */
    WhatIfPoint: { priority: number; band: "Critical" | "High" | "Medium" | "Low"; risk: number };
    /** WhatIfResult */
    WhatIfResult: {
      before: components["schemas"]["WhatIfPoint"];
      after: components["schemas"]["WhatIfPoint"];
      changed_features: { [key: string]: [number, number] };
    };
    /** AdviceAccepted */
    AdviceAccepted: { status: "queued" | "ready" | "disabled" };
    /** AdviceStep */
    AdviceStep: {
      order: number;
      action: string;
      targets: string[];
      effort_hours: [number, number];
    };
    /** Advice */
    Advice: {
      summary: string;
      steps: components["schemas"]["AdviceStep"][];
      tests_to_add: string[];
      risks: string[];
      confidence: "low" | "medium" | "high";
      dropped_steps: number;
    };
    /** AdviceState */
    AdviceState: {
      status: "queued" | "ready" | "failed" | "disabled" | "none";
      advice: components["schemas"]["Advice"] | null;
      model: string | null;
      created_at: string | null;
    };
    /** FeedbackIn */
    FeedbackIn: {
      file_id: number;
      verdict: "agree" | "not_a_problem" | "should_be_higher";
      comment?: string | null;
    };
    /** FeedbackCreated */
    FeedbackCreated: { id: number };
    /** TrendPoint */
    TrendPoint: {
      analysis_id: string;
      commit_sha: string;
      created_at: string;
      critical: number;
      high: number;
      critical_hours: number;
    };
    /** Trend */
    Trend: { points: components["schemas"]["TrendPoint"][] };
    /** Health */
    Health: { status: string };
    /** Ready */
    Ready: { db: string; redis: string };
  };
  responses: never;
  parameters: never;
  requestBodies: never;
  headers: never;
  pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
  create_analysis: {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    requestBody: { content: { "application/json": components["schemas"]["AnalysisCreate"] } };
    responses: {
      /** @description Successful Response */
      202: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["AnalysisAccepted"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  get_analysis: {
    parameters: { query?: never; header?: never; path: { id: string }; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Analysis"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  stream_analysis_events: {
    parameters: { query?: never; header?: never; path: { id: string }; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: { headers: { [name: string]: unknown }; content: { "text/event-stream": string } };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  list_files: {
    parameters: {
      query?: {
        sort?: "priority" | "risk" | "cost" | "path";
        order?: "desc" | "asc";
        /** @description Comma separated band names */ band?: string;
        debt_type?: string;
        path_prefix?: string;
        limit?: number;
        cursor?: string;
      };
      header?: never;
      path: { id: string };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["FilePage"] };
      };
      /** @description Invalid cursor */
      400: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description The analysis has not succeeded */
      409: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  get_file: {
    parameters: {
      query?: never;
      header?: never;
      path: { id: string; file_id: number };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["FileDetail"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description The analysis has not succeeded */
      409: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  get_graph: {
    parameters: { query?: never; header?: never; path: { id: string }; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Graph"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description The analysis has not succeeded */
      409: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  get_treemap: {
    parameters: { query?: never; header?: never; path: { id: string }; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["TreemapNode"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description The analysis has not succeeded */
      409: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  get_plan: {
    parameters: {
      query?: { budget_hours?: number };
      header?: never;
      path: { id: string };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Plan"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description The analysis has not succeeded */
      409: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  run_whatif: {
    parameters: { query?: never; header?: never; path: { id: string }; cookie?: never };
    requestBody: { content: { "application/json": components["schemas"]["WhatIfRequest"] } };
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["WhatIfResult"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description The analysis has not succeeded */
      409: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  request_advice: {
    parameters: { query?: never; header?: never; path: { file_id: number }; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      202: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["AdviceAccepted"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  get_advice: {
    parameters: { query?: never; header?: never; path: { file_id: number }; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["AdviceState"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  create_feedback: {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    requestBody: { content: { "application/json": components["schemas"]["FeedbackIn"] } };
    responses: {
      /** @description Successful Response */
      201: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["FeedbackCreated"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  get_trend: {
    parameters: { query?: never; header?: never; path: { repo_id: string }; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Trend"] };
      };
      /** @description Not found */
      404: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
      /** @description Validation error */
      422: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Problem"] };
      };
    };
  };
  healthz: {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Health"] };
      };
    };
  };
  readyz: {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Ready"] };
      };
      /** @description A dependency is unavailable */
      503: {
        headers: { [name: string]: unknown };
        content: { "application/json": components["schemas"]["Ready"] };
      };
    };
  };
  metrics: {
    parameters: { query?: never; header?: never; path?: never; cookie?: never };
    requestBody?: never;
    responses: {
      /** @description Successful Response */
      200: { headers: { [name: string]: unknown }; content: { "text/plain": string } };
    };
  };
}

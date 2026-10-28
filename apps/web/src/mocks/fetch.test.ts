// @vitest-environment node
import { describe, expect, it } from "vitest";

import { mockFetch } from "./fetch";
import { MOCK_ANALYSIS_ID, MOCK_FIRST_FILE_ID, mockAnalysis } from "./fixtures";

const BASE = "http://localhost:3000";

describe("mockFetch", () => {
  it("answers a GET in-process", async () => {
    const response = await mockFetch(new Request(`${BASE}/api/v1/analyses/${MOCK_ANALYSIS_ID}`));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(mockAnalysis);
  });

  it("answers a POST with a JSON body", async () => {
    const response = await mockFetch(
      new Request(`${BASE}/api/v1/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_id: MOCK_FIRST_FILE_ID, verdict: "agree" }),
      }),
    );
    expect(response.status).toBe(201);
    expect(await response.json()).toEqual({ id: 1 });
  });

  it("returns a 404 problem for an unknown path", async () => {
    const response = await mockFetch(new Request(`${BASE}/api/v1/nothing`, { method: "DELETE" }));
    expect(response.status).toBe(404);
    expect(response.headers.get("Content-Type")).toBe("application/problem+json");
    expect(await response.json()).toEqual({
      type: "about:blank",
      title: "Not found",
      status: 404,
      detail: "No mock handler for DELETE /api/v1/nothing",
    });
  });
});

// @vitest-environment node
import { describe, expect, it } from "vitest";

import { ANALYSIS_SECTIONS, parseAnalysisPath, routes } from "./routes";

describe("routes", () => {
  it("builds every page URL", () => {
    expect(routes.home()).toBe("/");
    expect(routes.analysis("a1")).toBe("/analyses/a1");
    expect(routes.files("a1")).toBe("/analyses/a1/files");
    expect(routes.file("a1", 12)).toBe("/analyses/a1/files/12");
    expect(routes.map("a1")).toBe("/analyses/a1/map");
    expect(routes.graph("a1")).toBe("/analyses/a1/graph");
    expect(routes.plan("a1")).toBe("/analyses/a1/plan");
    expect(routes.trend("r1")).toBe("/repositories/r1/trend");
  });

  it("encodes ids", () => {
    expect(routes.analysis("a/b c")).toBe("/analyses/a%2Fb%20c");
    expect(routes.trend("r?1")).toBe("/repositories/r%3F1/trend");
  });

  it("lists the five analysis sections in order", () => {
    expect(ANALYSIS_SECTIONS.map((s) => s.label)).toEqual([
      "Overview",
      "Files",
      "Map",
      "Graph",
      "Plan",
    ]);
    expect(ANALYSIS_SECTIONS.map((s) => s.href("x"))).toEqual([
      "/analyses/x",
      "/analyses/x/files",
      "/analyses/x/map",
      "/analyses/x/graph",
      "/analyses/x/plan",
    ]);
  });
});

describe("parseAnalysisPath", () => {
  it.each([
    ["/analyses/x", "overview"],
    ["/analyses/x/", "overview"],
    ["/analyses/x/files", "files"],
    ["/analyses/x/files/12", "files"],
    ["/analyses/x/map", "map"],
    ["/analyses/x/graph/", "graph"],
    ["/analyses/x/plan", "plan"],
  ] as const)("finds the section of %s", (pathname, section) => {
    expect(parseAnalysisPath(pathname)).toEqual({ analysisId: "x", section });
  });

  it("decodes the id", () => {
    expect(parseAnalysisPath("/analyses/a%2Fb")).toEqual({
      analysisId: "a/b",
      section: "overview",
    });
  });

  it.each([
    "/",
    "/analyses",
    "/analyses/",
    "/repositories/r/trend",
    "/analyses/x/other",
    "/x/analyses/y",
  ])("returns null for %s", (pathname) => {
    expect(parseAnalysisPath(pathname)).toBeNull();
  });
});

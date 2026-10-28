// @vitest-environment node
import { describe, expect, it } from "vitest";

import {
  formatDateTime,
  formatHours,
  formatHoursRange,
  formatPercent,
  formatRiskRange,
  shortSha,
} from "./format";

describe("format", () => {
  const outputs = [
    formatPercent(0.423),
    formatPercent(0.4237, 1),
    formatRiskRange(0.3, 0.55),
    formatHours(13.25),
    formatHoursRange(10.6, 16),
    formatDateTime("2026-10-27T09:16:12Z"),
    shortSha("8f3a1c5e9b2d4f6a0c7e1b3d5f7a9c2e4b6d8f0a"),
  ];

  it("formats the documented examples", () => {
    expect(outputs).toEqual([
      "42%",
      "42.4%",
      "30 to 55%",
      "13.3 h",
      "10.6 to 16.0 h",
      "27 Oct 2026, 09:16 UTC",
      "8f3a1c5",
    ]);
  });

  it("returns an unparsable date unchanged", () => {
    expect(formatDateTime("not a date")).toBe("not a date");
  });

  it("never uses en or em dashes", () => {
    for (const text of outputs) expect(text).not.toMatch(/[\u2013\u2014]/);
  });
});

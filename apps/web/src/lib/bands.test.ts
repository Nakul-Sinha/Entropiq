// @vitest-environment node
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { BAND_CLASSES, BAND_COLORS, BAND_FOREGROUNDS, BANDS, bandFor } from "./bands";

describe("bandFor", () => {
  it.each([
    [100, "Critical"],
    [90, "Critical"],
    [89, "High"],
    [75, "High"],
    [74, "Medium"],
    [50, "Medium"],
    [49, "Low"],
    [0, "Low"],
    [-1, "Low"],
  ] as const)("maps priority %d to %s like tdp_core band_for", (priority, band) => {
    expect(bandFor(priority)).toBe(band);
  });
});

describe("band palette", () => {
  it("uses the plan's colours", () => {
    expect(BANDS).toEqual(["Critical", "High", "Medium", "Low"]);
    expect(BAND_COLORS).toEqual({
      Critical: "#B23A3A",
      High: "#C76B1A",
      Medium: "#9A8A2E",
      Low: "#4A7A5C",
    });
  });

  it("matches the --band-* tokens in globals.css", () => {
    const css = readFileSync(new URL("../app/globals.css", import.meta.url), "utf8");
    for (const band of BANDS) {
      const name = band.toLowerCase();
      const fill = new RegExp(`--band-${name}:\\s*(#[0-9a-f]{6})`, "i").exec(css);
      const text = new RegExp(`--band-${name}-foreground:\\s*(#[0-9a-f]{6})`, "i").exec(css);
      expect(fill?.[1].toLowerCase()).toBe(BAND_COLORS[band].toLowerCase());
      expect(text?.[1].toLowerCase()).toBe(BAND_FOREGROUNDS[band].toLowerCase());
    }
  });

  it("gives every band literal Tailwind classes", () => {
    for (const band of BANDS) {
      const name = band.toLowerCase();
      expect(BAND_CLASSES[band]).toBe(`bg-band-${name} text-band-${name}-foreground`);
    }
  });
});

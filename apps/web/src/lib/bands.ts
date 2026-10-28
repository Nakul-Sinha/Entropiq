import type { Band } from "@/lib/api/types";

/** Bands from most to least urgent. */
export const BANDS: readonly Band[] = ["Critical", "High", "Medium", "Low"];

/** Lowest priority (0..100) of each band, the same table as tdp_core.contracts.BANDS. */
export const BAND_THRESHOLDS = [
  ["Critical", 90],
  ["High", 75],
  ["Medium", 50],
  ["Low", 0],
] as const;

/** The band of a 0..100 priority, identical to tdp_core band_for (below 0 is Low). */
export function bandFor(priority: number): Band {
  for (const [band, lowest] of BAND_THRESHOLDS) {
    if (priority >= lowest) return band;
  }
  return "Low";
}

/** Band fill colours; keep in sync with the --band-* tokens in src/app/globals.css. */
export const BAND_COLORS: Record<Band, string> = {
  Critical: "#B23A3A",
  High: "#C76B1A",
  Medium: "#9A8A2E",
  Low: "#4A7A5C",
};

/** Text colour on each band fill (every pair has a contrast ratio of at least 4.9:1). */
export const BAND_FOREGROUNDS: Record<Band, string> = {
  Critical: "#FFFFFF",
  High: "#111111",
  Medium: "#111111",
  Low: "#FFFFFF",
};

/** Tailwind classes for a band fill with its text colour. Always show the band name as well. */
export const BAND_CLASSES: Record<Band, string> = {
  Critical: "bg-band-critical text-band-critical-foreground",
  High: "bg-band-high text-band-high-foreground",
  Medium: "bg-band-medium text-band-medium-foreground",
  Low: "bg-band-low text-band-low-foreground",
};

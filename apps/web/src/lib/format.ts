/** Display formatting. Ranges use the word "to", never an en or em dash. */

/** A 0..1 fraction as a percentage: formatPercent(0.423) is "42%". */
export function formatPercent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** A risk range of two 0..1 fractions: formatRiskRange(0.3, 0.55) is "30 to 55%". */
export function formatRiskRange(low: number, high: number): string {
  return `${Math.round(low * 100)} to ${Math.round(high * 100)}%`;
}

/** Hours with one decimal: formatHours(13.25) is "13.3 h". */
export function formatHours(hours: number): string {
  return `${hours.toFixed(1)} h`;
}

/** An hour range: formatHoursRange(10.6, 16) is "10.6 to 16.0 h". */
export function formatHoursRange(low: number, high: number): string {
  return `${low.toFixed(1)} to ${high.toFixed(1)} h`;
}

// A fixed table instead of Intl, so the server render and the browser always agree.
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/**
 * An ISO 8601 instant in UTC (en-GB style): "2026-10-27T09:16:12Z" is "27 Oct 2026, 09:16 UTC".
 * An unparsable string is returned unchanged.
 */
export function formatDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${date.getUTCDate()} ${MONTHS[date.getUTCMonth()]} ${date.getUTCFullYear()}, ` +
    `${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())} UTC`
  );
}

/** The first seven characters of a commit sha. */
export function shortSha(sha: string): string {
  return sha.slice(0, 7);
}

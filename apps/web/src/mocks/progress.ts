/**
 * Stateless simulation of a running analysis. The id encodes the start time, and status, stage
 * and progress follow from the time elapsed, so a server render and the browser (separate JS
 * runtimes with separate mock state) always agree.
 */
import { ANALYSIS_STAGES } from "@/lib/domain";

/** Milliseconds per simulated stage (mutable for tests). A run takes 8 steps: 7 stages and done. */
export const mockTiming = { stepMs: 700 };

export const MOCK_STEPS = ANALYSIS_STAGES.length + 1;

const RUNNING_PREFIX = "00000000-0000-4000-8000-";
const RUNNING_ID = /^00000000-0000-4000-8000-([0-9a-f]{12})$/;

/** The id of a simulated analysis started at `startedAtMs` (Unix ms, 12 lowercase hex digits). */
export function mockRunningAnalysisId(startedAtMs: number): string {
  return RUNNING_PREFIX + Math.max(0, Math.floor(startedAtMs)).toString(16).padStart(12, "0");
}

/** The start time encoded in a simulated analysis id, or null for any other id. */
export function runningStartedAt(id: string): number | null {
  const match = RUNNING_ID.exec(id);
  return match ? parseInt(match[1], 16) : null;
}

export type RunningState = {
  status: "queued" | "running" | "succeeded";
  stage: string | null;
  progress: number;
  /** Steps finished, 0..8. */
  completed: number;
  finishedAtMs: number | null;
};

/**
 * The state of a simulated analysis at `nowMs`: queued for the first step, then one stage of
 * ANALYSIS_STAGES per step, then succeeded with stage "done" and progress 100.
 */
export function runningState(startedAtMs: number, nowMs: number): RunningState {
  const { stepMs } = mockTiming;
  const completed =
    stepMs <= 0
      ? MOCK_STEPS
      : Math.min(MOCK_STEPS, Math.max(0, Math.floor((nowMs - startedAtMs) / stepMs)));
  if (completed === 0) {
    return { status: "queued", stage: null, progress: 0, completed, finishedAtMs: null };
  }
  if (completed < MOCK_STEPS) {
    const { stage, pct } = ANALYSIS_STAGES[completed - 1];
    return { status: "running", stage, progress: pct, completed, finishedAtMs: null };
  }
  return {
    status: "succeeded",
    stage: "done",
    progress: 100,
    completed,
    finishedAtMs: startedAtMs + MOCK_STEPS * Math.max(0, stepMs),
  };
}

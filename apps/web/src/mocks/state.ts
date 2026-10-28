/**
 * Mutable mock backend state. Each JS runtime (the Next server process, each browser tab, each
 * test file) has its own copy, so nothing here may be needed by both a server render and the
 * browser.
 */
import { mockTiming } from "./progress";

export const mockState = {
  /** Advice polls per file id since the last POST. */
  advice: new Map<number, { polls: number }>(),
  nextFeedbackId: 1,
  /** Submitted analyses by id. */
  submitted: new Map<string, { repoUrl: string; budgetHours: number }>(),
};

/** Restores the initial state and the default timing (700 ms per step). */
export function resetMockState(): void {
  mockState.advice.clear();
  mockState.submitted.clear();
  mockState.nextFeedbackId = 1;
  mockTiming.stepMs = 700;
}

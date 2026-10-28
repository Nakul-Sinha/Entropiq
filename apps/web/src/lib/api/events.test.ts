// @vitest-environment node
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { MOCK_ANALYSIS_ID, MOCK_FAILED_ANALYSIS_ID, MOCK_STAGES } from "@/mocks/fixtures";
import { mockRunningAnalysisId, mockTiming } from "@/mocks/progress";

import { getAnalysis } from "./endpoints";
import { ApiError } from "./errors";
import { createSseParser, subscribeToProgress, type ProgressHandlers } from "./events";
import type { DoneEventData, ProgressEventData } from "./types";

describe("createSseParser", () => {
  it("joins lines split across chunks", () => {
    const parser = createSseParser();
    expect(parser.push("event: prog")).toEqual([]);
    expect(parser.push('ress\ndata: {"a"')).toEqual([]);
    expect(parser.push(":1}\n")).toEqual([]);
    expect(parser.push("\n")).toEqual([{ event: "progress", data: '{"a":1}', id: null }]);
  });

  it("treats a CRLF split across chunks as one line break", () => {
    const parser = createSseParser();
    expect(parser.push("data: x\r")).toEqual([]);
    expect(parser.push("\n\r")).toEqual([]);
    expect(parser.push("\n")).toEqual([{ event: "message", data: "x", id: null }]);
  });

  it("accepts CRLF and CR line endings", () => {
    expect(createSseParser().push("event: done\r\ndata: 1\r\n\r\n")).toEqual([
      { event: "done", data: "1", id: null },
    ]);
    expect(createSseParser().push("data: 2\r\rdata: 3\r\r:")).toEqual([
      { event: "message", data: "2", id: null },
      { event: "message", data: "3", id: null },
    ]);
  });

  it("ignores comments and unknown fields", () => {
    expect(createSseParser().push(": connected\n\nretry: 5\nfoo\ndata: y\n\n")).toEqual([
      { event: "message", data: "y", id: null },
    ]);
  });

  it("joins multi-line data, keeps the last id and resets the event name", () => {
    const parser = createSseParser();
    expect(parser.push("id: 7\nevent: a\ndata: one\ndata:two\n\ndata: three\n\n")).toEqual([
      { event: "a", data: "one\ntwo", id: "7" },
      { event: "message", data: "three", id: "7" },
    ]);
  });

  it("does not dispatch an event without data or without a blank line", () => {
    const parser = createSseParser();
    expect(parser.push("event: lonely\n\n")).toEqual([]);
    expect(parser.push("data: pending\n")).toEqual([]);
  });
});

type Recorded = {
  progress: ProgressEventData[];
  done: DoneEventData[];
  errors: unknown[];
  finished: Promise<void>;
  handlers: ProgressHandlers;
};

function record(): Recorded {
  let resolve: () => void = () => undefined;
  const finished = new Promise<void>((r) => {
    resolve = r;
  });
  const rec: Recorded = {
    progress: [],
    done: [],
    errors: [],
    finished,
    handlers: {
      onProgress: (e) => rec.progress.push(e),
      onDone: (e) => {
        rec.done.push(e);
        resolve();
      },
      onError: (e) => {
        rec.errors.push(e);
        resolve();
      },
    },
  };
  return rec;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe("subscribeToProgress", () => {
  beforeAll(async () => {
    // Warm up the client and handlers so the timed run below starts on schedule.
    await getAnalysis(MOCK_ANALYSIS_ID);
  });

  afterEach(() => {
    mockTiming.stepMs = 700;
  });

  it("reports every stage of a simulated run, then done", async () => {
    mockTiming.stepMs = 40;
    const rec = record();
    subscribeToProgress(mockRunningAnalysisId(Date.now()), rec.handlers);
    await rec.finished;
    expect(rec.errors).toEqual([]);
    expect(rec.progress).toEqual(
      MOCK_STAGES.map(({ stage, pct, message }) => ({ stage, pct, message })),
    );
    const pcts = rec.progress.map((p) => p.pct);
    expect([...pcts].sort((a, b) => a - b)).toEqual(pcts);
    expect(rec.done).toEqual([{ status: "succeeded" }]);
  });

  it("finishes at once for a finished analysis", async () => {
    const rec = record();
    subscribeToProgress(MOCK_ANALYSIS_ID, rec.handlers);
    await rec.finished;
    expect(rec.progress).toEqual([]);
    expect(rec.done).toEqual([{ status: "succeeded" }]);
  });

  it("reports a failed analysis", async () => {
    const rec = record();
    subscribeToProgress(MOCK_FAILED_ANALYSIS_ID, rec.handlers);
    await rec.finished;
    expect(rec.done).toEqual([{ status: "failed" }]);
  });

  it("reports an unknown analysis as an ApiError 404", async () => {
    const rec = record();
    subscribeToProgress("no-such-analysis", rec.handlers);
    await rec.finished;
    expect(rec.done).toEqual([]);
    expect(rec.errors).toHaveLength(1);
    expect(rec.errors[0]).toBeInstanceOf(ApiError);
    expect((rec.errors[0] as ApiError).status).toBe(404);
  });

  it("stops calling handlers after unsubscribe", async () => {
    mockTiming.stepMs = 10;
    const rec = record();
    let unsubscribe: () => void = () => undefined;
    const first = new Promise<void>((resolve) => {
      unsubscribe = subscribeToProgress(mockRunningAnalysisId(Date.now()), {
        ...rec.handlers,
        onProgress: (e) => {
          rec.progress.push(e);
          unsubscribe();
          resolve();
        },
      });
    });
    await first;
    unsubscribe();
    await sleep(150);
    expect(rec.progress).toHaveLength(1);
    expect(rec.done).toEqual([]);
    expect(rec.errors).toEqual([]);
  });
});

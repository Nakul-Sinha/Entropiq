/**
 * Server-sent progress events read with fetch streaming through the typed client.
 *
 * EventSource is not used: jsdom has none, the in-process mock backend never touches the network,
 * and fetch keeps every API call on the typed client.
 */
import { api, type ApiClient } from "./client";
import { ApiError, toProblem } from "./errors";
import type { DoneEventData, ProgressEventData } from "./types";

export type SseMessage = { event: string; data: string; id: string | null };

/**
 * An incremental parser for a subset of the WHATWG event-stream format.
 *
 * Lines end with CRLF, LF or CR (a CR at the end of a chunk waits for the next chunk, so a split
 * CRLF counts once). A blank line dispatches the pending event when it has at least one data line;
 * `event` defaults to "message" and data lines are joined with "\n". Lines starting with ":" are
 * comments (heartbeats). Fields other than event, data and id are ignored, and an event that is
 * not terminated by a blank line is never dispatched.
 */
export function createSseParser(): { push(text: string): SseMessage[] } {
  let buffer = "";
  let event = "";
  let data: string[] = [];
  let lastId: string | null = null;

  function processLine(line: string, out: SseMessage[]) {
    if (line === "") {
      if (data.length > 0) out.push({ event: event || "message", data: data.join("\n"), id: lastId });
      event = "";
      data = [];
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
    else if (field === "id") lastId = value;
  }

  return {
    push(text: string): SseMessage[] {
      buffer += text;
      const out: SseMessage[] = [];
      let start = 0;
      for (let i = 0; i < buffer.length; i++) {
        const ch = buffer[i];
        if (ch !== "\n" && ch !== "\r") continue;
        if (ch === "\r" && i === buffer.length - 1) break; // may be the first half of CRLF
        processLine(buffer.slice(start, i), out);
        if (ch === "\r" && buffer[i + 1] === "\n") i++;
        start = i + 1;
      }
      buffer = buffer.slice(start);
      return out;
    },
  };
}

export type ProgressHandlers = {
  onProgress?: (event: ProgressEventData) => void;
  onDone?: (event: DoneEventData) => void;
  onError?: (error: unknown) => void;
};

/**
 * Follows GET /analyses/{id}/events. Calls `onProgress` for each progress event, `onDone` once
 * for the final event, and `onError` for a failed request, malformed data or a stream that ends
 * before "done" (callers then fall back to polling). Returns an idempotent unsubscribe function;
 * after it is called, or after done or an error, no handler fires again.
 */
export function subscribeToProgress(
  analysisId: string,
  handlers: ProgressHandlers,
  client: ApiClient = api,
): () => void {
  const controller = new AbortController();
  let reader: ReadableStreamDefaultReader<string> | null = null;
  let closed = false;

  const close = () => {
    if (closed) return;
    closed = true;
    controller.abort();
    // In-process mock streams ignore the request signal, so cancel the reader explicitly.
    reader?.cancel().catch(() => undefined);
  };
  const fail = (error: unknown) => {
    if (closed) return;
    close();
    handlers.onError?.(error);
  };

  const run = async () => {
    const { data, error, response } = await client.GET("/api/v1/analyses/{id}/events", {
      params: { path: { id: analysisId } },
      parseAs: "stream",
      headers: { Accept: "text/event-stream" },
      signal: controller.signal,
    });
    if (closed) return;
    if (error !== undefined || !response.ok || !data) {
      fail(new ApiError(toProblem(error, response)));
      return;
    }
    reader = data.pipeThrough(new TextDecoderStream()).getReader();
    const parser = createSseParser();
    while (!closed) {
      const { value, done } = await reader.read();
      if (closed) return;
      if (done) {
        fail(new Error("The progress stream ended before the analysis finished."));
        return;
      }
      for (const message of parser.push(value)) {
        if (closed) return;
        if (message.event === "progress") {
          handlers.onProgress?.(JSON.parse(message.data) as ProgressEventData);
        } else if (message.event === "done") {
          const done = JSON.parse(message.data) as DoneEventData;
          close();
          handlers.onDone?.(done);
          return;
        }
      }
    }
  };

  run().catch((error: unknown) => fail(error));
  return close;
}

/** RFC 9457 problem details, the error body of every API endpoint. */
export type Problem = { type: string; title: string; status: number; detail: string };

/** A non-2xx API response. `message` is the problem's detail, or its title when detail is empty. */
export class ApiError extends Error {
  readonly status: number;
  readonly problem: Problem;

  constructor(problem: Problem) {
    super(problem.detail || problem.title);
    this.name = "ApiError";
    this.status = problem.status;
    this.problem = problem;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Normalises an error body into a Problem.
 *
 * - An RFC 9457 object is kept; missing fields are filled from the response.
 * - A FastAPI body (`{detail: string}` or `{detail: [{msg}, ...]}`) keeps the messages, joined by
 *   "; ", as detail; the title is "Invalid request" for a list, 400 or 422, else the status text.
 * - Anything else (no body, text, HTML) gives the status text as title and an empty detail.
 */
export function toProblem(body: unknown, response: Response): Problem {
  const fallbackTitle = response.statusText || "Request failed";
  if (isRecord(body)) {
    if (typeof body.title === "string" || typeof body.type === "string") {
      return {
        type: typeof body.type === "string" ? body.type : "about:blank",
        title: typeof body.title === "string" ? body.title : fallbackTitle,
        status: typeof body.status === "number" ? body.status : response.status,
        detail: typeof body.detail === "string" ? body.detail : "",
      };
    }
    const detail = body.detail;
    if (typeof detail === "string" || Array.isArray(detail)) {
      const messages =
        typeof detail === "string"
          ? [detail]
          : detail
              .map((item) => (isRecord(item) && typeof item.msg === "string" ? item.msg : null))
              .filter((msg): msg is string => msg !== null);
      const invalid = Array.isArray(detail) || response.status === 400 || response.status === 422;
      return {
        type: "about:blank",
        title: invalid ? "Invalid request" : fallbackTitle,
        status: response.status,
        detail: messages.join("; "),
      };
    }
  }
  return { type: "about:blank", title: fallbackTitle, status: response.status, detail: "" };
}

/** True for an ApiError, optionally with the given HTTP status. */
export function isApiError(error: unknown, status?: number): error is ApiError {
  return error instanceof ApiError && (status === undefined || error.status === status);
}

/** A sentence to show the user for any thrown value. */
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.problem.detail || error.problem.title;
  if (error instanceof Error && error.message) return error.message;
  return "Something went wrong.";
}

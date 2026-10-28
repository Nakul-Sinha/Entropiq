import createClient, { type Client } from "openapi-fetch";

import type { paths } from "./schema";

export type ApiClient = Client<paths>;
export type FetchLike = (request: Request) => Promise<Response>;

/** Inlined at build and dev time: "1" answers every call from the in-process mock backend. */
export const USE_MOCKS = process.env.NEXT_PUBLIC_USE_MOCKS === "1";

/**
 * Base URL for API calls, read at call time. In the browser this is the page's own origin, so
 * requests go through the Next rewrite of /api/v1 and never need CORS. On the server it is
 * `API_URL` (default http://localhost:8000) without trailing slashes.
 */
export function apiBaseUrl(): string {
  if (typeof window !== "undefined") return window.location.origin;
  return (process.env.API_URL ?? "http://localhost:8000").replace(/\/+$/, "");
}

/**
 * The fetch used by the default client. openapi-fetch captures its fetch when the client is
 * created, so this wrapper looks up `globalThis.fetch` on every call (test servers patch it later)
 * and loads the mock backend lazily, keeping MSW out of production bundles' first load.
 */
export const apiFetch: FetchLike = async (request) => {
  if (USE_MOCKS) {
    const { mockFetch } = await import("@/mocks/fetch");
    return mockFetch(request);
  }
  return globalThis.fetch(request);
};

/** A typed client for the HTTP API. Defaults: `apiBaseUrl()` and `apiFetch`. */
export function createApiClient(options: { baseUrl?: string; fetch?: FetchLike } = {}): ApiClient {
  return createClient<paths>({
    baseUrl: options.baseUrl ?? apiBaseUrl(),
    fetch: options.fetch ?? apiFetch,
  });
}

/** The shared client used by every endpoint helper unless one is passed explicitly. */
export const api: ApiClient = createApiClient();

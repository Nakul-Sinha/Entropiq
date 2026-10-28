/**
 * The in-process mock backend used in mock mode (NEXT_PUBLIC_USE_MOCKS=1): each request is
 * answered by the MSW handlers without a service worker or network, in the Next server and in the
 * browser alike.
 */
import { getResponse } from "msw";

import type { FetchLike } from "@/lib/api/client";

import { handlers, problem } from "./handlers";

/** Resolves a request against the mock handlers; unmatched requests get a 404 problem. */
export const mockFetch: FetchLike = async (request) => {
  const response = await getResponse(handlers, request);
  if (response) return response;
  const { pathname } = new URL(request.url);
  return problem(404, "Not found", `No mock handler for ${request.method} ${pathname}`);
};

import { isServer, QueryClient } from "@tanstack/react-query";

import { isApiError } from "@/lib/api/errors";

/**
 * A QueryClient with the app defaults: data stays fresh for 30 s, no refetch on window focus,
 * and failed queries are retried twice except for client errors (4xx), which never change.
 */
export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        refetchOnWindowFocus: false,
        retry: (failureCount, error) =>
          !(isApiError(error) && error.status < 500) && failureCount < 2,
      },
    },
  });
}

let browserClient: QueryClient | undefined;

/** A new client per server render; one shared client in the browser. */
export function getQueryClient(): QueryClient {
  if (isServer) return makeQueryClient();
  browserClient ??= makeQueryClient();
  return browserClient;
}

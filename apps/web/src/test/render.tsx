/** Rendering helpers for component tests. */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderOptions, type RenderResult } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";

/** A QueryClient that never retries, so failures show up at once. */
export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
}

/** Renders `ui` inside a QueryClientProvider (a fresh test client unless one is given). */
export function renderWithClient(
  ui: ReactElement,
  options: RenderOptions & { client?: QueryClient } = {},
): RenderResult & { client: QueryClient } {
  const { client = createTestQueryClient(), ...renderOptions } = options;
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return { ...render(ui, { wrapper, ...renderOptions }), client };
}

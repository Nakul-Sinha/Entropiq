import { useQueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Providers } from "./providers";

function Probe() {
  const client = useQueryClient();
  return <p>stale time {String(client.getDefaultOptions().queries?.staleTime)}</p>;
}

describe("Providers", () => {
  it("renders children with the app QueryClient", () => {
    render(
      <Providers>
        <h1>Child</h1>
        <Probe />
      </Providers>,
    );
    expect(screen.getByRole("heading", { name: "Child" })).toBeInTheDocument();
    expect(screen.getByText("stale time 30000")).toBeInTheDocument();
  });
});

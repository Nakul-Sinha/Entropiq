import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { MOCK_ANALYSIS_ID, MOCK_REPOSITORY_ID } from "@/mocks/ids";
import { setPathname } from "@/test/next-navigation";
import { renderWithClient } from "@/test/render";

import { AnalysisNav } from "./analysis-nav";

vi.mock("next/navigation", () => import("@/test/next-navigation"));

describe("AnalysisNav", () => {
  it("links every section of the current analysis and marks the active one", async () => {
    setPathname(`/analyses/${MOCK_ANALYSIS_ID}/files`);
    renderWithClient(<AnalysisNav />);
    const nav = screen.getByRole("navigation", { name: "Analysis sections" });
    expect(nav).toBeInTheDocument();
    const base = `/analyses/${MOCK_ANALYSIS_ID}`;
    const expected: [string, string][] = [
      ["Overview", base],
      ["Files", `${base}/files`],
      ["Map", `${base}/map`],
      ["Graph", `${base}/graph`],
      ["Plan", `${base}/plan`],
    ];
    for (const [name, href] of expected) {
      expect(screen.getByRole("link", { name })).toHaveAttribute("href", href);
    }
    expect(screen.getByRole("link", { name: "Files" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Overview" })).not.toHaveAttribute("aria-current");

    const trend = await screen.findByRole("link", { name: "Trend" });
    expect(trend).toHaveAttribute("href", `/repositories/${MOCK_REPOSITORY_ID}/trend`);
  });

  it("marks the files section on a file page", () => {
    setPathname(`/analyses/${MOCK_ANALYSIS_ID}/files/1001`);
    renderWithClient(<AnalysisNav />);
    expect(screen.getByRole("link", { name: "Files" })).toHaveAttribute("aria-current", "page");
  });

  it("renders nothing outside an analysis", () => {
    setPathname("/");
    const { container } = renderWithClient(<AnalysisNav />);
    expect(container).toBeEmptyDOMElement();
  });
});

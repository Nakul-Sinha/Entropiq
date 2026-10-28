import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BAND_CLASSES, BANDS } from "@/lib/bands";

import { BandBadge } from "./band-badge";

describe("BandBadge", () => {
  it.each(BANDS)("shows the %s band as text with its colour classes", (band) => {
    const { container } = render(<BandBadge band={band} className="extra" />);
    const badge = container.firstElementChild as HTMLElement;
    expect(badge).toHaveTextContent(new RegExp(`^${band}$`));
    expect(badge).toHaveAttribute("data-band", band);
    for (const cls of BAND_CLASSES[band].split(" ")) expect(badge).toHaveClass(cls);
    expect(badge).toHaveClass("extra");
  });

  it("shows the priority after the band", () => {
    render(<BandBadge band="Critical" priority={98} />);
    expect(screen.getByText("Critical")).toHaveTextContent("Critical98");
    expect(screen.getByText("98")).toBeInTheDocument();
  });
});

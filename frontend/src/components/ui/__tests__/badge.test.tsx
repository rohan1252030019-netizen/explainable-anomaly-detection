import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Badge, SeverityBadge, severityStyle } from "../Badge";

describe("Badge", () => {
  it("renders its children", () => {
    render(<Badge tone="critical">HIGH</Badge>);
    expect(screen.getByText("HIGH")).toBeInTheDocument();
  });

  it("applies the tone class for a known tone", () => {
    render(<Badge tone="good">STABLE</Badge>);
    expect(screen.getByText("STABLE").className).toContain("text-success");
  });
});

describe("SeverityBadge", () => {
  it("upper-cases an arbitrary severity string", () => {
    render(<SeverityBadge severity="high" />);
    expect(screen.getByText("HIGH")).toBeInTheDocument();
  });

  it("falls back to NORMAL for an unknown severity", () => {
    render(<SeverityBadge severity="banana" />);
    expect(screen.getByText("BANANA")).toBeInTheDocument();
    expect(severityStyle("banana").hex).toBe("var(--info)");
  });

  it("hides the dot when showDot is false", () => {
    const { container } = render(<SeverityBadge severity="CRITICAL" showDot={false} />);
    expect(container.querySelector('[aria-hidden="true"]')).toBeNull();
  });
});

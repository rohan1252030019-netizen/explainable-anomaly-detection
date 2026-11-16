"use client";

import React from "react";
import { cn } from "@/lib/cn";

export const severityStyles: Record<
  string,
  { chip: string; dot: string; text: string; bar: string; hex: string }
> = {
  CRITICAL: {
    chip: "bg-critical-soft text-critical border-critical/30",
    dot: "bg-critical",
    text: "text-critical",
    bar: "from-critical to-warn",
    hex: "var(--critical)",
  },
  HIGH: {
    chip: "bg-warn-soft text-warn border-warn/30",
    dot: "bg-warn",
    text: "text-warn",
    bar: "from-warn to-caution",
    hex: "var(--warn)",
  },
  MEDIUM: {
    chip: "bg-caution-soft text-caution border-caution/30",
    dot: "bg-caution",
    text: "text-caution",
    bar: "from-caution to-good",
    hex: "var(--caution)",
  },
  LOW: {
    chip: "bg-good-soft text-good border-good/30",
    dot: "bg-good",
    text: "text-good",
    bar: "from-good to-signal",
    hex: "var(--good)",
  },
  NORMAL: {
    chip: "bg-signal-soft text-signal border-signal/30",
    dot: "bg-signal",
    text: "text-signal",
    bar: "from-signal to-accent",
    hex: "var(--signal)",
  },
};

export function severityStyle(severity?: string) {
  return severityStyles[(severity || "").toUpperCase()] ?? severityStyles.NORMAL;
}

export interface SeverityBadgeProps {
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NORMAL" | string;
  size?: "sm" | "md";
  showDot?: boolean;
  className?: string;
}

export const SeverityBadge: React.FC<SeverityBadgeProps> = ({
  severity,
  size = "md",
  showDot = true,
  className,
}) => {
  const sev = (severity || "").toUpperCase() || "NORMAL";
  const style = severityStyle(sev);

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-semibold uppercase tracking-wide whitespace-nowrap",
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-[11px]",
        style.chip,
        className
      )}
    >
      {showDot && (
        <span
          className={cn("size-1.5 rounded-full", style.dot)}
          aria-hidden="true"
        />
      )}
      {sev}
    </span>
  );
};

export interface BadgeProps {
  tone?: "critical" | "caution" | "warning" | "good" | "neutral";
  className?: string;
  children: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({
  tone = "neutral",
  className,
  children,
}) => {
  const tones = {
    critical: "bg-critical-soft text-critical border-critical/30",
    caution: "bg-caution-soft text-caution border-caution/30",
    warning: "bg-warn-soft text-warn border-warn/30",
    good: "bg-good-soft text-good border-good/30",
    neutral: "bg-surface-elevated text-ink-muted border-edge",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold tracking-wide whitespace-nowrap",
        tones[tone],
        className
      )}
    >
      {children}
    </span>
  );
};

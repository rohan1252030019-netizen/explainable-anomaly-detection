"use client";

import React from "react";
import { cn } from "@/lib/cn";

export const severityStyles: Record<
  string,
  { chip: string; dot: string; text: string; bar: string; hex: string }
> = {
  CRITICAL: {
    chip: "bg-danger-soft text-danger border-danger-line",
    dot: "bg-danger",
    text: "text-danger",
    bar: "from-danger to-warning",
    hex: "var(--danger)",
  },
  HIGH: {
    chip: "bg-warning-soft text-warning border-warning-line",
    dot: "bg-warning",
    text: "text-warning",
    bar: "from-warning to-caution",
    hex: "var(--warning)",
  },
  MEDIUM: {
    chip: "bg-caution-soft text-caution border-caution-line",
    dot: "bg-caution",
    text: "text-caution",
    bar: "from-caution to-success",
    hex: "var(--caution)",
  },
  LOW: {
    chip: "bg-success-soft text-success border-success-line",
    dot: "bg-success",
    text: "text-success",
    bar: "from-success to-info",
    hex: "var(--success)",
  },
  NORMAL: {
    chip: "bg-info-soft text-info border-info-line",
    dot: "bg-info",
    text: "text-info",
    bar: "from-info to-primary",
    hex: "var(--info)",
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
    critical: "bg-danger-soft text-danger border-danger-line",
    caution: "bg-caution-soft text-caution border-caution-line",
    warning: "bg-warning-soft text-warning border-warning-line",
    good: "bg-success-soft text-success border-success-line",
    neutral: "bg-surface-muted text-muted-foreground border-border",
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

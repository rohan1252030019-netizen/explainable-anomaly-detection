"use client";

import React from "react";
import { cn } from "@/lib/cn";

/** Semantic series colours resolved from the active theme. */
export const SERIES = {
  normal: "var(--signal)",
  anomaly: "var(--critical)",
  threshold: "var(--accent)",
  info: "var(--info)",
  grid: "var(--edge)",
  axisText: "var(--ink-faint)",
} as const;

export function ChartFrame({
  title,
  subtitle,
  legend,
  action,
  children,
  className,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  legend?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <figure className={cn("w-full", className)}>
      <figcaption className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-[13px] font-bold tracking-tight text-ink">
            {title}
          </h3>
          {subtitle && (
            <p className="mt-0.5 text-[11px] leading-relaxed text-ink-muted">
              {subtitle}
            </p>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          {legend}
          {action}
        </div>
      </figcaption>
      {children}
    </figure>
  );
}

export function LegendItem({
  color,
  label,
  shape = "square",
  className,
}: {
  color: string;
  label: React.ReactNode;
  shape?: "square" | "dot" | "line";
  className?: string;
}) {
  return (
    <span
      className={cn(
        "flex items-center gap-1.5 text-[10.5px] font-medium text-ink-muted",
        className
      )}
    >
      {shape === "line" ? (
        <span
          className="h-px w-3.5 rounded-full"
          style={{ backgroundColor: color }}
          aria-hidden="true"
        />
      ) : (
        <span
          className={cn(
            "size-2",
            shape === "dot" ? "rounded-full" : "rounded-[2px]"
          )}
          style={{ backgroundColor: color }}
          aria-hidden="true"
        />
      )}
      {label}
    </span>
  );
}

export function ChartEmpty({
  message,
  height = 240,
}: {
  message: string;
  height?: number;
}) {
  return (
    <div
      className="grid place-items-center rounded-lg border border-dashed border-edge px-4 text-center text-xs text-ink-faint"
      style={{ minHeight: height }}
      role="status"
    >
      {message}
    </div>
  );
}

export function TooltipCard({
  x,
  y,
  width,
  height,
  children,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  children: React.ReactNode;
}) {
  const leftPct = Math.max(6, Math.min(94, (x / width) * 100));
  const topPct = Math.max(4, Math.min(72, (y / height) * 100));
  const flip = leftPct > 60;

  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-10 min-w-[9rem] -translate-y-1/2 rounded-lg border border-edge bg-elevated px-3 py-2 text-[11px] text-ink shadow-xl"
      style={{
        left: flip ? undefined : `calc(${leftPct}% + 10px)`,
        right: flip ? `calc(${100 - leftPct}% + 10px)` : undefined,
        top: `${topPct}%`,
      }}
    >
      {children}
    </div>
  );
}

export function TooltipRow({
  label,
  value,
  color,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  color?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="flex items-center gap-1.5 text-ink-muted">
        {color && (
          <span
            className="size-1.5 rounded-full"
            style={{ backgroundColor: color }}
            aria-hidden="true"
          />
        )}
        {label}
      </span>
      <span className="font-bold tabular text-ink">{value}</span>
    </div>
  );
}

"use client";

import React, { useState } from "react";
import { cn } from "@/lib/cn";
import { severityStyle } from "@/components/ui/Badge";
import { ChartEmpty, ChartFrame } from "./ChartFrame";

interface SeverityDonutChartProps {
  counts: {
    CRITICAL: number;
    HIGH: number;
    MEDIUM: number;
    LOW: number;
  };
}

const ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const;

export const SeverityDonutChart: React.FC<SeverityDonutChartProps> = ({
  counts,
}) => {
  const [active, setActive] = useState<string | null>(null);

  const items = ORDER.map((key) => ({
    key,
    label: key.charAt(0) + key.slice(1).toLowerCase(),
    count: counts?.[key] ?? 0,
    style: severityStyle(key),
  }));

  const total = items.reduce((acc, i) => acc + i.count, 0);

  if (total === 0) {
    return <ChartEmpty message="No anomalies to break down." height={180} />;
  }

  const size = 168;
  const strokeWidth = 16;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  // Pre-compute arc geometry so no variable is mutated during render.
  const fractions = items.map((item) => item.count / total);
  const arcs = items.map((item, i) => ({
    key: item.key,
    pct: fractions[i],
    dash: fractions[i] * circumference,
    offset: -fractions.slice(0, i).reduce((a, b) => a + b, 0) * circumference,
  }));

  return (
    <ChartFrame title="Severity Classification" subtitle="Score quantile tiers">
      <div className="flex flex-col items-center gap-5 sm:flex-row sm:justify-center">
        <div className="relative shrink-0" style={{ width: size, height: size }}>
          <svg
            width={size}
            height={size}
            viewBox={`0 0 ${size} ${size}`}
            className="-rotate-90"
            role="img"
            aria-label={`Severity breakdown of ${total} anomalies`}
          >
            {arcs.map((arc) => {
              const item = items.find((i) => i.key === arc.key)!;
              if (arc.pct <= 0) return null;

              return (
                <circle
                  key={arc.key}
                  cx={size / 2}
                  cy={size / 2}
                  r={radius}
                  fill="none"
                  stroke={item.style.hex}
                  strokeWidth={active === arc.key ? strokeWidth + 4 : strokeWidth}
                  strokeDasharray={`${arc.dash} ${circumference - arc.dash}`}
                  strokeDashoffset={arc.offset}
                  opacity={active && active !== item.key ? 0.35 : 1}
                  style={{ transition: "all 200ms cubic-bezier(0.16,1,0.3,1)" }}
                  onMouseEnter={() => setActive(item.key)}
                  onMouseLeave={() => setActive(null)}
                />
              );
            })}
          </svg>

          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-extrabold tabular text-foreground">
              {active ? items.find((i) => i.key === active)?.count : total}
            </span>
            <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-subtle-foreground">
              {active ?? "Anomalies"}
            </span>
          </div>
        </div>

        <ul className="w-full min-w-[150px] space-y-1.5">
          {items.map((item) => (
            <li key={item.key}>
              <button
                onMouseEnter={() => setActive(item.key)}
                onMouseLeave={() => setActive(null)}
                onFocus={() => setActive(item.key)}
                onBlur={() => setActive(null)}
                className={cn(
                  "flex w-full items-center justify-between gap-4 rounded-md px-2 py-1 text-xs transition-colors",
                  active === item.key ? "bg-surface-muted" : "hover:bg-surface-muted"
                )}
              >
                <span className="flex items-center gap-2 font-medium text-muted-foreground">
                  <span
                    className="size-2.5 rounded-full"
                    style={{ backgroundColor: item.style.hex }}
                    aria-hidden="true"
                  />
                  {item.label}
                </span>
                <span className="font-bold tabular text-foreground">
                  {item.count}
                  <span className="ml-1 text-[10px] font-normal text-subtle-foreground">
                    {((item.count / total) * 100).toFixed(0)}%
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </ChartFrame>
  );
};

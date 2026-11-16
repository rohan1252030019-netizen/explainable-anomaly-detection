"use client";

import React from "react";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { FeatureComparisonItem } from "@/lib/api";
import { cn } from "@/lib/cn";

interface ObservedVsTypicalRangeProps {
  items: FeatureComparisonItem[];
}

const fmt = (value: unknown) =>
  typeof value === "number"
    ? value.toLocaleString(undefined, { maximumFractionDigits: 3 })
    : String(value ?? "—");

function RangeTrack({ item }: { item: FeatureComparisonItem }) {
  const observed = typeof item.observed === "number" ? item.observed : null;
  const median = item.typical_median;
  if (observed === null || typeof median !== "number") return null;

  const low = item.typical_min ?? Math.min(observed, median);
  const high = item.typical_max ?? Math.max(observed, median);
  const domainLow = Math.min(low, observed, median);
  const domainHigh = Math.max(high, observed, median);
  const span = domainHigh - domainLow || 1;
  const padding = span * 0.08;

  const toPct = (value: number) =>
    ((value - (domainLow - padding)) / (span + padding * 2)) * 100;

  const bandStart = Math.max(0, Math.min(100, toPct(low)));
  const bandEnd = Math.max(0, Math.min(100, toPct(high)));
  const medianPos = Math.max(0, Math.min(100, toPct(median)));
  const observedPos = Math.max(0, Math.min(100, toPct(observed)));
  const outside = observed < low || observed > high;

  return (
    <div className="space-y-1.5">
      <div
        className="relative h-3 rounded-full bg-inset"
        role="img"
        aria-label={`Reference range ${fmt(low)} to ${fmt(
          high
        )}, median ${fmt(median)}, this record ${fmt(observed)}`}
      >
        <span
          className="absolute inset-y-0 rounded-full border border-edge bg-edge/60"
          style={{
            left: `${bandStart}%`,
            width: `${Math.max(1.5, bandEnd - bandStart)}%`,
          }}
          aria-hidden="true"
        />
        <span
          className="absolute inset-y-0 w-px bg-ink-faint"
          style={{ left: `${medianPos}%` }}
          aria-hidden="true"
        />
        <span
          className={cn(
            "absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface",
            outside ? "bg-critical" : "bg-signal"
          )}
          style={{ left: `${observedPos}%` }}
          aria-hidden="true"
        />
      </div>
      <div className="flex justify-between text-[9px] tabular text-ink-faint">
        <span>{fmt(domainLow)}</span>
        <span>
          reference {fmt(low)}–{fmt(high)}
          {item.typical_iqr ? ` · IQR ${fmt(item.typical_iqr)}` : ""}
        </span>
        <span>{fmt(domainHigh)}</span>
      </div>
    </div>
  );
}

export const ObservedVsTypicalRange: React.FC<ObservedVsTypicalRangeProps> = ({
  items,
}) => {
  if (!items || items.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-[10px] font-bold uppercase tracking-[0.14em] text-ink-faint">
          Observed value vs reference distribution
        </h3>
        <p className="text-[10px] text-ink-faint">
          Marker = this record · bar = reference min–max
        </p>
      </div>

      <ul className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {items.map((item, index) => {
          const isAbove = item.direction.toLowerCase().includes("above");

          return (
            <li
              key={`${item.feature}-${index}`}
              className="panel-flush space-y-3 p-4 transition-colors hover:border-edge-strong"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-xs font-semibold text-ink">
                  {item.feature}
                </span>
                <span
                  className={cn(
                    "shrink-0 rounded border px-1.5 py-0.5 text-[9.5px] font-bold",
                    isAbove
                      ? "border-critical/30 bg-critical-soft text-critical"
                      : "border-signal/30 bg-signal-soft text-signal"
                  )}
                >
                  {item.deviation}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 border-t border-edge pt-3">
                <div>
                  <div className="text-[9px] font-semibold uppercase tracking-[0.1em] text-ink-faint">
                    This record
                  </div>
                  <div
                    className={cn(
                      "mt-0.5 text-base font-bold tabular",
                      isAbove ? "text-critical" : "text-signal"
                    )}
                  >
                    {fmt(item.observed)}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[9px] font-semibold uppercase tracking-[0.1em] text-ink-faint">
                    Reference median
                  </div>
                  <div className="mt-0.5 text-base font-bold tabular text-ink-muted">
                    {fmt(item.typical_median)}
                  </div>
                </div>
              </div>

              <RangeTrack item={item} />

              <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-ink-muted">
                <span className="flex items-center gap-1">
                  {isAbove ? (
                    <ArrowUpRight size={10} aria-hidden="true" />
                  ) : (
                    <ArrowDownRight size={10} aria-hidden="true" />
                  )}
                  {item.direction}
                </span>
                <span>Impact: {item.contribution}</span>
              </div>

              {item.interpretation && (
                <p className="text-[11px] leading-relaxed text-ink-muted">
                  {item.interpretation}
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
};

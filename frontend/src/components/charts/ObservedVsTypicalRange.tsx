"use client";

import React from "react";
import { FeatureComparisonItem } from "@/lib/api";

interface ObservedVsTypicalRangeProps {
  items: FeatureComparisonItem[];
}

export const ObservedVsTypicalRange: React.FC<ObservedVsTypicalRangeProps> = ({ items }) => {
  if (!items || items.length === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
        Observed Value vs Baseline Normal Distribution
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {items.map((item, i) => {
          const isAbove = item.direction.toLowerCase().includes("above");
          const devColor = isAbove ? "text-rose-500" : "text-blue-500";
          const barColor = isAbove ? "bg-rose-500" : "bg-blue-500";

          const obsVal = typeof item.observed === "number" ? item.observed.toLocaleString() : String(item.observed ?? "—");
          const medVal = typeof item.typical_median === "number" ? item.typical_median.toLocaleString() : String(item.typical_median ?? "—");

          return (
            <div
              key={i}
              className="saas-card p-3.5 space-y-2 border border-slate-200 dark:border-slate-800"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wide">
                  {item.feature}
                </span>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${isAbove ? "bg-rose-500/10 text-rose-400" : "bg-blue-500/10 text-blue-400"}`}>
                  {item.deviation}
                </span>
              </div>

              {/* Observed vs Median Value row */}
              <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-100 dark:border-slate-800/80">
                <div>
                  <div className="text-[10px] text-slate-400 uppercase font-semibold">
                    This Record
                  </div>
                  <div className={`text-base font-extrabold ${devColor}`}>
                    {obsVal}
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-[10px] text-slate-400 uppercase font-semibold">
                    Typical Median
                  </div>
                  <div className="text-base font-extrabold text-slate-600 dark:text-slate-300">
                    {medVal}
                  </div>
                </div>
              </div>

              {/* Relative indicator bar */}
              <div className="space-y-1 pt-1">
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>Deviation: {item.direction}</span>
                  <span>Impact: {item.contribution}</span>
                </div>
                <div className="h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                  <div className={`h-full rounded-full ${barColor}`} style={{ width: "75%" }} />
                </div>
              </div>

              {item.interpretation && (
                <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed pt-1">
                  {item.interpretation}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

"use client";

import React from "react";

interface FeatureContribution {
  feature: string;
  combined_score?: number;
  deviation_direction?: string;
  robust_z_score?: number;
  contribution?: string;
  deviation?: string;
  direction?: string;
}

interface FeatureContributionChartProps {
  features: FeatureContribution[];
  title?: string;
}

export const FeatureContributionChart: React.FC<FeatureContributionChartProps> = ({
  features,
  title = "Feature Contributions (Why was it flagged?)",
}) => {
  if (!features || features.length === 0) {
    return (
      <div className="text-xs text-slate-400 py-6 text-center">
        No feature contribution breakdown available.
      </div>
    );
  }

  const sorted = [...features].sort((a, b) => {
    const scoreA = a.combined_score ?? 0;
    const scoreB = b.combined_score ?? 0;
    return scoreB - scoreA;
  });

  const maxVal = Math.max(...sorted.map((f) => f.combined_score ?? Math.abs(f.robust_z_score ?? 1)), 1);

  return (
    <div className="space-y-3">
      {title && (
        <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
          {title}
        </div>
      )}

      <div className="space-y-2.5">
        {sorted.map((item, i) => {
          const score = item.combined_score ?? Math.abs(item.robust_z_score ?? 0);
          const pct = Math.min(100, Math.max(10, (score / maxVal) * 100));
          const dir = item.deviation_direction || item.direction || "above";
          const isAbove = dir.toLowerCase().includes("above");

          return (
            <div key={i} className="text-xs space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-800 dark:text-slate-200">
                  {item.feature}
                </span>
                <div className="flex items-center gap-2">
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      isAbove
                        ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                        : "bg-blue-500/10 text-blue-400 border border-blue-500/20"
                    }`}
                  >
                    {isAbove ? "↑ Above normal" : "↓ Below normal"}
                  </span>
                  {item.deviation && (
                    <span className="text-[11px] text-slate-400 font-mono">
                      {item.deviation}
                    </span>
                  )}
                </div>
              </div>

              {/* Progress bar */}
              <div className="h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isAbove
                      ? "bg-gradient-to-r from-rose-500 to-orange-400"
                      : "bg-gradient-to-r from-blue-500 to-indigo-400"
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

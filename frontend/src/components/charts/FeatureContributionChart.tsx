"use client";

import React from "react";
import { motion } from "framer-motion";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { ChartEmpty } from "./ChartFrame";
import { cn } from "@/lib/cn";

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
  title = "Feature contributions",
}) => {
  if (!features || features.length === 0) {
    return (
      <ChartEmpty message="No feature contribution breakdown available." height={140} />
    );
  }

  const sorted = [...features].sort(
    (a, b) => (b.combined_score ?? 0) - (a.combined_score ?? 0)
  );

  const maxVal = Math.max(
    ...sorted.map((feature) =>
      Math.abs(feature.combined_score ?? feature.robust_z_score ?? 0)
    ),
    0.0001
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-[10px] font-bold uppercase tracking-[0.14em] text-ink-faint">
          {title}
        </h3>
        <p className="text-[10px] text-ink-faint">
          Combined score, scaled to the largest contributor
        </p>
      </div>

      <ul className="space-y-3">
        {sorted.map((item, index) => {
          const score = Math.abs(
            item.combined_score ?? item.robust_z_score ?? 0
          );
          const pct = Math.min(100, Math.max(2, (score / maxVal) * 100));
          const direction =
            item.deviation_direction || item.direction || "above";
          const isAbove = direction.toLowerCase().includes("above");

          return (
            <li key={`${item.feature}-${index}`} className="space-y-1.5">
              <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5">
                <span className="min-w-0 truncate text-xs font-semibold text-ink">
                  {item.feature}
                </span>
                <span className="flex shrink-0 items-center gap-2">
                  {item.robust_z_score !== undefined &&
                    item.robust_z_score !== null && (
                      <span className="font-mono text-[10px] tabular text-ink-faint">
                        z {item.robust_z_score.toFixed(2)}
                      </span>
                    )}
                  <span className="font-mono text-[10px] font-bold tabular text-ink-muted">
                    {score.toFixed(3)}
                  </span>
                  <span
                    className={cn(
                      "inline-flex items-center gap-1 rounded border px-1.5 py-px text-[9.5px] font-bold",
                      isAbove
                        ? "border-critical/30 bg-critical-soft text-critical"
                        : "border-signal/30 bg-signal-soft text-signal"
                    )}
                  >
                    {isAbove ? (
                      <ArrowUpRight size={10} aria-hidden="true" />
                    ) : (
                      <ArrowDownRight size={10} aria-hidden="true" />
                    )}
                    {isAbove ? "Above" : "Below"} typical
                  </span>
                </span>
              </div>

              <div
                className="h-2 overflow-hidden rounded-full bg-inset"
                role="meter"
                aria-valuemin={0}
                aria-valuemax={Number(maxVal.toFixed(3))}
                aria-valuenow={Number(score.toFixed(3))}
                aria-label={`${item.feature} contribution`}
              >
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{
                    duration: 0.6,
                    ease: [0.16, 1, 0.3, 1],
                    delay: index * 0.04,
                  }}
                  className={cn(
                    "h-full rounded-full",
                    isAbove
                      ? "bg-gradient-to-r from-critical/70 to-critical"
                      : "bg-gradient-to-r from-signal/70 to-signal"
                  )}
                />
              </div>

              {item.deviation && (
                <p className="text-[10px] leading-relaxed text-ink-faint">
                  {item.deviation}
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
};

"use client";

import React, { useState } from "react";
import { ChartEmpty, ChartFrame } from "./ChartFrame";
import { cn } from "@/lib/cn";

interface CorrelationHeatmapProps {
  features: string[];
  matrix: number[][];
}

function cellStyle(val: number) {
  const strength = Math.min(1, Math.abs(val));
  const series = val >= 0 ? "var(--info)" : "var(--warn)";
  return {
    background: `color-mix(in srgb, ${series} ${Math.round(
      10 + strength * 38
    )}%, transparent)`,
  };
}

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  features,
  matrix,
}) => {
  const [hoveredCell, setHoveredCell] = useState<{
    f1: string;
    f2: string;
    val: number;
  } | null>(null);

  if (!features || features.length === 0 || !matrix || matrix.length === 0) {
    return <ChartEmpty message="No correlation matrix data available." />;
  }

  const n = features.length;

  return (
    <ChartFrame
      title="Feature correlation"
      subtitle={`Pearson correlation across ${n} preprocessed features`}
      legend={
        <span className="flex items-center gap-2 text-[10px] text-ink-faint">
          <span className="tabular">−1</span>
          <span
            className="h-2 w-20 rounded-full"
            style={{
              background:
                "linear-gradient(90deg, color-mix(in srgb, var(--warn) 48%, transparent), var(--inset), color-mix(in srgb, var(--info) 48%, transparent))",
            }}
            aria-hidden="true"
          />
          <span className="tabular">+1</span>
        </span>
      }
    >
      <p className="sr-only">
        Correlation matrix of {n} features. Positive correlations are shown in
        blue and negative correlations in orange; darker cells indicate a
        stronger relationship.
      </p>
      <div className="data-scroll overflow-x-auto pb-2">
        <div
          className="grid min-w-full gap-px"
          style={{
            gridTemplateColumns: `minmax(96px, 128px) repeat(${n}, minmax(38px, 1fr))`,
          }}
        >
          <div aria-hidden="true" />

          {features.map((feat, i) => (
            <div
              key={`h-${i}`}
              className="flex h-24 items-end justify-center overflow-hidden pb-1"
              title={feat}
            >
              <span className="truncate text-[10px] font-medium text-ink-faint [writing-mode:vertical-rl] rotate-180">
                {feat}
              </span>
            </div>
          ))}

          {features.map((rowFeat, r) => (
            <React.Fragment key={r}>
              <div
                className="flex items-center truncate pr-2 text-[10px] font-semibold text-ink-muted"
                title={rowFeat}
              >
                {rowFeat}
              </div>

              {features.map((colFeat, c) => {
                const val = matrix[r]?.[c] ?? 0;
                const strong = Math.abs(val) > 0.5;
                return (
                  <button
                    key={c}
                    type="button"
                    aria-label={`${rowFeat} correlated with ${colFeat}: ${val.toFixed(3)}`}
                    onMouseEnter={() =>
                      setHoveredCell({ f1: rowFeat, f2: colFeat, val })
                    }
                    onMouseLeave={() => setHoveredCell(null)}
                    onFocus={() => setHoveredCell({ f1: rowFeat, f2: colFeat, val })}
                    onBlur={() => setHoveredCell(null)}
                    className={cn(
                      "h-9 border border-edge/60 font-mono text-[10px] tabular transition-transform",
                      "hover:z-10 hover:scale-110 hover:border-accent",
                      strong ? "font-bold text-ink" : "text-ink-muted"
                    )}
                    style={cellStyle(val)}
                  >
                    {val.toFixed(2)}
                  </button>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      <p
        className="mt-2 h-8 text-xs text-ink-muted"
        role="status"
        aria-live="polite"
      >
        {hoveredCell ? (
          <>
            <span className="font-semibold text-ink">{hoveredCell.f1}</span>
            {" × "}
            <span className="font-semibold text-ink">{hoveredCell.f2}</span>
            {": "}
            <span
              className="font-bold tabular"
              style={{
                color: hoveredCell.val >= 0 ? "var(--info)" : "var(--warn)",
              }}
            >
              {hoveredCell.val.toFixed(3)}
            </span>
          </>
        ) : (
          <span className="text-ink-faint">
            Hover a cell to inspect the coefficient.
          </span>
        )}
      </p>
    </ChartFrame>
  );
};

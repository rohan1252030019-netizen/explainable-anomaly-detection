"use client";

import React, { useState } from "react";
import { ChartEmpty, ChartFrame } from "./ChartFrame";
import { cn } from "@/lib/cn";

interface CorrelationHeatmapProps {
  features: string[];
  matrix: number[][];
}

type RGB = [number, number, number];

/**
 * Light-friendly diverging scale.
 *
 * A single hue pair is used for the whole matrix so the sign of a
 * correlation is readable at a glance: slate-50 at zero, blue-700 for a
 * positive relationship and orange-700 for a negative one. Both endpoints
 * are dark enough to carry white label text, and the midpoint stays pale
 * enough for dark label text, so the label colour is switched on the
 * luminance of the cell it sits in.
 */
const NEUTRAL: RGB = [241, 245, 249];
const POSITIVE: RGB = [29, 78, 216];
const NEGATIVE: RGB = [194, 65, 12];

const LABEL_DARK = "#0f172a";
const LABEL_LIGHT = "#ffffff";
const DARK_INK: RGB = [15, 23, 42];

function mix(base: RGB, target: RGB, amount: number): RGB {
  return [
    Math.round(base[0] + (target[0] - base[0]) * amount),
    Math.round(base[1] + (target[1] - base[1]) * amount),
    Math.round(base[2] + (target[2] - base[2]) * amount),
  ];
}

/** WCAG relative luminance. */
function luminance([r, g, b]: RGB): number {
  const channel = (value: number) => {
    const c = value / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  return (
    0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
  );
}

/**
 * Picks whichever of the two label inks has the better WCAG contrast
 * against the cell background.
 */
function labelColor(background: RGB): string {
  const bg = luminance(background);
  const onLight = 1.05 / (bg + 0.05);
  const onDark = (bg + 0.05) / (luminance(DARK_INK) + 0.05);
  return onLight >= onDark ? LABEL_LIGHT : LABEL_DARK;
}

function cellStyle(val: number) {
  const strength = Math.min(1, Math.abs(val));
  const target = val >= 0 ? POSITIVE : NEGATIVE;
  // Eased so mid-strength correlations stay visible instead of washing out.
  const amount = Math.pow(strength, 0.75);
  const rgb = mix(NEUTRAL, target, amount);
  return {
    background: `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`,
    color: labelColor(rgb),
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
        <span className="flex items-center gap-2 text-[10px] text-subtle-foreground">
          <span className="tabular">−1</span>
          <span
            className="h-2 w-20 rounded-full"
            style={{
              background: `linear-gradient(90deg, rgb(${NEGATIVE.join(", ")}), rgb(${NEUTRAL.join(", ")}), rgb(${POSITIVE.join(", ")}))`,
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
              <span className="truncate text-[10px] font-medium text-subtle-foreground [writing-mode:vertical-rl] rotate-180">
                {feat}
              </span>
            </div>
          ))}

          {features.map((rowFeat, r) => (
            <React.Fragment key={r}>
              <div
                className="flex items-center truncate pr-2 text-[10px] font-semibold text-muted-foreground"
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
                      "h-9 border border-border/60 font-mono text-[10px] tabular transition-transform",
                      "hover:z-10 hover:scale-110 hover:border-primary",
                      strong && "font-bold"
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
        className="mt-2 h-8 text-xs text-muted-foreground"
        role="status"
        aria-live="polite"
      >
        {hoveredCell ? (
          <>
            <span className="font-semibold text-foreground">{hoveredCell.f1}</span>
            {" × "}
            <span className="font-semibold text-foreground">{hoveredCell.f2}</span>
            {": "}
            <span
              className="font-bold tabular"
              style={{
                color: hoveredCell.val >= 0 ? "var(--info)" : "var(--warning)",
              }}
            >
              {hoveredCell.val.toFixed(3)}
            </span>
          </>
        ) : (
          <span className="text-subtle-foreground">
            Hover a cell to inspect the coefficient.
          </span>
        )}
      </p>
    </ChartFrame>
  );
};

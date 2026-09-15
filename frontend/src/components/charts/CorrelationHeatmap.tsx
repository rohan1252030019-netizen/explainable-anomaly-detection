"use client";

import React, { useState } from "react";

interface CorrelationHeatmapProps {
  features: string[];
  matrix: number[][];
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
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-400">
        No correlation matrix data available.
      </div>
    );
  }

  const n = features.length;

  return (
    <div className="w-full overflow-x-auto">
      <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-2">
        Feature Correlation Matrix
      </div>

      <div className="inline-block min-w-full">
        <div className="grid" style={{ gridTemplateColumns: `100px repeat(${n}, minmax(36px, 1fr))` }}>
          {/* Header corner */}
          <div className="p-1" />

          {/* Top header labels */}
          {features.map((feat, i) => (
            <div
              key={i}
              className="p-1 text-[10px] font-medium text-slate-400 truncate -rotate-45 origin-bottom-left h-20 flex items-end"
            >
              {feat}
            </div>
          ))}

          {/* Rows */}
          {features.map((rowFeat, r) => (
            <React.Fragment key={r}>
              {/* Row Label */}
              <div className="p-1 text-[10px] font-medium text-slate-400 truncate flex items-center">
                {rowFeat}
              </div>

              {/* Cells */}
              {features.map((colFeat, c) => {
                const val = matrix[r]?.[c] ?? 0;
                // Color scaling: positive = red/orange, negative = blue, 0 = neutral
                let bg = "bg-slate-800/40";
                if (val > 0.5) bg = "bg-rose-500/80 text-white";
                else if (val > 0.2) bg = "bg-rose-500/40 text-slate-200";
                else if (val < -0.5) bg = "bg-blue-600/80 text-white";
                else if (val < -0.2) bg = "bg-blue-600/40 text-slate-200";

                return (
                  <div
                    key={c}
                    onMouseEnter={() =>
                      setHoveredCell({ f1: rowFeat, f2: colFeat, val })
                    }
                    onMouseLeave={() => setHoveredCell(null)}
                    className={`h-9 flex items-center justify-center text-[10px] font-mono border border-slate-200 dark:border-slate-800/60 transition-colors cursor-pointer ${bg}`}
                  >
                    {val.toFixed(2)}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      {hoveredCell && (
        <div className="mt-3 text-xs text-slate-400 bg-slate-900 border border-slate-800 p-2 rounded max-w-sm">
          <span className="font-semibold text-slate-200">{hoveredCell.f1}</span> ×{" "}
          <span className="font-semibold text-slate-200">{hoveredCell.f2}</span>:{" "}
          <span className="font-bold text-indigo-400">{hoveredCell.val.toFixed(3)}</span>
        </div>
      )}
    </div>
  );
};

"use client";

import React, { useState } from "react";
import { ScoreTimelinePoint } from "@/lib/api";

interface ObservationTimelineChartProps {
  data: ScoreTimelinePoint[];
  threshold?: number;
}

export const ObservationTimelineChart: React.FC<ObservationTimelineChartProps> = ({
  data,
  threshold = 0.5,
}) => {
  const [hovered, setHovered] = useState<ScoreTimelinePoint | null>(null);

  if (!data || data.length === 0) {
    return (
      <div className="h-52 flex items-center justify-center text-xs text-slate-400">
        No observation timeline available.
      </div>
    );
  }

  const width = 800;
  const height = 180;
  const padLeft = 40;
  const padRight = 20;
  const padTop = 15;
  const padBottom = 25;

  const chartWidth = width - padLeft - padRight;
  const chartHeight = height - padTop - padBottom;

  const minIdx = 0;
  const maxIdx = Math.max(...data.map((d) => d.index), 1);

  const thresholdY = padTop + chartHeight * (1 - threshold);

  return (
    <div className="w-full relative">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
          Anomaly Score by Observation Sequence
        </span>
        <div className="flex items-center gap-3 text-[11px] text-slate-400">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-blue-500/50" /> Normal
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-rose-500" /> Flagged Outlier
          </span>
        </div>
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
          {/* Y Grid lines */}
          {[0, 0.5, 1.0].map((val, i) => {
            const y = padTop + chartHeight * (1 - val);
            return (
              <g key={i}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke="currentColor"
                  className="text-slate-200 dark:text-slate-800"
                  strokeDasharray="2 2"
                />
                <text x={padLeft - 6} y={y + 3} textAnchor="end" className="text-[9px] fill-slate-400">
                  {val.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Threshold line */}
          <line
            x1={padLeft}
            y1={thresholdY}
            x2={width - padRight}
            y2={thresholdY}
            stroke="#6366F1"
            strokeWidth="1.5"
            strokeDasharray="3 3"
          />

          {/* Points */}
          {data.map((pt, i) => {
            const x = padLeft + (pt.index / maxIdx) * chartWidth;
            const y = padTop + chartHeight * (1 - Math.max(0, Math.min(1, pt.score)));

            if (pt.is_anomaly) {
              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r="3.5"
                  className="fill-rose-500 stroke-white dark:stroke-slate-900 cursor-pointer hover:r-5 transition-all"
                  onMouseEnter={() => setHovered(pt)}
                  onMouseLeave={() => setHovered(null)}
                />
              );
            }

            return (
              <circle
                key={i}
                cx={x}
                cy={y}
                r="1.8"
                className="fill-blue-500/35 hover:fill-blue-400 cursor-pointer"
                onMouseEnter={() => setHovered(pt)}
                onMouseLeave={() => setHovered(null)}
              />
            );
          })}
        </svg>

        {/* Hover tooltip */}
        {hovered && (
          <div className="absolute top-2 right-4 px-3 py-1.5 rounded bg-slate-900 border border-slate-700 text-[11px] shadow-md text-slate-200 pointer-events-none">
            <div>Observation Index: #{hovered.index}</div>
            <div>Score: {hovered.score.toFixed(4)}</div>
            <div className={hovered.is_anomaly ? "text-rose-400 font-bold" : "text-blue-400"}>
              {hovered.is_anomaly ? `Anomaly (${hovered.severity})` : "Normal"}
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-between text-[10px] text-slate-400 mt-1 px-10">
        <span>Obs #0</span>
        <span>Obs #{maxIdx}</span>
      </div>
    </div>
  );
};

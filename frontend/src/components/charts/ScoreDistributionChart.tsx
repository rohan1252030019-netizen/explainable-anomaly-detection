"use client";

import React, { useState } from "react";
import { ScoreHistBin } from "@/lib/api";

interface ScoreDistributionChartProps {
  data: ScoreHistBin[];
  threshold?: number;
  modelName?: string;
}

export const ScoreDistributionChart: React.FC<ScoreDistributionChartProps> = ({
  data,
  threshold = 0.5,
  modelName = "",
}) => {
  const [hoveredBin, setHoveredBin] = useState<ScoreHistBin | null>(null);

  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-400">
        No score distribution data available.
      </div>
    );
  }

  const width = 600;
  const height = 240;
  const padLeft = 40;
  const padBottom = 30;
  const padTop = 20;
  const padRight = 20;

  const chartWidth = width - padLeft - padRight;
  const chartHeight = height - padTop - padBottom;

  const maxCount = Math.max(
    ...data.map((d) => d.normal_count + d.anomaly_count),
    1
  );

  const binWidth = chartWidth / data.length;
  const thresholdX = padLeft + threshold * chartWidth;

  return (
    <div className="w-full relative">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
          Anomaly Score Histogram {modelName && `· ${modelName}`}
        </div>
        <div className="flex items-center gap-4 text-[11px] text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-blue-500/70" /> Normal
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded bg-rose-500/90" /> Anomaly
          </span>
          <span className="flex items-center gap-1.5 font-medium text-indigo-400">
            <span className="w-3 h-0.5 bg-indigo-400" /> Threshold ({(threshold).toFixed(3)})
          </span>
        </div>
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
            const y = padTop + chartHeight * (1 - p);
            const val = Math.round(maxCount * p);
            return (
              <g key={i}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke="currentColor"
                  className="text-slate-200 dark:text-slate-800"
                  strokeDasharray="3 3"
                />
                <text
                  x={padLeft - 8}
                  y={y + 3}
                  textAnchor="end"
                  className="text-[9px] fill-slate-400"
                >
                  {val}
                </text>
              </g>
            );
          })}

          {/* Histogram stacked bars */}
          {data.map((bin, i) => {
            const x = padLeft + i * binWidth;
            const normH = (bin.normal_count / maxCount) * chartHeight;
            const anomH = (bin.anomaly_count / maxCount) * chartHeight;

            return (
              <g
                key={i}
                onMouseEnter={() => setHoveredBin(bin)}
                onMouseLeave={() => setHoveredBin(null)}
                className="cursor-pointer group"
              >
                {/* Normal count bar */}
                {normH > 0 && (
                  <rect
                    x={x + 1}
                    y={padTop + chartHeight - normH}
                    width={Math.max(1, binWidth - 2)}
                    height={normH}
                    className="fill-blue-500/60 hover:fill-blue-500/80 transition-colors"
                  />
                )}
                {/* Anomaly count bar stacked on top */}
                {anomH > 0 && (
                  <rect
                    x={x + 1}
                    y={padTop + chartHeight - normH - anomH}
                    width={Math.max(1, binWidth - 2)}
                    height={anomH}
                    className="fill-rose-500/85 hover:fill-rose-500 transition-colors"
                  />
                )}
              </g>
            );
          })}

          {/* Threshold line */}
          <line
            x1={thresholdX}
            y1={padTop}
            x2={thresholdX}
            y2={padTop + chartHeight}
            stroke="#6366F1"
            strokeWidth="2"
            strokeDasharray="4 4"
          />

          {/* X Axis Ticks */}
          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick, i) => {
            const x = padLeft + tick * chartWidth;
            return (
              <text
                key={i}
                x={x}
                y={padTop + chartHeight + 16}
                textAnchor="middle"
                className="text-[10px] fill-slate-400"
              >
                {tick.toFixed(1)}
              </text>
            );
          })}
        </svg>

        {/* Hover Tooltip */}
        {hoveredBin && (
          <div className="absolute top-2 left-12 px-3 py-1.5 rounded bg-slate-900 border border-slate-700 text-[11px] shadow-lg text-slate-200 pointer-events-none">
            <div>Score: {hoveredBin.bin_start} – {hoveredBin.bin_end}</div>
            <div className="text-blue-400">Normal: {hoveredBin.normal_count}</div>
            <div className="text-rose-400 font-semibold">Anomalies: {hoveredBin.anomaly_count}</div>
          </div>
        )}
      </div>

      <div className="text-center text-[10px] text-slate-400 mt-1">
        Normalised Anomaly Score (0.0 = Normal · 1.0 = Highly Anomalous)
      </div>
    </div>
  );
};

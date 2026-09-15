"use client";

import React, { useState } from "react";
import { PcaAnalyticsResponse } from "@/lib/api";

interface PcaScatterChartProps {
  data: PcaAnalyticsResponse;
}

export const PcaScatterChart: React.FC<PcaScatterChartProps> = ({ data }) => {
  const [hovered, setHovered] = useState<any | null>(null);

  if (!data || !data.points || data.points.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-xs text-slate-400">
        No PCA projection data available.
      </div>
    );
  }

  const width = 640;
  const height = 320;
  const pad = 40;

  const chartWidth = width - pad * 2;
  const chartHeight = height - pad * 2;

  const pc1Vals = data.points.map((p) => p.pc1);
  const pc2Vals = data.points.map((p) => p.pc2);

  const minX = Math.min(...pc1Vals);
  const maxX = Math.max(...pc1Vals);
  const minY = Math.min(...pc2Vals);
  const maxY = Math.max(...pc2Vals);

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const varX = data.variance_explained?.[0]
    ? (data.variance_explained[0] * 100).toFixed(1)
    : "PC1";
  const varY = data.variance_explained?.[1]
    ? (data.variance_explained[1] * 100).toFixed(1)
    : "PC2";

  return (
    <div className="w-full relative">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
          2D PCA Projection (Cluster Separation)
        </span>
        <div className="flex items-center gap-4 text-[11px] text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500/50" /> Normal cluster
          </span>
          <span className="flex items-center gap-1.5 font-semibold text-rose-400">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" /> Anomalies
          </span>
        </div>
      </div>

      <div className="relative">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
          {/* Axis lines */}
          <line
            x1={pad}
            y1={height - pad}
            x2={width - pad}
            y2={height - pad}
            stroke="currentColor"
            className="text-slate-300 dark:text-slate-700"
          />
          <line
            x1={pad}
            y1={pad}
            x2={pad}
            y2={height - pad}
            stroke="currentColor"
            className="text-slate-300 dark:text-slate-700"
          />

          {/* Scatter points */}
          {data.points.map((pt, i) => {
            const x = pad + ((pt.pc1 - minX) / rangeX) * chartWidth;
            const y = height - pad - ((pt.pc2 - minY) / rangeY) * chartHeight;

            if (pt.is_anomaly) {
              return (
                <circle
                  key={i}
                  cx={x}
                  cy={y}
                  r="4.5"
                  className="fill-rose-500 stroke-white dark:stroke-slate-900 cursor-pointer hover:r-6 transition-all"
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
                r="2.5"
                className="fill-blue-500/35 hover:fill-blue-400 cursor-pointer"
                onMouseEnter={() => setHovered(pt)}
                onMouseLeave={() => setHovered(null)}
              />
            );
          })}
        </svg>

        {/* Hover details */}
        {hovered && (
          <div className="absolute top-2 right-4 px-3 py-1.5 rounded bg-slate-900 border border-slate-700 text-[11px] shadow-lg text-slate-200 pointer-events-none">
            <div>Observation #{hovered.id}</div>
            <div>Score: {hovered.score.toFixed(4)}</div>
            <div className={hovered.is_anomaly ? "text-rose-400 font-bold" : "text-blue-400"}>
              {hovered.is_anomaly ? `Anomaly (${hovered.severity})` : "Normal"}
            </div>
            <div className="text-[10px] text-slate-400">
              PC1: {hovered.pc1} · PC2: {hovered.pc2}
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-between text-[11px] text-slate-400 mt-1 px-8">
        <span>Principal Component 1 ({varX}% variance)</span>
        <span>Principal Component 2 ({varY}% variance)</span>
      </div>
    </div>
  );
};

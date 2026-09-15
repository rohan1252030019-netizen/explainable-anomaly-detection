"use client";

import React from "react";

interface SeverityDonutChartProps {
  counts: {
    CRITICAL: number;
    HIGH: number;
    MEDIUM: number;
    LOW: number;
  };
}

export const SeverityDonutChart: React.FC<SeverityDonutChartProps> = ({ counts }) => {
  const total =
    (counts.CRITICAL || 0) +
    (counts.HIGH || 0) +
    (counts.MEDIUM || 0) +
    (counts.LOW || 0);

  if (total === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-xs text-slate-400">
        No anomalies to breakdown.
      </div>
    );
  }

  const items = [
    { label: "Critical", count: counts.CRITICAL || 0, color: "#EF4444" },
    { label: "High", count: counts.HIGH || 0, color: "#F97316" },
    { label: "Medium", count: counts.MEDIUM || 0, color: "#F59E0B" },
    { label: "Low", count: counts.LOW || 0, color: "#10B981" },
  ];

  const size = 160;
  const strokeWidth = 18;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  let accumulated = 0;

  return (
    <div className="flex flex-col sm:flex-row items-center justify-center gap-6 p-2">
      {/* SVG Donut */}
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          {items.map((item, i) => {
            const pct = item.count / total;
            const strokeDasharray = `${pct * circumference} ${circumference}`;
            const strokeDashoffset = -accumulated * circumference;
            accumulated += pct;

            if (pct <= 0) return null;

            return (
              <circle
                key={i}
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="none"
                stroke={item.color}
                strokeWidth={strokeWidth}
                strokeDasharray={strokeDasharray}
                strokeDashoffset={strokeDashoffset}
                className="transition-all duration-500 hover:opacity-90"
              />
            );
          })}
        </svg>

        {/* Center Total Count */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <span className="text-2xl font-black text-slate-900 dark:text-slate-100">
            {total}
          </span>
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
            Anomalies
          </span>
        </div>
      </div>

      {/* Legend */}
      <div className="space-y-2 text-xs">
        {items.map((item, i) => {
          const pct = total > 0 ? ((item.count / total) * 100).toFixed(0) : "0";
          return (
            <div key={i} className="flex items-center justify-between gap-4 min-w-[120px]">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-slate-600 dark:text-slate-300 font-medium">{item.label}</span>
              </div>
              <div className="flex items-center gap-1.5 font-semibold text-slate-900 dark:text-slate-100">
                <span>{item.count}</span>
                <span className="text-[10px] text-slate-400">({pct}%)</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

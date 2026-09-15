import React from "react";

interface ScoreGaugeProps {
  score: number;
  severity: string;
  threshold?: number;
  size?: number;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({
  score,
  severity,
  threshold = 0.5,
  size = 180,
}) => {
  const normScore = Math.max(0, Math.min(1, score));
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const circumference = Math.PI * radius; // half circle arc
  const strokeDashoffset = circumference - normScore * circumference;

  let color = "#10B981"; // Low (emerald)
  if (severity.toUpperCase() === "CRITICAL") color = "#EF4444";
  else if (severity.toUpperCase() === "HIGH") color = "#F97316";
  else if (severity.toUpperCase() === "MEDIUM") color = "#F59E0B";

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative" style={{ width: size, height: size / 2 + 20 }}>
        <svg width={size} height={size / 2 + 10} className="overflow-visible">
          {/* Background track */}
          <path
            d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
            fill="none"
            stroke="currentColor"
            className="text-slate-200 dark:text-slate-800"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />

          {/* Active score fill */}
          <path
            d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>

        {/* Center score readout */}
        <div className="absolute inset-x-0 bottom-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-extrabold tracking-tight" style={{ color }}>
            {normScore.toFixed(4)}
          </span>
          <span className="text-[11px] font-medium text-slate-400 mt-0.5">
            Normalized Anomaly Score
          </span>
        </div>
      </div>

      <div className="mt-3 flex items-center justify-between w-full max-w-[200px] text-[10px] text-slate-400">
        <span>0.00 (Normal)</span>
        <span className="font-semibold text-indigo-400">
          Threshold: {threshold.toFixed(3)}
        </span>
        <span>1.00 (Outlier)</span>
      </div>
    </div>
  );
};

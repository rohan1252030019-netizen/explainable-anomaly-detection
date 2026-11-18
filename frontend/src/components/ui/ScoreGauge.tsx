import React from "react";
import { cn } from "@/lib/cn";
import { severityStyle } from "./Badge";

interface ScoreGaugeProps {
  score: number;
  severity: string;
  threshold?: number;
  size?: number;
  className?: string;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({
  score,
  severity,
  threshold = 0.5,
  size = 180,
  className,
}) => {
  const normScore = Math.max(0, Math.min(1, score));
  const strokeWidth = 11;
  const radius = (size - strokeWidth) / 2;
  const circumference = Math.PI * radius;
  const offset = circumference - normScore * circumference;
  const style = severityStyle(severity);
  const thresholdX =
    strokeWidth / 2 + radius * 2 * Math.max(0, Math.min(1, threshold));

  return (
    <div
      className={cn("flex flex-col items-center", className)}
      role="meter"
      aria-valuenow={Number(normScore.toFixed(4))}
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuetext={`${normScore.toFixed(4)} out of 1, ${severity.toLowerCase()}`}
      aria-label="Normalised anomaly score"
    >
      <div className="relative" style={{ width: size, height: size / 2 + 18 }}>
        <svg
          width={size}
          height={size / 2 + 18}
          viewBox={`0 0 ${size} ${size / 2 + 18}`}
          className="overflow-visible"
          aria-hidden="true"
        >
          <defs>
            <linearGradient id="gauge-fill" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor={style.hex} stopOpacity={0.45} />
              <stop offset="100%" stopColor={style.hex} />
            </linearGradient>
          </defs>

          <path
            d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
            fill="none"
            stroke="var(--surface-muted)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />

          <path
            d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
            fill="none"
            stroke="url(#gauge-fill)"
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{
              transition: "stroke-dashoffset 700ms cubic-bezier(0.16,1,0.3,1)",
            }}
          />

          {threshold > 0 && threshold < 1 && (
            <line
              x1={thresholdX}
              y1={size / 2 - radius - 6}
              x2={thresholdX}
              y2={size / 2 + 3}
              stroke="var(--primary)"
              strokeWidth={1.5}
              strokeDasharray="3 3"
              opacity={0.9}
            />
          )}
        </svg>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col items-center">
          <span
            className="text-[1.7rem] font-extrabold leading-none tracking-tight tabular"
            style={{ color: style.hex }}
          >
            {normScore.toFixed(4)}
          </span>
          <span className="mt-1.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-subtle-foreground">
            Normalised score
          </span>
        </div>
      </div>

      <div className="mt-3 flex w-full max-w-[220px] items-center justify-between text-[9px] tabular text-subtle-foreground">
        <span>0.00</span>
        <span className="font-semibold text-primary">
          threshold {threshold.toFixed(3)}
        </span>
        <span>1.00</span>
      </div>
    </div>
  );
};

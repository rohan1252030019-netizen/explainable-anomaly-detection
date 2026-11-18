"use client";

import React, { useState } from "react";
import { ScoreHistBin } from "@/lib/api";
import {
  ChartEmpty,
  ChartFrame,
  LegendItem,
  SERIES,
  TooltipCard,
  TooltipRow,
} from "./ChartFrame";

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
  const [hover, setHover] = useState<{
    bin: ScoreHistBin;
    x: number;
    y: number;
  } | null>(null);

  if (!data || data.length === 0) {
    return <ChartEmpty message="No score distribution data available." />;
  }

  const width = 640;
  const height = 260;
  const padLeft = 44;
  const padBottom = 38;
  const padTop = 18;
  const padRight = 18;

  const chartWidth = width - padLeft - padRight;
  const chartHeight = height - padTop - padBottom;

  const maxCount = Math.max(...data.map((d) => d.normal_count + d.anomaly_count), 1);
  const binWidth = chartWidth / data.length;
  const thresholdX = padLeft + threshold * chartWidth;

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((p) => ({
    p,
    y: padTop + chartHeight * (1 - p),
    value: Math.round(maxCount * p),
  }));

  const anomalyTotal = data.reduce((sum, bin) => sum + bin.anomaly_count, 0);
  const recordTotal = data.reduce(
    (sum, bin) => sum + bin.normal_count + bin.anomaly_count,
    0
  );

  return (
    <ChartFrame
      title="Anomaly score distribution"
      subtitle={
        modelName
          ? `${modelName} · ${recordTotal.toLocaleString()} records, ${anomalyTotal.toLocaleString()} flagged`
          : `${recordTotal.toLocaleString()} records`
      }
      legend={
        <>
          <LegendItem color={SERIES.normal} label="Normal" />
          <LegendItem color={SERIES.anomaly} label="Anomaly" />
          <LegendItem
            color={SERIES.threshold}
            shape="line"
            label={`Threshold ${threshold.toFixed(3)}`}
          />
        </>
      }
    >
      <p className="sr-only">
        Histogram of {recordTotal} normalised anomaly scores across {data.length}{" "}
        bins. The decision threshold is {threshold.toFixed(3)}. {anomalyTotal}{" "}
        records are flagged as anomalous.
      </p>
      <div className="relative">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-auto w-full"
          role="img"
          aria-label="Histogram of normalised anomaly scores split into normal and anomalous counts"
          onMouseLeave={() => setHover(null)}
        >
          {yTicks.map((t) => (
            <g key={t.p}>
              <line
                x1={padLeft}
                y1={t.y}
                x2={width - padRight}
                y2={t.y}
                stroke={SERIES.grid}
                strokeDasharray="3 4"
              />
              <text
                x={padLeft - 10}
                y={t.y + 3}
                textAnchor="end"
                className="fill-subtle-foreground text-[9px] tabular"
              >
                {t.value}
              </text>
            </g>
          ))}

          {data.map((bin, i) => {
            const x = padLeft + i * binWidth;
            const normH = (bin.normal_count / maxCount) * chartHeight;
            const anomH = (bin.anomaly_count / maxCount) * chartHeight;
            const active = hover?.bin === bin;
            const barW = Math.max(1.5, binWidth - 2);

            return (
              <g key={i}>
                {normH > 0 && (
                  <rect
                    x={x + 1}
                    y={padTop + chartHeight - normH}
                    width={barW}
                    height={normH}
                    rx={Math.min(2, barW / 3)}
                    fill={SERIES.normal}
                    opacity={active ? 0.95 : 0.55}
                    style={{ transition: "opacity 120ms ease" }}
                  />
                )}
                {anomH > 0 && (
                  <rect
                    x={x + 1}
                    y={padTop + chartHeight - normH - anomH}
                    width={barW}
                    height={anomH}
                    rx={Math.min(2, barW / 3)}
                    fill={SERIES.anomaly}
                    opacity={active ? 1 : 0.9}
                    style={{ transition: "opacity 120ms ease" }}
                  />
                )}
                {/* Invisible hover target spanning the full column height */}
                <rect
                  x={x}
                  y={padTop}
                  width={binWidth}
                  height={chartHeight}
                  fill="transparent"
                  onMouseMove={(e) => {
                    const rect = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
                    setHover({
                      bin,
                      x: ((e.clientX - rect.left) / rect.width) * width,
                      y: ((e.clientY - rect.top) / rect.height) * height,
                    });
                  }}
                />
              </g>
            );
          })}

          <line
            x1={thresholdX}
            y1={padTop - 4}
            x2={thresholdX}
            y2={padTop + chartHeight}
            stroke={SERIES.threshold}
            strokeWidth={2}
            strokeDasharray="4 4"
          />

          {[0, 0.2, 0.4, 0.6, 0.8, 1].map((tick) => (
            <text
              key={tick}
              x={padLeft + tick * chartWidth}
              y={height - 14}
              textAnchor="middle"
              className="fill-subtle-foreground text-[10px] tabular"
            >
              {tick.toFixed(1)}
            </text>
          ))}

          <text
            x={padLeft + chartWidth / 2}
            y={height - 2}
            textAnchor="middle"
            className="fill-subtle-foreground text-[9px]"
          >
            Normalised anomaly score
          </text>

          <text
            x={thresholdX + 4}
            y={padTop + 9}
            className="fill-primary text-[9px] font-semibold tabular"
            aria-hidden="true"
          >
            τ {threshold.toFixed(2)}
          </text>
        </svg>

        {hover && (
          <TooltipCard x={hover.x} y={hover.y} width={width} height={height}>
            <div className="mb-1 flex items-baseline justify-between gap-3 font-semibold text-foreground">
              <span>
                {hover.bin.bin_start.toFixed(3)} – {hover.bin.bin_end.toFixed(3)}
              </span>
              <span className="text-[9px] font-normal text-subtle-foreground tabular">
                {(
                  hover.bin.normal_count + hover.bin.anomaly_count
                ).toLocaleString()}{" "}
                records
              </span>
            </div>
            <div className="space-y-0.5">
              <TooltipRow
                label="Normal"
                value={hover.bin.normal_count}
                color={SERIES.normal}
              />
              <TooltipRow
                label="Anomalies"
                value={hover.bin.anomaly_count}
                color={SERIES.anomaly}
              />
            </div>
          </TooltipCard>
        )}
      </div>

      <p className="mt-1 text-center text-[10px] text-subtle-foreground">
        0.0 = typical behaviour · 1.0 = most anomalous in this run
      </p>
    </ChartFrame>
  );
};

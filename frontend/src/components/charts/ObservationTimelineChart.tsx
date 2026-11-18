"use client";

import React, { useState } from "react";
import { ScoreTimelinePoint } from "@/lib/api";
import { sampleSeries } from "@/lib/downsample";
import {
  ChartEmpty,
  ChartFrame,
  LegendItem,
  SERIES,
  TooltipCard,
  TooltipRow,
} from "./ChartFrame";

interface ObservationTimelineChartProps {
  data: ScoreTimelinePoint[];
  threshold?: number;
}

export const ObservationTimelineChart: React.FC<ObservationTimelineChartProps> = ({
  data,
  threshold = 0.5,
}) => {
  const [hover, setHover] = useState<{
    pt: ScoreTimelinePoint;
    x: number;
    y: number;
  } | null>(null);

  if (!data || data.length === 0) {
    return <ChartEmpty message="No observation timeline available." height={200} />;
  }

  const width = 820;
  const height = 200;
  const padLeft = 40;
  const padRight = 18;
  const padTop = 14;
  const padBottom = 26;

  const chartWidth = width - padLeft - padRight;
  const chartHeight = height - padTop - padBottom;

  const maxIdx = data.reduce((acc, d) => (d.index > acc ? d.index : acc), 1);
  const thresholdY = padTop + chartHeight * (1 - threshold);
  const flagged = data.filter((point) => point.is_anomaly).length;

  // Rendering one node per observation is unusable past a few thousand rows.
  // Anomalies survive the thinning; the normals are strided.
  const rendered = sampleSeries(data, 2500, (point) => point.is_anomaly);

  const onMove = (
    e: React.MouseEvent<SVGCircleElement>,
    pt: ScoreTimelinePoint
  ) => {
    const rect = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
    setHover({
      pt,
      x: ((e.clientX - rect.left) / rect.width) * width,
      y: ((e.clientY - rect.top) / rect.height) * height,
    });
  };

  return (
    <ChartFrame
      title="Score by observation sequence"
      subtitle={
        rendered.sampled
          ? `${data.length.toLocaleString()} records in dataset order · showing ${rendered.points.length.toLocaleString()} · ${flagged.toLocaleString()} flagged`
          : `${data.length.toLocaleString()} records in dataset order · ${flagged.toLocaleString()} flagged`
      }
      legend={
        <>
          <LegendItem color={SERIES.normalPoint} shape="dot" label="Normal" />
          <LegendItem color={SERIES.anomaly} shape="dot" label="Flagged" />
          <LegendItem
            color={SERIES.threshold}
            shape="line"
            label={`Threshold ${threshold.toFixed(3)}`}
          />
        </>
      }
    >
      <p className="sr-only">
        Line of {data.length} anomaly scores in dataset order with a threshold at{" "}
        {threshold.toFixed(3)}. {flagged} records are above the threshold.
      </p>
      <div className="relative">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-auto w-full"
          role="img"
          aria-label="Anomaly score plotted against observation index with a threshold line"
          onMouseLeave={() => setHover(null)}
        >
          {[0, 0.5, 1].map((val) => {
            const y = padTop + chartHeight * (1 - val);
            return (
              <g key={val}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke={SERIES.grid}
                  strokeDasharray="2 4"
                />
                <text
                  x={padLeft - 8}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-subtle-foreground text-[9px] tabular"
                >
                  {val.toFixed(1)}
                </text>
              </g>
            );
          })}

          <line
            x1={padLeft}
            y1={thresholdY}
            x2={width - padRight}
            y2={thresholdY}
            stroke={SERIES.threshold}
            strokeWidth={1.5}
            strokeDasharray="4 4"
          />
          <text
            x={width - padRight}
            y={thresholdY - 4}
            textAnchor="end"
            className="fill-primary text-[9px] font-semibold tabular"
          >
            τ {threshold.toFixed(2)}
          </text>

          {rendered.points.map((pt, i) => {
            const x = padLeft + (pt.index / maxIdx) * chartWidth;
            const y = padTop + chartHeight * (1 - Math.max(0, Math.min(1, pt.score)));
            const active = hover?.pt === pt;
            const r = Math.max(8, chartWidth / rendered.points.length / 2);

            return (
              <g key={i}>
                <circle
                  cx={x}
                  cy={y}
                  r={r}
                  fill="transparent"
                  onMouseMove={(e) => onMove(e, pt)}
                />
                {pt.is_anomaly ? (
                  <circle
                    cx={x}
                    cy={y}
                    r={active ? 5 : 3.5}
                    fill={SERIES.anomaly}
                    className="pointer-events-none stroke-surface"
                    strokeWidth={1.2}
                    style={{ transition: "r 120ms ease" }}
                  />
                ) : (
                  <circle
                    cx={x}
                    cy={y}
                    r={active ? 3.5 : 1.8}
                    fill={SERIES.normalPoint}
                    opacity={active ? 1 : 0.75}
                    className="pointer-events-none"
                    style={{ transition: "r 120ms ease, opacity 120ms ease" }}
                  />
                )}
              </g>
            );
          })}
        </svg>

        {hover && (
          <TooltipCard x={hover.x} y={hover.y} width={width} height={height}>
            <div className="mb-1 font-semibold text-foreground">
              Observation #{hover.pt.index}
            </div>
            <TooltipRow
              label="Score"
              value={hover.pt.score.toFixed(4)}
              color={
                hover.pt.is_anomaly ? SERIES.anomaly : SERIES.normalPoint
              }
            />
            <div
              className="mt-0.5 text-[10px] font-bold"
              style={{
                color: hover.pt.is_anomaly
                  ? "var(--danger)"
                  : "var(--muted-foreground)",
              }}
            >
              {hover.pt.is_anomaly ? hover.pt.severity : "Normal"}
            </div>
          </TooltipCard>
        )}
      </div>

      <div className="mt-1 flex justify-between px-8 text-[10px] text-subtle-foreground tabular">
        <span>Observation #0</span>
        <span>#{maxIdx.toLocaleString()}</span>
      </div>
    </ChartFrame>
  );
};

"use client";

import React, { useState } from "react";
import { PcaAnalyticsResponse } from "@/lib/api";
import {
  ChartEmpty,
  ChartFrame,
  LegendItem,
  SERIES,
  TooltipCard,
  TooltipRow,
} from "./ChartFrame";
import { extent, sampleSeries } from "@/lib/downsample";

type Point = PcaAnalyticsResponse["points"][number];

interface PcaScatterChartProps {
  data: PcaAnalyticsResponse;
}

const tickLabel = (value: number) =>
  Math.abs(value) >= 1000
    ? value.toLocaleString(undefined, { maximumFractionDigits: 0 })
    : value.toLocaleString(undefined, { maximumFractionDigits: 1 });

export const PcaScatterChart: React.FC<PcaScatterChartProps> = ({ data }) => {
  const [hover, setHover] = useState<{ pt: Point; x: number; y: number } | null>(
    null
  );

  if (!data || !data.points || data.points.length === 0) {
    return <ChartEmpty message="No PCA projection data available." height={280} />;
  }

  const width = 660;
  const height = 340;
  const pad = 48;

  const chartWidth = width - pad * 2;
  const chartHeight = height - pad * 2;

  const pc1Vals = data.points.map((point) => point.pc1);
  const pc2Vals = data.points.map((point) => point.pc2);

  // A loop, not Math.min(...values): spreading a 200k-element array blows the
  // call stack.  The extent is computed over every point so the axes do not
  // shift when the render set is down-sampled.
  const [minX, maxX] = extent(pc1Vals);
  const [minY, maxY] = extent(pc2Vals);

  // One <circle> per observation does not scale.  Anomalies are always kept;
  // normals are thinned with a stride so the cluster shape survives.
  const rendered = sampleSeries(data.points, 4000, (point) => point.is_anomaly);

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  const varX =
    data.variance_explained?.[0] !== undefined
      ? (data.variance_explained[0] * 100).toFixed(1)
      : null;
  const varY =
    data.variance_explained?.[1] !== undefined
      ? (data.variance_explained[1] * 100).toFixed(1)
      : null;

  const anomalies = data.points.filter((point) => point.is_anomaly).length;
  const tickFractions = [0, 0.25, 0.5, 0.75, 1];

  const onMove = (event: React.MouseEvent<SVGCircleElement>, point: Point) => {
    const rect = event.currentTarget.ownerSVGElement!.getBoundingClientRect();
    setHover({
      pt: point,
      x: ((event.clientX - rect.left) / rect.width) * width,
      y: ((event.clientY - rect.top) / rect.height) * height,
    });
  };

  return (
    <ChartFrame
      title="PCA projection"
      subtitle={
        rendered.sampled
          ? `First two principal components · showing ${rendered.points.length.toLocaleString()} of ${data.points.length.toLocaleString()} observations (all anomalies kept)`
          : `First two principal components · ${data.points.length.toLocaleString()} observations`
      }
      legend={
        <>
          <LegendItem
            color={SERIES.normalPoint}
            shape="dot"
            label={`Normal (${(data.points.length - anomalies).toLocaleString()})`}
          />
          <LegendItem
            color={SERIES.anomaly}
            shape="dot"
            label={`Anomalies (${anomalies.toLocaleString()})`}
          />
        </>
      }
    >
      <p className="sr-only">
        Scatter plot of {data.points.length} observations projected onto the
        first two principal components. {anomalies} observations are flagged as
        anomalies and rendered in a distinct colour.
      </p>
      <div className="relative">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-auto w-full"
          role="img"
          aria-label="Scatter plot of the first two principal components, highlighting anomalous observations"
          onMouseLeave={() => setHover(null)}
        >
          <defs>
            <clipPath id="pca-plot-clip">
              <rect
                x={pad}
                y={pad}
                width={chartWidth}
                height={chartHeight}
              />
            </clipPath>
          </defs>

          {tickFractions.map((fraction) => {
            const x = pad + fraction * chartWidth;
            const y = pad + chartHeight * (1 - fraction);
            return (
              <g key={fraction}>
                <line
                  x1={x}
                  y1={pad}
                  x2={x}
                  y2={height - pad}
                  stroke={SERIES.grid}
                  strokeDasharray="2 4"
                  opacity={fraction === 0 ? 1 : 0.7}
                />
                <line
                  x1={pad}
                  y1={y}
                  x2={width - pad}
                  y2={y}
                  stroke={SERIES.grid}
                  strokeDasharray="2 4"
                  opacity={fraction === 0 ? 1 : 0.7}
                />
                <text
                  x={x}
                  y={height - pad + 15}
                  textAnchor="middle"
                  className="fill-subtle-foreground text-[9px] tabular"
                >
                  {tickLabel(minX + rangeX * fraction)}
                </text>
                <text
                  x={pad - 8}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-subtle-foreground text-[9px] tabular"
                >
                  {tickLabel(minY + rangeY * fraction)}
                </text>
              </g>
            );
          })}

          <g clipPath="url(#pca-plot-clip)">
            {rendered.points.map((point, index) => {
              const x = pad + ((point.pc1 - minX) / rangeX) * chartWidth;
              const y = height - pad - ((point.pc2 - minY) / rangeY) * chartHeight;
              const active = hover?.pt === point;

              return point.is_anomaly ? (
                <circle
                  key={index}
                  cx={x}
                  cy={y}
                  r={active ? 6 : 4}
                  fill={SERIES.anomaly}
                  className="stroke-surface"
                  strokeWidth={1.4}
                  style={{
                    transition: "r 120ms ease",
                    cursor: "pointer",
                  }}
                  onMouseMove={(event) => onMove(event, point)}
                />
              ) : (
                <circle
                  key={index}
                  cx={x}
                  cy={y}
                  r={active ? 4 : 2.4}
                  fill={SERIES.normalPoint}
                  opacity={active ? 1 : 0.75}
                  style={{
                    transition: "r 120ms ease, opacity 120ms ease",
                    cursor: "pointer",
                  }}
                  onMouseMove={(event) => onMove(event, point)}
                />
              );
            })}
          </g>

          <text
            x={pad + chartWidth / 2}
            y={height - 4}
            textAnchor="middle"
            className="fill-subtle-foreground text-[9px]"
          >
            PC1{varX ? ` · ${varX}% variance explained` : ""}
          </text>
          <text
            x={10}
            y={pad + chartHeight / 2}
            textAnchor="middle"
            transform={`rotate(-90 10 ${pad + chartHeight / 2})`}
            className="fill-subtle-foreground text-[9px]"
          >
            PC2{varY ? ` · ${varY}%` : ""}
          </text>
        </svg>

        {hover && (
          <TooltipCard x={hover.x} y={hover.y} width={width} height={height}>
            <div className="mb-1 font-semibold text-foreground">
              Observation #{hover.pt.id}
            </div>
            <div className="space-y-0.5">
              <TooltipRow
                label="Score"
                value={hover.pt.score.toFixed(4)}
                color={
                  hover.pt.is_anomaly ? SERIES.anomaly : SERIES.normalPoint
                }
              />
              <div className="text-[10px] tabular text-subtle-foreground">
                PC1 {hover.pt.pc1.toFixed(2)} · PC2 {hover.pt.pc2.toFixed(2)}
              </div>
              <div
                className="text-[10px] font-bold"
                style={{
                  color: hover.pt.is_anomaly
                    ? "var(--danger)"
                    : "var(--muted-foreground)",
                }}
              >
                {hover.pt.is_anomaly ? hover.pt.severity : "Normal"}
              </div>
            </div>
          </TooltipCard>
        )}
      </div>
    </ChartFrame>
  );
};

"use client";

/**
 * Presentational pieces for the Rigor view.
 *
 * `RigorView.tsx` had grown to ~1,240 lines with formatters, four small
 * components and the entire page body in one file.  These are the parts that
 * carry no state and no data fetching, so they lift out cleanly: the view keeps
 * the request lifecycle, and the primitives live here where they can be read
 * (and tested) on their own.  Nothing about the rendering changed.
 */

import {
  AlertTriangle,
  Minus,
  RefreshCw,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import type { HoldoutMetrics } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { SectionLabel } from "@/components/ui/Card";
import { cn } from "@/lib/cn";

export const isNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

export const fmt = (value: number | null | undefined, digits = 3) =>
  isNumber(value) ? value.toFixed(digits) : "—";

export const pct = (value: number | null | undefined, digits = 1) =>
  isNumber(value) ? `${(value * 100).toFixed(digits)}%` : "—";

export const signed = (
  value: number | null | undefined,
  digits = 3,
  percent = false
) => {
  if (!isNumber(value)) return "—";
  const scaled = percent ? value * 100 : value;
  const body = Math.abs(scaled).toFixed(digits);
  if (Number(body) === 0)
    return percent ? `0.${"0".repeat(digits)}%` : `0.${"0".repeat(digits)}`;
  return `${scaled > 0 ? "+" : "−"}${body}${percent ? "%" : ""}`;
};

export function DeltaChip({
  value,
  betterWhenLower = true,
  label,
  percent = false,
}: {
  value: number | null | undefined;
  betterWhenLower?: boolean;
  label: string;
  percent?: boolean;
}) {
  if (!isNumber(value)) {
    return (
      <span className="text-[10px] text-subtle-foreground" title="Not available">
        n/a
      </span>
    );
  }

  const magnitude = Math.abs(value);
  const meaningful = magnitude >= (percent ? 0.005 : 0.0005);
  const isImprovement = betterWhenLower ? value < 0 : value > 0;
  const tone = !meaningful
    ? "text-subtle-foreground"
    : isImprovement
      ? "text-success"
      : "text-caution";
  const Icon = !meaningful ? Minus : value < 0 ? TrendingDown : TrendingUp;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-border px-2 py-0.5 text-[10px] font-bold tabular",
        tone
      )}
      title={
        meaningful
          ? `${label}: ${isImprovement ? "better" : "worse"} out of sample by ${
              percent
                ? `${(magnitude * 100).toFixed(2)} points`
                : magnitude.toFixed(4)
            }`
          : `${label}: negligible difference`
      }
    >
      <Icon size={10} aria-hidden="true" />
      {signed(value, percent ? 2 : 3, percent)}
    </span>
  );
}

export function MetricRow({
  label,
  inSample,
  outOfSample,
  gap,
}: {
  label: string;
  inSample?: number | null;
  outOfSample?: number | null;
  gap?: number | null;
}) {
  return (
    <tr className="border-t border-border">
      <th
        scope="row"
        className="py-2.5 pr-3 text-left text-xs font-semibold text-foreground"
      >
        {label}
      </th>
      <td className="py-2.5 pr-3 text-right text-xs tabular text-muted-foreground">
        {fmt(inSample)}
      </td>
      <td className="py-2.5 pr-3 text-right text-xs font-bold tabular text-foreground">
        {fmt(outOfSample)}
      </td>
      <td className="py-2.5 text-right">
        <DeltaChip
          value={gap}
          label={`${label} optimism gap`}
          betterWhenLower
        />
      </td>
    </tr>
  );
}

export function InlineError({
  title,
  message,
  onRetry,
  retryLabel = "Retry",
  busy = false,
}: {
  title: string;
  message: string;
  onRetry: () => void;
  retryLabel?: string;
  busy?: boolean;
}) {
  return (
    <div
      role="alert"
      className="flex flex-col gap-3 rounded-xl border border-danger-line bg-danger-soft p-4 sm:flex-row sm:items-start"
    >
      <AlertTriangle
        size={15}
        className="mt-0.5 shrink-0 text-danger"
        aria-hidden="true"
      />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-bold text-foreground">{title}</p>
        <p className="mt-0.5 break-words text-[11px] leading-relaxed text-muted-foreground">
          {message}
        </p>
      </div>
      <Button
        variant="ghost"
        size="xs"
        onClick={onRetry}
        disabled={busy}
        className="shrink-0 self-start"
      >
        <RefreshCw
          size={11}
          className={busy ? "animate-spin" : undefined}
          aria-hidden="true"
        />
        {retryLabel}
      </Button>
    </div>
  );
}

export function TransferPanel({ holdout }: { holdout: HoldoutMetrics }) {
  const rows = [
    {
      label: "Flag rate",
      inValue: holdout.flag_rate_in_sample,
      outValue: holdout.flag_rate_out_of_sample,
      format: (value: number) => pct(value),
    },
    {
      label: "Mean score",
      inValue: holdout.score_mean_in_sample,
      outValue: holdout.score_mean_out_of_sample,
      format: (value: number) => value.toFixed(4),
    },
  ];

  if (rows.every((row) => !isNumber(row.inValue) && !isNumber(row.outValue))) {
    return null;
  }

  return (
    <div className="mt-4 rounded-xl border border-border bg-surface-muted p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SectionLabel>Distribution transfer</SectionLabel>
        <span className="text-[10px] text-subtle-foreground">
          Same threshold applied to both splits
        </span>
      </div>

      <div className="mt-3 space-y-3">
        {rows.map((row) => {
          const inValue = isNumber(row.inValue) ? row.inValue : null;
          const outValue = isNumber(row.outValue) ? row.outValue : null;
          const max = Math.max(
            Math.abs(inValue ?? 0),
            Math.abs(outValue ?? 0),
            1e-9
          );
          const drift =
            inValue !== null && outValue !== null ? outValue - inValue : null;

          return (
            <div key={row.label}>
              <div className="mb-1.5 flex items-center justify-between gap-3">
                <span className="text-[11px] font-semibold text-foreground">
                  {row.label}
                </span>
                <span className="text-[10px] text-subtle-foreground">
                  {drift === null
                    ? "Drift unavailable"
                    : `${signed(
                        drift,
                        row.label === "Flag rate" ? 2 : 4,
                        row.label === "Flag rate"
                      )} held-out vs in-sample`}
                </span>
              </div>

              <div className="space-y-1.5">
                {[
                  { name: "In-sample", value: inValue, tone: "bg-info/55" },
                  { name: "Held-out", value: outValue, tone: "bg-primary" },
                ].map((bar) => (
                  <div key={bar.name} className="flex items-center gap-2">
                    <span className="w-16 shrink-0 text-[10px] text-subtle-foreground">
                      {bar.name}
                    </span>
                    <span
                      className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface"
                      aria-hidden="true"
                    >
                      <span
                        className={cn("block h-full rounded-full", bar.tone)}
                        style={{
                          width:
                            bar.value === null
                              ? "0%"
                              : `${Math.max(
                                  1.5,
                                  (Math.abs(bar.value) / max) * 100
                                )}%`,
                        }}
                      />
                    </span>
                    <span className="w-16 shrink-0 text-right text-[10px] font-semibold tabular text-muted-foreground">
                      {bar.value === null ? "—" : row.format(bar.value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export const TEST_SIZES = [0.2, 0.3, 0.4];

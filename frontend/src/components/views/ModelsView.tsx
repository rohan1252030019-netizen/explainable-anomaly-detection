"use client";

import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  Layers,
  Play,
  RefreshCw,
  ShieldQuestion,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api, type ModelBenchmarkResponse } from "@/lib/api";
import { StatCard } from "@/components/ui/StatCard";
import { Button } from "@/components/ui/Button";
import { Card, SectionLabel } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Spinner } from "@/components/ui/Feedback";
import { useToast } from "@/components/ui/Toast";
import { severityStyle } from "@/components/ui/Badge";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

const toNumber = (value: unknown): number | null => {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const num = (value: unknown, digits = 3) => {
  const parsed = toNumber(value);
  return parsed === null ? "N/A" : parsed.toFixed(digits);
};

const SHORT_NAMES: Record<string, string> = {
  "Isolation Forest": "IF",
  "Local Outlier Factor": "LOF",
  "One-Class SVM": "OCSVM",
  "Elliptic Envelope": "EE",
};

export const ModelsView: React.FC = () => {
  const { dataset, setActiveTab } = useApp();
  const { toast } = useToast();

  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ModelBenchmarkResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [contamination, setContamination] = useState(0.05);

  const matrixReady = dataset?.feature_matrix_ready === true;

  const handleBenchmark = async () => {
    if (!matrixReady) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.benchmarkModels(contamination);
      setData(result);
      toast(
        "success",
        "Benchmark complete",
        `${result.all_model_consensus.toLocaleString()} records were flagged by every detector.`
      );
    } catch (err) {
      const message = errorMessage(err);
      setError(message);
      toast("error", "Benchmark failed", message);
    } finally {
      setLoading(false);
    }
  };

  const models = data?.comparison_table.map((row) => row.Model) ?? [];

  const countByModel = useMemo(() => {
    const map = new Map<string, number>();
    data?.model_anomaly_counts?.forEach((item) => {
      map.set(item.model, item.anomaly_count);
    });
    return map;
  }, [data?.model_anomaly_counts]);

  const jaccardLookup = useMemo(() => {
    const map = new Map<string, number>();
    data?.pairwise_jaccard.forEach((item) => {
      map.set(`${item.model_a}::${item.model_b}`, item.jaccard);
      map.set(`${item.model_b}::${item.model_a}`, item.jaccard);
    });
    return map;
  }, [data?.pairwise_jaccard]);

  if (!dataset?.loaded) {
    return (
      <EmptyState
        icon={Database}
        title="No dataset to benchmark"
        description="Load a CSV or the benchmark sample, then compare all four unsupervised detectors on the same feature matrix."
        actionText="Open dataset"
        onAction={() => setActiveTab("dataset")}
      />
    );
  }

  return (
    <div className="space-y-5">
      {/* Benchmark setup */}
      <Card className="flex flex-col items-start justify-between gap-4 p-5 lg:flex-row lg:items-center">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-accent" aria-hidden="true" />
            <SectionLabel>Benchmark setup</SectionLabel>
          </div>
          <p className="mt-1.5 max-w-2xl text-[11px] leading-relaxed text-ink-muted">
            Fit Isolation Forest, LOF, One-Class SVM and Elliptic Envelope on the
            same preprocessed matrix. Every detector receives the same
            contamination rate so overlap is comparable.
          </p>
        </div>

        <div className="flex shrink-0 flex-wrap items-center gap-3 lg:flex-nowrap">
          <div className="flex items-center gap-2 rounded-lg border border-edge bg-inset px-3 py-1.5">
            <label
              htmlFor="bench-contamination"
              className="text-[11px] text-ink-muted"
            >
              Contamination
            </label>
            <span className="font-mono text-[11px] font-bold tabular text-accent">
              {(contamination * 100).toFixed(0)}%
            </span>
          </div>
          <input
            id="bench-contamination"
            type="range"
            min={0.01}
            max={0.25}
            step={0.01}
            value={contamination}
            aria-valuetext={`${(contamination * 100).toFixed(0)} percent`}
            style={{
              ["--range-progress" as string]: `${
                ((contamination - 0.01) / 0.24) * 100
              }%`,
            }}
            onChange={(event) =>
              setContamination(Number(event.target.value))
            }
            className="w-28"
          />
          <Button
            onClick={handleBenchmark}
            disabled={loading || !matrixReady}
          >
            {loading ? <Spinner /> : <Play size={14} aria-hidden="true" />}
            {loading ? "Training all detectors…" : "Run benchmark"}
          </Button>
        </div>
      </Card>

      {!matrixReady && (
        <div className="flex flex-col gap-3 rounded-xl border border-caution/30 bg-caution-soft p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <Database
              size={15}
              className="mt-0.5 shrink-0 text-caution"
              aria-hidden="true"
            />
            <div>
              <p className="text-xs font-bold text-ink">
                Build the feature matrix first
              </p>
              <p className="mt-0.5 text-[11px] text-ink-muted">
                The benchmark trains on the preprocessed matrix, which is not
                ready for this dataset.
              </p>
            </div>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setActiveTab("dataset")}
            className="shrink-0"
          >
            Prepare features
          </Button>
        </div>
      )}

      {error && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-xl border border-critical/30 bg-critical-soft p-4"
        >
          <AlertTriangle
            size={15}
            className="mt-0.5 shrink-0 text-critical"
            aria-hidden="true"
          />
          <div className="min-w-0">
            <p className="text-xs font-bold text-ink">Benchmark failed</p>
            <p className="mt-0.5 break-words text-[11px] leading-relaxed text-ink-muted">
              {error}
            </p>
          </div>
          <Button
            variant="ghost"
            size="xs"
            onClick={handleBenchmark}
            disabled={loading || !matrixReady}
            className="ml-auto shrink-0"
          >
            <RefreshCw size={11} aria-hidden="true" />
            Retry
          </Button>
        </div>
      )}

      {data ? (
        <div className="space-y-4" aria-busy={loading}>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <StatCard
              label="All-model consensus"
              value={data.all_model_consensus}
              sub="Flagged by every detector"
              icon={CheckCircle2}
              accentColor={severityStyle("CRITICAL").hex}
            />
            <StatCard
              label="Model disagreements"
              value={data.disagreed_count}
              sub="Flagged by some models only"
              icon={ShieldQuestion}
              accentColor={severityStyle("MEDIUM").hex}
            />
            <StatCard
              label="Agreed normal"
              value={data.agreed_normal}
              sub="Unanimously considered inliers"
              icon={CheckCircle2}
              accentColor={severityStyle("LOW").hex}
            />
          </div>

          <Card className="overflow-hidden">
            <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-edge bg-elevated px-4 py-3">
              <div>
                <h3 className="text-xs font-bold text-ink">
                  Detector comparison
                </h3>
                <p className="mt-0.5 text-[10px] text-ink-faint">
                  Metrics are computed against the available labels; PR-AUC is
                  shown where the backend reports it.
                </p>
              </div>
              <span className="rounded-full border border-edge bg-inset px-2 py-0.5 text-[10px] font-semibold text-ink-faint">
                {models.length} detectors ·{" "}
                {(contamination * 100).toFixed(0)}% contamination
              </span>
            </div>
            <div className="overflow-x-auto" data-scroll>
              <table className="data-table min-w-[48rem] text-[11px]">
                <caption className="sr-only">
                  Comparative metrics per detector, including flagged count and
                  PR-AUC
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col">Flagged</th>
                    <th scope="col">Score sep.</th>
                    <th scope="col">Fit time</th>
                    <th scope="col">F1</th>
                    <th scope="col">ROC-AUC</th>
                    <th scope="col">PR-AUC</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {data.comparison_table.map((row, index) => (
                    <tr key={`${row.Model}-${index}`}>
                      <td className="font-sans">
                        <span className="block font-bold text-ink">
                          {row.Model}
                        </span>
                        <span className="mt-0.5 block text-[10px] text-ink-faint">
                          {countByModel.get(row.Model) ?? 0} flagged
                        </span>
                      </td>
                      <td className="tabular text-ink-muted">
                        {(countByModel.get(row.Model) ?? 0).toLocaleString()}
                      </td>
                      <td className="font-bold tabular text-accent">
                        {num(row["Score Sep."], 4)}
                      </td>
                      <td className="tabular text-ink-muted">
                        {(() => {
                          const fitTime = toNumber(
                            row["Runtime (s)"] ?? row["Fit Time (s)"]
                          );
                          return fitTime === null
                            ? "N/A"
                            : `${fitTime.toFixed(3)}s`;
                        })()}
                      </td>
                      <td className="tabular text-ink-muted">{num(row.F1)}</td>
                      <td className="tabular text-ink-muted">
                        {num(row["ROC-AUC"])}
                      </td>
                      <td className="tabular text-ink-muted">
                        {num(row["PR-AUC"])}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card className="p-5">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <div>
                <SectionLabel>Pairwise Jaccard overlap</SectionLabel>
                <p className="mt-1 text-[11px] text-ink-muted">
                  Intersection over union of outlier sets. 1.0 means identical
                  flags; 0.0 means completely disjoint.
                </p>
              </div>
              <span className="text-[10px] text-ink-faint">
                Diagonal = same model
              </span>
            </div>

            <div className="mt-4 overflow-x-auto" data-scroll>
              <table className="w-full min-w-[30rem] border-collapse text-center text-[11px]">
                <caption className="sr-only">
                  Pairwise Jaccard similarity matrix between detectors
                </caption>
                <thead>
                  <tr>
                    <th className="p-1.5" scope="col">
                      <span className="sr-only">Model</span>
                    </th>
                    {models.map((model) => (
                      <th
                        key={model}
                        scope="col"
                        className="p-1.5 text-[10px] font-bold text-ink-faint"
                        title={model}
                      >
                        {SHORT_NAMES[model] ?? model}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {models.map((rowModel) => (
                    <tr key={rowModel}>
                      <th
                        scope="row"
                        className="whitespace-nowrap p-1.5 text-left text-[10px] font-semibold text-ink-muted"
                        title={rowModel}
                      >
                        {SHORT_NAMES[rowModel] ?? rowModel}
                      </th>
                      {models.map((colModel) => {
                        const same = rowModel === colModel;
                        const value = same
                          ? 1
                          : jaccardLookup.get(
                              `${rowModel}::${colModel}`
                            ) ?? null;

                        return (
                          <td key={colModel} className="p-1">
                            <div
                              className={cn(
                                "grid h-10 place-items-center rounded-md border text-[11px] font-bold tabular",
                                same
                                  ? "border-edge bg-inset text-ink-faint"
                                  : value === null
                                    ? "border-edge bg-inset text-ink-faint"
                                    : "border-edge text-ink"
                              )}
                              style={
                                !same && value !== null
                                  ? {
                                      backgroundColor: `color-mix(in srgb, var(--accent) ${
                                        Math.round(value * 55)
                                      }%, transparent)`,
                                    }
                                  : undefined
                              }
                              aria-label={
                                value === null
                                  ? `No Jaccard value for ${rowModel} and ${colModel}`
                                  : `Jaccard similarity between ${rowModel} and ${colModel}: ${value.toFixed(
                                      3
                                    )}`
                              }
                              title={
                                value === null
                                  ? "No value"
                                  : `${rowModel} × ${colModel}: ${value.toFixed(
                                      3
                                    )}`
                              }
                            >
                              {value === null ? "—" : value.toFixed(2)}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      ) : (
        !loading && (
          <Card className="p-10 text-center">
            <span className="mx-auto grid size-10 place-items-center rounded-xl border border-edge bg-inset text-ink-faint">
              <Layers size={18} aria-hidden="true" />
            </span>
            <p className="mt-3 text-sm font-bold text-ink">
              No benchmark run yet
            </p>
            <p className="mx-auto mt-1 max-w-md text-xs leading-relaxed text-ink-muted">
              Run all four detectors to compare flag counts, separation, ranking
              metrics and pairwise overlap on the same rows.
            </p>
          </Card>
        )
      )}

      {loading && !data && (
        <Card className="flex items-center justify-center gap-2 p-10 text-xs text-ink-muted">
          <Spinner />
          Training detectors on the current feature matrix…
        </Card>
      )}
    </div>
  );
};

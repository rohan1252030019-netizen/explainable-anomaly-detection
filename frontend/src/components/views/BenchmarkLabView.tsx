"use client";

import React, { useState } from "react";
import {
  Database,
  FlaskConical,
  Info,
  Play,
  Trophy,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api, type SyntheticBenchmarkResponse } from "@/lib/api";
import { Card, SectionLabel } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Spinner } from "@/components/ui/Feedback";
import { StatCard } from "@/components/ui/StatCard";
import { Badge } from "@/components/ui/Badge";
import { useToast } from "@/components/ui/Toast";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

const ANOMALY_TYPES = [
  {
    id: "multivariate",
    label: "Multivariate Subspace Violations",
    desc: "Violates feature correlations and joint covariance structure while keeping marginal 1D distributions looking normal. Crucial for testing subspace detectors.",
  },
  {
    id: "point",
    label: "Extreme Point Anomalies",
    desc: "Extreme magnitude values displaced > 4 IQR from the median across isolated dimensions. Tests detector robustness to tail spikes.",
  },
  {
    id: "cluster",
    label: "Micro-Cluster Anomalies",
    desc: "Dense localized cluster of anomalous points in low-density space. Tests whether local density algorithms can differentiate small dense anomalies.",
  },
];

export const BenchmarkLabView: React.FC = () => {
  const { dataset } = useApp();
  const { toast } = useToast();

  const [loading, setLoading] = useState(false);
  const [anomalyType, setAnomalyType] = useState<"point" | "multivariate" | "cluster">("multivariate");
  const [contamination, setContamination] = useState(0.05);
  const [benchmarkResult, setBenchmarkResult] = useState<SyntheticBenchmarkResponse | null>(null);

  const handleRunBenchmark = async () => {
    if (!dataset?.loaded) return;
    try {
      setLoading(true);
      const res = await api.runSyntheticBenchmark({
        anomaly_type: anomalyType,
        contamination: contamination,
      });
      setBenchmarkResult(res);
      toast(
        "success",
        "Benchmark Completed",
        `Best detector: ${res.best_model ?? "N/A"}`
      );
    } catch (err) {
      toast(
        "error",
        "Benchmark failed",
        errorMessage(err)
      );
    } finally {
      setLoading(false);
    }
  };

  if (!dataset?.loaded) {
    return (
      <EmptyState
        icon={Database}
        title="No dataset loaded"
        description="Load or upload a dataset first to benchmark detectors against controlled synthetic anomalies."
      />
    );
  }

  const resultsTable = benchmarkResult?.results_table ?? [];

  return (
    <div className="space-y-6">
      {/* Scientific Notice */}
      <Card className="border-l-4 border-l-primary p-4">
        <div className="flex items-start gap-3">
          <Info size={16} className="text-primary shrink-0 mt-0.5" />
          <div className="space-y-1">
            <SectionLabel>Scientific Validation Notice</SectionLabel>
            <p className="text-xs text-muted-foreground leading-relaxed">
              In real-world unsupervised settings, ground-truth anomaly labels do not exist. To mathematically evaluate and rank detection algorithms without fabricating artificial metrics, the <strong>Synthetic Injection Laboratory</strong> introduces mathematically controlled perturbations (point, multivariate subspace, or cluster). Detectors are benchmarked strictly against these traceable injected ground truths.
            </p>
          </div>
        </div>
      </Card>

      {/* Benchmark Experiment Configuration */}
      <Card className="p-5">
        <div className="flex items-center gap-2 border-b border-border pb-3">
          <FlaskConical size={16} className="text-primary" />
          <SectionLabel>Experiment Configuration</SectionLabel>
        </div>

        <div className="mt-4 space-y-4">
          <div>
            <label className="text-xs font-semibold text-foreground">Perturbation Geometry</label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
              {ANOMALY_TYPES.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setAnomalyType(t.id as "point" | "multivariate" | "cluster")}
                  className={cn(
                    "flex flex-col text-left p-3 rounded-lg border transition-all",
                    anomalyType === t.id
                      ? "border-primary bg-primary/5 ring-1 ring-primary"
                      : "border-border bg-background hover:border-subtle-foreground"
                  )}
                >
                  <span className="text-xs font-bold text-foreground">{t.label}</span>
                  <span className="text-[11px] text-muted-foreground mt-1 leading-snug">{t.desc}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
            <div className="flex items-center gap-4">
              <label htmlFor="contamination-slider" className="text-xs font-semibold text-foreground">
                Contamination Ratio: <span className="font-mono text-primary">{(contamination * 100).toFixed(1)}%</span>
              </label>
              <input
                id="contamination-slider"
                type="range"
                min={0.01}
                max={0.20}
                step={0.01}
                value={contamination}
                onChange={(e) => setContamination(Number(e.target.value))}
                className="w-44 accent-primary cursor-pointer"
              />
            </div>

            <Button
              onClick={handleRunBenchmark}
              disabled={loading}
              className="gap-2"
            >
              {loading ? <Spinner /> : <Play size={13} />}
              {loading ? "Running Benchmark..." : "Run Benchmark Experiment"}
            </Button>
          </div>
        </div>
      </Card>

      {/* Benchmark Results */}
      {benchmarkResult && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Best Performing Detector"
              value={benchmarkResult.best_model ?? "—"}
              accentColor="var(--success)"
              sub="Highest combined ROC-AUC & PR-AUC"
            />
            <StatCard
              label="Injected Records"
              value={String(benchmarkResult.n_injected ?? 0)}
              sub={`Controlled ${benchmarkResult.anomaly_type} perturbations`}
            />
            <StatCard
              label="Contamination Rate"
              value={`${((benchmarkResult.contamination ?? 0) * 100).toFixed(1)}%`}
              sub="Ground-truth positive ratio"
            />
            <StatCard
              label="Benchmark Standard"
              value="ROC-AUC + PR-AUC"
              sub="Evaluated on genuine binary labels"
            />
          </div>

          {/* Leaderboard Table */}
          <Card className="overflow-hidden">
            <div className="border-b border-border p-4 flex items-center justify-between">
              <div>
                <SectionLabel>Detector Leaderboard & Sensitivity Ranking</SectionLabel>
                <p className="text-xs text-subtle-foreground mt-0.5">
                  Models sorted by discrimination performance on {benchmarkResult.anomaly_type} anomalies.
                </p>
              </div>
              {benchmarkResult.best_model && (
                <Badge tone="good" className="flex items-center gap-1 font-semibold">
                  <Trophy size={11} />
                  Top: {benchmarkResult.best_model}
                </Badge>
              )}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border bg-background text-subtle-foreground">
                    <th className="py-2.5 px-4 font-semibold w-16">Rank</th>
                    <th className="py-2.5 px-4 font-semibold">Detector Algorithm</th>
                    <th className="py-2.5 px-4 font-semibold">ROC-AUC</th>
                    <th className="py-2.5 px-4 font-semibold">PR-AUC</th>
                    <th className="py-2.5 px-4 font-semibold">Precision@K</th>
                    <th className="py-2.5 px-4 font-semibold">Performance Bar</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {resultsTable.map((row) => {
                    const isTop = row.rank === 1;
                    const aucPct = Math.min(Math.max(row.roc_auc * 100, 0), 100);

                    return (
                      <tr
                        key={row.model}
                        className={cn(
                          "transition-colors hover:bg-surface-muted/40",
                          isTop && "bg-primary/5 font-semibold"
                        )}
                      >
                        <td className="py-3 px-4 font-mono">
                          {isTop ? (
                            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-primary-foreground">
                              1
                            </span>
                          ) : (
                            `#${row.rank}`
                          )}
                        </td>
                        <td className="py-3 px-4 text-foreground flex items-center gap-2">
                          {row.model}
                          {isTop && (
                            <span className="text-[10px] uppercase tracking-wide text-primary font-bold">
                              Winner
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4 font-mono tabular-nums text-foreground">
                          {row.roc_auc.toFixed(4)}
                        </td>
                        <td className="py-3 px-4 font-mono tabular-nums text-foreground">
                          {row.pr_auc.toFixed(4)}
                        </td>
                        <td className="py-3 px-4 font-mono tabular-nums text-muted-foreground">
                          {row.precision_at_k.toFixed(4)}
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-28 rounded-full bg-border overflow-hidden">
                              <div
                                className={cn(
                                  "h-full rounded-full transition-all",
                                  isTop ? "bg-primary" : "bg-border-strong"
                                )}
                                style={{ width: `${aucPct}%` }}
                              />
                            </div>
                            <span className="text-[10px] text-subtle-foreground tabular-nums font-mono">
                              {aucPct.toFixed(1)}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {!benchmarkResult && !loading && (
        <Card className="p-8 text-center">
          <FlaskConical size={28} className="mx-auto text-subtle-foreground" />
          <h3 className="mt-2 text-sm font-semibold text-foreground">No Benchmark Executed Yet</h3>
          <p className="mt-1 text-xs text-muted-foreground max-w-md mx-auto">
            Choose a perturbation geometry above and run the experiment to see how your detectors handle extreme values, subspace violations, or dense cluster outliers.
          </p>
        </Card>
      )}
    </div>
  );
};

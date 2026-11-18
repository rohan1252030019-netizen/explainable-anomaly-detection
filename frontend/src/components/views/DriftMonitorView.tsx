"use client";

import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Database,
  Info,
  RefreshCw,
  Sliders,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api, isAbortError, type DriftReport } from "@/lib/api";
import { Card, SectionLabel } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { Spinner } from "@/components/ui/Feedback";
import { StatCard } from "@/components/ui/StatCard";
import { Badge } from "@/components/ui/Badge";
import { useToast } from "@/components/ui/Toast";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

export const DriftMonitorView: React.FC = () => {
  const { dataset } = useApp();
  const { toast } = useToast();

  const [loading, setLoading] = useState(true);
  const [report, setReport] = useState<DriftReport | null>(null);
  const [psiBins, setPsiBins] = useState(10);

  // The fetch runs through promise callbacks so state is only updated after the
  // request settles (react-hooks/set-state-in-effect forbids synchronous
  // setState from an effect body, which caused a cascading render on mount).
  // The AbortController is passed into fetch, so leaving the view cancels the
  // request instead of just ignoring its result.
  useEffect(() => {
    if (!dataset?.loaded) return;
    const controller = new AbortController();

    api
      .getDriftStatus({ signal: controller.signal })
      .then((res) => {
        if (res && res.overall_severity) {
          setReport(res);
        }
      })
      .catch((err: unknown) => {
        // An abort is the user navigating away.  Anything else used to be
        // swallowed here, which hid a failing endpoint behind an empty panel.
        if (isAbortError(err)) return;
        toast("error", "Could not load the drift report", errorMessage(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [dataset?.loaded, toast]);

  const handleEvaluate = async () => {
    try {
      setLoading(true);
      const res = await api.evaluateDrift({ psi_bins: psiBins });
      setReport(res);
      toast(
        "success",
        "Drift analysis complete",
        `Severity: ${res.overall_severity} (${
          (res.features_high_drift?.length ?? 0) +
          (res.features_moderate_drift?.length ?? 0)
        } features drifted)`
      );
    } catch (err) {
      toast(
        "error",
        "Failed to evaluate drift",
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
        description="Load or upload a dataset first to analyze feature and anomaly score distribution drift."
      />
    );
  }

  const reportsList = report?.feature_reports ?? [];
  const nFeaturesDrifted =
    (report?.features_high_drift?.length ?? 0) +
    (report?.features_moderate_drift?.length ?? 0);

  const severityAccent =
    report?.overall_severity === "HIGH"
      ? "var(--danger)"
      : report?.overall_severity === "MODERATE" ||
          report?.overall_severity === "LOW"
        ? "var(--caution)"
        : "var(--success)";

  return (
    <div className="space-y-6">
      {/* Top action header */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-border bg-surface p-4">
        <div>
          <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Activity size={16} className="text-primary" />
            Covariate & Score Drift Intelligence
          </h2>
          <p className="text-xs text-subtle-foreground mt-0.5">
            Compares training baseline distributions against evaluation batches using Population Stability Index (PSI) and Kolmogorov-Smirnov 2-sample tests.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Sliders size={13} className="text-subtle-foreground" />
            <span>PSI Bins:</span>
            <select
              aria-label="PSI Bins"
              value={psiBins}
              onChange={(e) => setPsiBins(Number(e.target.value))}
              disabled={loading}
              className="rounded border border-border bg-background px-2 py-1 text-xs text-foreground"
            >
              <option value={5}>5 bins</option>
              <option value={10}>10 bins (standard)</option>
              <option value={20}>20 bins</option>
            </select>
          </div>

          <Button
            onClick={handleEvaluate}
            disabled={loading}
            className="gap-2"
          >
            {loading ? <Spinner /> : <RefreshCw size={13} />}
            {loading ? "Evaluating..." : report ? "Re-evaluate Drift" : "Run Drift Analysis"}
          </Button>
        </div>
      </div>

      {report?.overall_severity && (
        <>
          {/* Summary stats */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Overall Drift Severity"
              value={report.overall_severity}
              accentColor={severityAccent}
              sub={
                report.overall_severity === "HIGH"
                  ? "Significant covariate shift detected"
                  : report.overall_severity === "MODERATE" ||
                      report.overall_severity === "LOW"
                    ? "Minor shifts observed; monitor ongoing batches"
                    : "Baseline and current distributions align"
              }
            />
            <StatCard
              label="Features Drifted"
              value={`${nFeaturesDrifted} / ${report.n_features_evaluated ?? 0}`}
              sub="Features with PSI ≥ 0.25 or significant KS p-value"
            />
            <StatCard
              label="Drift Rate"
              value={
                report.n_features_evaluated && report.n_features_evaluated > 0
                  ? `${Math.round((nFeaturesDrifted / report.n_features_evaluated) * 100)}%`
                  : "0%"
              }
              sub="Percentage of total feature matrix"
            />
            <StatCard
              label="Statistical Standard"
              value="PSI + 2-Sample KS"
              sub="Non-parametric divergence + empirical CDF difference"
            />
          </div>

          {/* Recommendations Banner */}
          {report.actions && report.actions.length > 0 && (
            <Card className="border-l-4 border-l-primary p-4">
              <div className="flex items-start gap-3">
                <Info size={16} className="text-primary shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <SectionLabel>Statistical Guidance & Remediation</SectionLabel>
                  <ul className="list-disc pl-4 text-xs text-muted-foreground space-y-1">
                    {report.actions.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </Card>
          )}

          {/* Feature Breakdown Table */}
          <Card className="overflow-hidden">
            <div className="border-b border-border p-4">
              <SectionLabel>Per-Feature Drift Diagnostics</SectionLabel>
              <p className="text-xs text-subtle-foreground mt-1">
                PSI values below 0.1 indicate stability; 0.1–0.25 indicate moderate change; &gt;0.25 indicate severe shift requiring detector recalibration.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border bg-background text-subtle-foreground">
                    <th className="py-2.5 px-4 font-semibold">Feature</th>
                    <th className="py-2.5 px-4 font-semibold">PSI Metric</th>
                    <th className="py-2.5 px-4 font-semibold">PSI Level</th>
                    <th className="py-2.5 px-4 font-semibold">KS Statistic (D)</th>
                    <th className="py-2.5 px-4 font-semibold">KS p-value</th>
                    <th className="py-2.5 px-4 font-semibold">Verdict</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {reportsList.map((item) => {
                    const isSevere = item.drift_status === "HIGH";
                    const isMod = item.drift_status === "MODERATE";
                    const metricValue =
                      item.psi !== undefined
                        ? item.psi.toFixed(4)
                        : item.js_distance !== undefined
                          ? item.js_distance.toFixed(4)
                          : "—";

                    return (
                      <tr
                        key={item.feature}
                        className={cn(
                          "transition-colors hover:bg-surface-muted/40",
                          isSevere && "bg-danger/5",
                          isMod && "bg-caution/5"
                        )}
                      >
                        <td className="py-2.5 px-4 font-mono font-medium text-foreground">
                          {item.feature}
                        </td>
                        <td className="py-2.5 px-4 font-mono tabular-nums text-foreground">
                          {metricValue}
                          {item.type === "categorical" && (
                            <span className="ml-1 text-[9px] text-subtle-foreground">JS</span>
                          )}
                        </td>
                        <td className="py-2.5 px-4">
                          <Badge
                            tone={isSevere ? "critical" : isMod ? "caution" : "good"}
                          >
                            {item.drift_status}
                          </Badge>
                        </td>
                        <td className="py-2.5 px-4 font-mono tabular-nums text-muted-foreground">
                          {item.ks_statistic !== undefined
                            ? item.ks_statistic.toFixed(4)
                            : "—"}
                        </td>
                        <td className="py-2.5 px-4 font-mono tabular-nums text-muted-foreground">
                          {item.p_value !== undefined
                            ? item.p_value < 0.001
                              ? "< 0.001"
                              : item.p_value.toFixed(4)
                            : "—"}
                        </td>
                        <td className="py-2.5 px-4">
                          {item.statistically_significant ? (
                            <span className="inline-flex items-center gap-1 font-semibold text-caution">
                              <AlertTriangle size={12} />
                              Distribution Shift
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-success">
                              <CheckCircle2 size={12} />
                              Stable
                            </span>
                          )}
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

      {!report?.overall_severity && !loading && (
        <Card className="p-8 text-center">
          <Activity size={28} className="mx-auto text-subtle-foreground" />
          <h3 className="mt-2 text-sm font-semibold text-foreground">
            No Drift Report Generated
          </h3>
          <p className="mt-1 text-xs text-muted-foreground max-w-md mx-auto">
            Click &ldquo;Run Drift Analysis&rdquo; above to calculate Population Stability Index and 2-sample Kolmogorov-Smirnov test distributions across all dataset features.
          </p>
          <div className="mt-4">
            <Button onClick={handleEvaluate} className="gap-2">
              <Activity size={13} />
              Run Drift Analysis Now
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
};

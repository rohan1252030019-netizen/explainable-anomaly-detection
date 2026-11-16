"use client";

import React, { useState } from "react";
import {
  ArrowRight,
  AlertTriangle,
  BarChart3,
  Cpu,
  Database,
  Gauge,
  ListFilter,
  Play,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { StatCard } from "@/components/ui/StatCard";
import { ScoreDistributionChart } from "@/components/charts/ScoreDistributionChart";
import { SeverityDonutChart } from "@/components/charts/SeverityDonutChart";
import { ObservationTimelineChart } from "@/components/charts/ObservationTimelineChart";
import { Button } from "@/components/ui/Button";
import { Card, SectionLabel } from "@/components/ui/Card";
import { severityStyle } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Feedback";
import { useToast } from "@/components/ui/Toast";
import { api } from "@/lib/api";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

const DETECTORS = [
  { name: "Isolation Forest", note: "Global outliers" },
  { name: "Local Outlier Factor", note: "Local density" },
  { name: "One-Class SVM", note: "Non-linear boundary" },
  { name: "Elliptic Envelope", note: "Gaussian distance" },
];

const SEVERITY_KEYS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const;

function SeverityStrip({
  counts,
}: {
  counts: { CRITICAL: number; HIGH: number; MEDIUM: number; LOW: number };
}) {
  const total = SEVERITY_KEYS.reduce((sum, key) => sum + counts[key], 0) || 1;
  const colors = {
    CRITICAL: "bg-critical",
    HIGH: "bg-warn",
    MEDIUM: "bg-caution",
    LOW: "bg-good",
  };

  return (
    <div className="space-y-1.5">
      <div
        className="flex h-1.5 overflow-hidden rounded-full bg-inset"
        role="img"
        aria-label={SEVERITY_KEYS.map(
          (key) => `${counts[key]} ${key.toLowerCase()}`
        ).join(", ")}
      >
        {SEVERITY_KEYS.map((key) =>
          counts[key] > 0 ? (
            <span
              key={key}
              className={cn("h-full", colors[key])}
              style={{ width: `${(counts[key] / total) * 100}%` }}
              aria-hidden="true"
            />
          ) : null
        )}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-ink-faint">
        {SEVERITY_KEYS.map((key) => (
          <span key={key} className="inline-flex items-center gap-1">
            <span
              className={cn("size-1.5 rounded-full", colors[key])}
              aria-hidden="true"
            />
            {key.charAt(0) + key.slice(1).toLowerCase()}{" "}
            <span className="font-semibold tabular text-ink-muted">
              {counts[key]}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

export const DashboardView: React.FC = () => {
  const { dataset, overview, setActiveTab, refreshAll } = useApp();
  const { toast } = useToast();
  const [starting, setStarting] = useState(false);

  const handleRunDemo = async () => {
    setStarting(true);
    try {
      await api.loadSample();
      await api.runDetection({
        model_name: "Isolation Forest",
        contamination: 0.05,
      });
      await refreshAll();
      toast(
        "success",
        "Sample analysis ready",
        "880 records scored with Isolation Forest at 5% contamination."
      );
    } catch (err) {
      toast("error", "Could not prepare the demo", errorMessage(err));
    } finally {
      setStarting(false);
    }
  };

  /* ---------------------------------------------------------------- *
   * State 1 — nothing loaded. Lead with the next action, not a hero.
   * ---------------------------------------------------------------- */
  if (!dataset?.loaded) {
    return (
      <div className="mx-auto max-w-4xl space-y-5">
        <Card className="overflow-hidden">
          <div className="grid grid-cols-1 gap-6 p-6 lg:grid-cols-[1.15fr_0.85fr] lg:p-7">
            <div className="flex flex-col justify-center">
              <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-accent-line bg-accent-soft px-2.5 py-1 text-[10px] font-semibold text-accent">
                <Sparkles size={11} aria-hidden="true" />
                Start with a dataset
              </span>
              <h2 className="mt-3 text-lg font-bold tracking-tight text-ink">
                No analysis loaded yet
              </h2>
              <p className="mt-2 max-w-md text-xs leading-relaxed text-ink-muted">
                Upload a tabular CSV or load the benchmark sample. XAUD profiles
                the columns, builds a feature matrix and scores every record
                without labels.
              </p>
              <div className="mt-5 flex flex-wrap items-center gap-2">
                <Button onClick={handleRunDemo} disabled={starting}>
                  {starting ? (
                    <Spinner />
                  ) : (
                    <Sparkles size={14} aria-hidden="true" />
                  )}
                  {starting ? "Preparing analysis…" : "Load benchmark demo"}
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => setActiveTab("dataset")}
                  disabled={starting}
                >
                  <Upload size={14} aria-hidden="true" />
                  Upload a CSV
                </Button>
              </div>
              <p className="mt-3 text-[10px] text-ink-faint">
                Benchmark sample: 880 records · 10 attributes · 80 synthetic
                outliers.
              </p>
            </div>

            <div className="panel-inset flex flex-col justify-center p-4">
              <SectionLabel>What happens next</SectionLabel>
              <ol className="mt-3 space-y-3">
                {[
                  {
                    icon: Database,
                    title: "Profile the dataset",
                    body: "Column types, missing values and quality checks.",
                  },
                  {
                    icon: Play,
                    title: "Fit a detector",
                    body: "Choose one of four unsupervised methods.",
                  },
                  {
                    icon: ShieldCheck,
                    title: "Review evidence",
                    body: "Scores, explanations and counterfactuals.",
                  },
                ].map((step, index) => {
                  const Icon = step.icon;
                  return (
                    <li key={step.title} className="flex items-start gap-3">
                      <span className="grid size-7 shrink-0 place-items-center rounded-md border border-edge bg-surface text-ink-faint">
                        <Icon size={13} aria-hidden="true" />
                      </span>
                      <div className="min-w-0">
                        <p className="text-[11px] font-semibold text-ink">
                          {index + 1}. {step.title}
                        </p>
                        <p className="mt-0.5 text-[10px] leading-relaxed text-ink-faint">
                          {step.body}
                        </p>
                      </div>
                    </li>
                  );
                })}
              </ol>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between gap-3">
            <SectionLabel>Available detectors</SectionLabel>
            <span className="text-[10px] text-ink-faint">Unsupervised</span>
          </div>
          <ul className="mt-3 grid grid-cols-2 gap-2 lg:grid-cols-4">
            {DETECTORS.map((detector) => (
              <li
                key={detector.name}
                className="rounded-lg border border-edge bg-inset px-3 py-2.5"
              >
                <p className="truncate text-[11px] font-semibold text-ink">
                  {detector.name}
                </p>
                <p className="mt-0.5 text-[10px] text-ink-faint">
                  {detector.note}
                </p>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    );
  }

  /* ---------------------------------------------------------------- *
   * State 2 — data is ready but nothing has been scored.
   * ---------------------------------------------------------------- */
  if (!overview?.ready) {
    return (
      <div className="mx-auto max-w-4xl space-y-5">
        <Card className="p-6">
          <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
            <div className="min-w-0">
              <span className="inline-flex items-center gap-1.5 rounded-full border border-good/25 bg-good-soft px-2.5 py-1 text-[10px] font-semibold text-good">
                <ShieldCheck size={11} aria-hidden="true" />
                Dataset ready
              </span>
              <h2 className="mt-3 truncate text-lg font-bold tracking-tight text-ink">
                {dataset.dataset_name}
              </h2>
              <p className="mt-1 text-xs text-ink-muted">
                {dataset.n_rows?.toLocaleString()} rows · {dataset.n_cols}{" "}
                columns · {dataset.n_preprocessed_features || 0} features
                available for modelling
              </p>
            </div>
            <Button onClick={() => setActiveTab("detection")} className="shrink-0">
              <Play size={14} aria-hidden="true" />
              Configure detection
            </Button>
          </div>

          <div className="mt-5 grid grid-cols-2 gap-3 border-t border-edge pt-4 sm:grid-cols-4">
            {[
              ["Numerical columns", dataset.numerical_cols?.length ?? 0],
              ["Categorical columns", dataset.categorical_cols?.length ?? 0],
              ["Missing values", dataset.total_missing ?? 0],
              ["Duplicate rows", dataset.duplicate_rows ?? 0],
            ].map(([label, value]) => (
              <div key={String(label)}>
                <SectionLabel>{label}</SectionLabel>
                <p className="mt-1 text-lg font-bold tabular text-ink">
                  {Number(value).toLocaleString()}
                </p>
              </div>
            ))}
          </div>
        </Card>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {[
            {
              icon: Database,
              title: "1. Data profiled",
              body: "Column types and quality warnings are available in the dataset explorer.",
              done: true,
            },
            {
              icon: ShieldCheck,
              title: "2. Features prepared",
              body: dataset.feature_matrix_ready
                ? "The preprocessing matrix is ready for model fitting."
                : "Open the dataset screen to build the feature matrix.",
              done: !!dataset.feature_matrix_ready,
            },
            {
              icon: Play,
              title: "3. Score records",
              body: "Pick a detector and threshold strategy in the detection workflow.",
              done: false,
            },
          ].map((step) => {
            const Icon = step.icon;
            return (
              <Card key={step.title} className="p-4">
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      "grid size-7 place-items-center rounded-md border",
                      step.done
                        ? "border-good/25 bg-good-soft text-good"
                        : "border-edge bg-inset text-ink-faint"
                    )}
                  >
                    <Icon size={13} aria-hidden="true" />
                  </span>
                  <p className="text-[11px] font-bold text-ink">{step.title}</p>
                </div>
                <p className="mt-2 text-[11px] leading-relaxed text-ink-muted">
                  {step.body}
                </p>
              </Card>
            );
          })}
        </div>

        {dataset.warnings && dataset.warnings.length > 0 && (
          <div
            role="status"
            className="flex items-start gap-3 rounded-xl border border-caution/30 bg-caution-soft p-4 text-xs"
          >
            <AlertTriangle
              size={15}
              className="mt-0.5 shrink-0 text-caution"
              aria-hidden="true"
            />
            <div>
              <p className="font-semibold text-ink">
                {dataset.warnings.length} dataset quality{" "}
                {dataset.warnings.length === 1 ? "warning" : "warnings"}
              </p>
              <p className="mt-0.5 text-[11px] text-ink-muted">
                Review them in the dataset explorer before drawing conclusions.
              </p>
            </div>
          </div>
        )}
      </div>
    );
  }

  /* ---------------------------------------------------------------- *
   * State 3 — scored results.
   * ---------------------------------------------------------------- */
  const total = overview.n_total || 0;
  const anomalies = overview.n_anomalies || 0;
  const ratePct = (overview.anomaly_rate || 0) * 100;
  const severityCounts = overview.severity_counts ?? {
    CRITICAL: 0,
    HIGH: 0,
    MEDIUM: 0,
    LOW: 0,
  };
  const critCount = severityCounts.CRITICAL;
  const highCount = severityCounts.HIGH;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Records analysed"
          value={total.toLocaleString()}
          sub={`${dataset.n_preprocessed_features || 0} features scored`}
          icon={Database}
        />
        <StatCard
          label="Anomalies flagged"
          value={anomalies.toLocaleString()}
          sub={`${ratePct.toFixed(2)}% of records`}
          icon={ShieldAlert}
          accentColor="var(--critical)"
        />
        <StatCard
          label="Priority review"
          value={(critCount + highCount).toLocaleString()}
          sub={`${critCount} critical · ${highCount} high`}
          icon={AlertTriangle}
          accentColor="var(--warn)"
        />
        <StatCard
          label="Median anomaly score"
          value={(overview.median_anomaly_score ?? 0).toFixed(3)}
          sub={`Cutoff ${(overview.threshold_val ?? 0).toFixed(3)} · ${
            overview.threshold_method ?? "contamination"
          }`}
          icon={Gauge}
        />
      </div>

      <Card className="flex flex-col gap-4 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-2.5">
            <span className="grid size-8 place-items-center rounded-lg border border-edge bg-inset text-accent">
              <Cpu size={15} aria-hidden="true" />
            </span>
            <div>
              <SectionLabel>Active detector</SectionLabel>
              <p className="text-xs font-semibold text-ink">
                {overview.model_name}
              </p>
            </div>
          </div>
          <div className="hidden h-8 w-px bg-edge sm:block" aria-hidden="true" />
          <div>
            <SectionLabel>Threshold</SectionLabel>
            <p className="text-xs font-semibold text-ink">
              {overview.threshold_method ?? "contamination"} ·{" "}
              <span className="tabular">
                {(overview.threshold_val ?? 0).toFixed(4)}
              </span>
            </p>
          </div>
          <div className="hidden h-8 w-px bg-edge sm:block" aria-hidden="true" />
          <div>
            <SectionLabel>Severity spread</SectionLabel>
            <div className="mt-1">
              <SeverityStrip counts={severityCounts} />
            </div>
          </div>
        </div>

        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setActiveTab("rigor")}
          >
            <ShieldCheck size={13} aria-hidden="true" />
            Validate
          </Button>
          <Button size="sm" onClick={() => setActiveTab("anomalies")}>
            Inspect anomalies
            <ArrowRight size={13} aria-hidden="true" />
          </Button>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="p-5 lg:col-span-2">
          <ScoreDistributionChart
            data={overview.score_histogram || []}
            threshold={overview.threshold_val}
            modelName={overview.model_name}
          />
        </Card>

        <Card className="flex flex-col p-5">
          <SeverityDonutChart counts={severityCounts} />
          {overview.severity_bands?.levels?.length ? (
            <ul className="mt-3 space-y-1.5 border-t border-edge pt-3">
              {overview.severity_bands.levels.map((band) => (
                <li
                  key={band.severity}
                  className="flex items-baseline gap-2 text-[10px]"
                  title={band.definition}
                >
                  <span
                    className="w-14 shrink-0 font-bold"
                    style={{ color: severityStyle(band.severity).hex }}
                  >
                    {band.severity}
                  </span>
                  <span className="flex-1 truncate text-ink-faint">
                    {band.definition}
                  </span>
                  <span className="shrink-0 tabular text-ink-muted">
                    {band.count}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 border-t border-edge pt-3 text-center text-[10px] text-ink-faint">
              Tiers are assigned by score quantile within the flagged set.
            </p>
          )}
        </Card>
      </div>

      <Card className="p-5">
        <ObservationTimelineChart
          data={overview.score_timeline || []}
          threshold={overview.threshold_val}
        />
      </Card>

      {overview.score_normalization && (
        <Card className="flex items-start gap-3 p-4">
          <BarChart3
            size={15}
            className="mt-0.5 shrink-0 text-ink-faint"
            aria-hidden="true"
          />
          <p className="text-[11px] leading-relaxed text-ink-muted">
            <span className="font-semibold text-ink">Score scale · </span>
            Scores are normalised with the{" "}
            <span className="font-semibold text-ink">
              {overview.score_normalization.method}
            </span>{" "}
            method between the{" "}
            {Math.round(overview.score_normalization.low_quantile * 100)}th and{" "}
            {Math.round(overview.score_normalization.high_quantile * 100)}th
            percentiles. A score of 1.0 marks the most anomalous{" "}
            {Math.round((1 - overview.score_normalization.high_quantile) * 100)}%
            of records, not an isolated extreme value.
          </p>
        </Card>
      )}

      <Card className="flex flex-col items-start justify-between gap-4 border-accent-line bg-accent-soft/40 p-4 sm:flex-row sm:items-center">
        <div className="flex items-start gap-3">
          <ListFilter
            size={16}
            className="mt-0.5 shrink-0 text-accent"
            aria-hidden="true"
          />
          <div>
            <p className="text-xs font-bold text-ink">
              {anomalies.toLocaleString()} records are waiting for review
            </p>
            <p className="mt-0.5 text-[11px] text-ink-muted">
              Rank by score, then open a record for feature evidence and
              counterfactuals.
            </p>
          </div>
        </div>
        <Button size="sm" onClick={() => setActiveTab("anomalies")}>
          Open anomalies
          <ArrowRight size={13} aria-hidden="true" />
        </Button>
      </Card>
    </div>
  );
};

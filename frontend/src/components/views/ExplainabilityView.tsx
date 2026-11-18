"use client";

import React, { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  BarChart2,
  CheckCircle2,
  Info,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api, type GlobalImportanceResponse } from "@/lib/api";
import { StatCard } from "@/components/ui/StatCard";
import { Button } from "@/components/ui/Button";
import { Card, SectionLabel } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonRows, Spinner } from "@/components/ui/Feedback";
import { useToast } from "@/components/ui/Toast";
import { severityStyle } from "@/components/ui/Badge";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

interface FeatureDistribution {
  feature: string;
  normal_median: number;
  anomaly_median: number;
  separation: number;
}

export const ExplainabilityView: React.FC = () => {
  const { overview, setActiveTab } = useApp();
  const { toast } = useToast();

  const [globalLoading, setGlobalLoading] = useState(true);
  const [data, setData] = useState<GlobalImportanceResponse | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  const [useShap, setUseShap] = useState(false);
  const [topK, setTopK] = useState(8);
  const [generating, setGenerating] = useState(false);

  const [selectedFeature, setSelectedFeature] = useState("");
  const [distData, setDistData] = useState<FeatureDistribution | null>(null);
  const [distLoading, setDistLoading] = useState(false);
  const [distError, setDistError] = useState<string | null>(null);
  const [distReloadToken, setDistReloadToken] = useState(0);

  const ready = !!overview?.ready;

  const adoptImportance = useCallback((res: GlobalImportanceResponse) => {
    setData(res);
    if (res.global_importance?.length) {
      setSelectedFeature((prev) =>
        res.global_importance!.some((item) => item.feature === prev)
          ? prev
          : res.global_importance![0].feature
      );
    }
  }, []);

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;

    api
      .getGlobalImportance()
      .then((res) => {
        if (cancelled) return;
        adoptImportance(res);
        setGlobalError(null);
        setGlobalLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setGlobalError(errorMessage(err));
        setGlobalLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [ready, reloadToken, adoptImportance]);

  useEffect(() => {
    if (!selectedFeature) return;
    let cancelled = false;

    api
      .getFeatureDistribution(selectedFeature)
      .then((res) => {
        if (cancelled) return;
        setDistData(res);
        setDistError(null);
        setDistLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setDistError(errorMessage(err));
        setDistData(null);
        setDistLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedFeature, distReloadToken]);

  const selectFeature = (feature: string) => {
    if (feature === selectedFeature) return;
    setSelectedFeature(feature);
    setDistData(null);
    setDistError(null);
    setDistLoading(true);
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setGlobalError(null);
    try {
      const generated = await api.generateExplanations({
        use_shap: useShap,
        top_k: topK,
      });
      const res = await api.getGlobalImportance();
      adoptImportance(res);
      const count =
        generated.n_explained ?? res.n_explained ?? overview?.n_anomalies ?? 0;
      toast(
        "success",
        "Explanations generated",
        `${count.toLocaleString()} ${
          count === 1 ? "record was" : "records were"
        } explained across the active run.`
      );
    } catch (err) {
      const message = errorMessage(err);
      setGlobalError(message);
      toast("error", "Could not generate explanations", message);
    } finally {
      setGenerating(false);
      setGlobalLoading(false);
    }
  };

  if (!ready) {
    return (
      <EmptyState
        icon={Sparkles}
        title="No explanations to analyse"
        description="Run anomaly detection first. Global attributions compare the features behind flagged records with the normal population."
        actionText="Configure detection"
        onAction={() => setActiveTab("detection")}
      />
    );
  }

  const importance = data?.global_importance ?? [];
  const hasImportance = !!data?.ready && importance.length > 0;
  const maxScore = Math.max(
    ...importance.map((item) => item.mean_combined_score),
    0.0001
  );
  const distBusy =
    distLoading || (!!selectedFeature && !distData && !distError);

  return (
    <div className="space-y-5">
      {/* Configuration */}
      <Card className="flex flex-col items-start justify-between gap-4 p-5 lg:flex-row lg:items-center">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Sparkles size={14} className="text-primary" aria-hidden="true" />
            <SectionLabel>Attribution configuration</SectionLabel>
          </div>
          <p className="mt-1.5 max-w-2xl text-[11px] leading-relaxed text-muted-foreground">
            Decompose model decisions across the whole population to find which
            features separate outliers from normal observations. Regenerating
            replaces the current attribution set.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <label className="flex cursor-pointer select-none items-center gap-2 rounded-lg border border-border bg-surface-muted px-3 py-1.5 text-[11px] text-muted-foreground">
            <input
              type="checkbox"
              checked={useShap}
              onChange={(event) => setUseShap(event.target.checked)}
              className="size-3.5 rounded"
            />
            Use SHAP
          </label>

          <div className="flex items-center gap-2">
            <label htmlFor="top-k" className="text-[11px] text-muted-foreground">
              Top K
            </label>
            <select
              id="top-k"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              className="field h-8 w-auto py-1"
            >
              {[5, 8, 12, 20].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </div>

          <Button onClick={handleGenerate} disabled={generating}>
            {generating ? (
              <Spinner />
            ) : (
              <Sparkles size={14} aria-hidden="true" />
            )}
            {generating ? "Computing…" : "Generate explanations"}
          </Button>
        </div>
      </Card>

      {globalLoading ? (
        <Card className="p-6">
          <SkeletonRows rows={6} />
        </Card>
      ) : globalError ? (
        <Card
          role="alert"
          className="flex flex-col items-center gap-3 p-10 text-center"
        >
          <span className="grid size-10 place-items-center rounded-xl border border-danger-line bg-danger-soft text-danger">
            <AlertTriangle size={18} aria-hidden="true" />
          </span>
          <div>
            <p className="text-sm font-bold text-foreground">
              Could not load feature attributions
            </p>
            <p className="mx-auto mt-1 max-w-lg break-words text-xs leading-relaxed text-muted-foreground">
              {globalError}
            </p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              setGlobalLoading(true);
              setGlobalError(null);
              setReloadToken((token) => token + 1);
            }}
          >
            <RefreshCw size={13} aria-hidden="true" />
            Retry
          </Button>
        </Card>
      ) : hasImportance ? (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <StatCard
              label="Anomalies analysed"
              value={data?.n_explained ?? overview?.n_anomalies ?? 0}
              sub="Individual explanations generated"
              icon={CheckCircle2}
            />
            <StatCard
              label="Primary driver"
              value={data?.top_driver || "—"}
              sub="Highest combined impact score"
              icon={Sparkles}
              accentColor={severityStyle("CRITICAL").hex}
            />
            <StatCard
              label="Configured method"
              value={useShap ? "SHAP" : "Perturbation + robust Z"}
              sub="Applied when attributions are generated"
              icon={BarChart2}
            />
          </div>

          <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-3">
            <Card className="p-5 lg:col-span-2" aria-busy={generating}>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <SectionLabel>Top anomaly-driving features</SectionLabel>
                <span className="text-[10px] text-subtle-foreground">
                  Click a feature to inspect its distribution
                </span>
              </div>

              <ul className="mt-3 space-y-2">
                {importance.slice(0, topK).map((item, index) => {
                  const pct = (item.mean_combined_score / maxScore) * 100;
                  const isSelected = selectedFeature === item.feature;

                  return (
                    <li key={item.feature}>
                      <button
                        type="button"
                        onClick={() => selectFeature(item.feature)}
                        aria-pressed={isSelected}
                        className={cn(
                          "w-full rounded-lg border p-3 text-left transition-[border-color,background-color]",
                          isSelected
                            ? "border-primary bg-primary-soft"
                            : "border-border hover:border-primary hover:bg-surface-muted"
                        )}
                      >
                        <span className="mb-1.5 flex flex-wrap items-center justify-between gap-2 text-xs">
                          <span className="flex min-w-0 items-center gap-2 font-bold text-foreground">
                            <span className="text-primary tabular">
                              #{index + 1}
                            </span>
                            <span className="truncate">{item.feature}</span>
                          </span>
                          <span className="flex items-center gap-3 text-[11px] text-muted-foreground">
                            <span
                              className="tabular"
                              title={`Named in ${item.frequency} of ${
                                data?.n_explained ?? 0
                              } explanations`}
                            >
                              {item.frequency} appearances
                            </span>
                            {typeof item.mean_z_score === "number" && (
                              <span
                                className="font-mono tabular text-subtle-foreground"
                                title="Mean robust z-score"
                              >
                                z {item.mean_z_score.toFixed(2)}
                              </span>
                            )}
                            <span className="font-mono font-bold tabular text-primary">
                              {item.mean_combined_score.toFixed(3)}
                            </span>
                          </span>
                        </span>
                        <span
                          className="block h-1.5 overflow-hidden rounded-full bg-surface-muted"
                          role="meter"
                          aria-valuemin={0}
                          aria-valuemax={Number(maxScore.toFixed(3))}
                          aria-valuenow={Number(
                            item.mean_combined_score.toFixed(3)
                          )}
                          aria-label={`${item.feature} combined importance`}
                        >
                          <motion.span
                            initial={{ width: 0 }}
                            animate={{ width: `${pct}%` }}
                            transition={{
                              duration: 0.5,
                              delay: index * 0.04,
                              ease: [0.16, 1, 0.3, 1],
                            }}
                            className="block h-full rounded-full bg-gradient-to-r from-primary to-danger"
                          />
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </Card>

            <Card className="p-5">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <SectionLabel>Distribution separation</SectionLabel>
                  <h3 className="mt-2 truncate text-sm font-bold text-foreground">
                    {selectedFeature || "Select a feature"}
                  </h3>
                </div>
                {distData && !distBusy && (
                  <span className="shrink-0 rounded-full border border-border bg-surface-muted px-2 py-0.5 font-mono text-[10px] font-semibold tabular text-muted-foreground">
                    Δ {distData.separation}
                  </span>
                )}
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                Empirical medians for the normal cluster versus flagged records.
              </p>

              <div className="mt-4" aria-busy={distBusy}>
                {distBusy ? (
                  <SkeletonRows rows={2} />
                ) : distError ? (
                  <div className="rounded-lg border border-danger-line bg-danger-soft p-3">
                    <p className="text-[10px] font-semibold text-foreground">
                      Distribution unavailable
                    </p>
                    <p className="mt-1 break-words text-[10px] leading-relaxed text-muted-foreground">
                      {distError}
                    </p>
                    <button
                      type="button"
                      onClick={() => {
                        setDistError(null);
                        setDistLoading(true);
                        setDistReloadToken((token) => token + 1);
                      }}
                      className="mt-2 inline-flex items-center gap-1 text-[10px] font-semibold text-primary hover:text-foreground"
                    >
                      <RefreshCw size={10} aria-hidden="true" />
                      Retry
                    </button>
                  </div>
                ) : distData ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-center">
                      <div className="panel-inset border-info-line bg-info-soft p-3">
                        <div className="text-[10px] font-bold uppercase tracking-wider text-info">
                          Normal median
                        </div>
                        <div className="mt-0.5 text-base font-bold tabular text-foreground">
                          {distData.normal_median}
                        </div>
                      </div>
                      <div className="panel-inset border-danger-line bg-danger-soft p-3">
                        <div className="text-[10px] font-bold uppercase tracking-wider text-danger">
                          Anomaly median
                        </div>
                        <div className="mt-0.5 text-base font-bold tabular text-foreground">
                          {distData.anomaly_median}
                        </div>
                      </div>
                    </div>

                    <div className="panel-inset px-3 py-2">
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-[10px] font-bold uppercase tracking-wider text-subtle-foreground">
                          Distribution shift
                        </span>
                        <span className="text-lg font-extrabold tabular text-caution">
                          Δ {distData.separation}
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-[11px] text-subtle-foreground">
                    Choose a feature to load its distribution.
                  </p>
                )}
              </div>

              <p className="mt-4 border-t border-border pt-3 text-[10px] leading-relaxed text-subtle-foreground">
                A larger positive shift means flagged records sit further from
                the normal median on this feature.
              </p>
            </Card>
          </div>
        </>
      ) : (
        <Card className="flex flex-col items-center gap-4 p-10 text-center">
          <span className="grid size-10 place-items-center rounded-xl border border-border bg-surface-muted text-subtle-foreground">
            <Sparkles size={18} aria-hidden="true" />
          </span>
          <div>
            <p className="text-sm font-bold text-foreground">
              Global attributions have not been computed
            </p>
            <p className="mx-auto mt-1 max-w-md text-xs leading-relaxed text-muted-foreground">
              Generate explanations to rank the features that most often drive
              this run&apos;s anomaly scores. The operation can take a moment on
              large datasets.
            </p>
          </div>
          <Button onClick={handleGenerate} disabled={generating}>
            {generating ? (
              <Spinner />
            ) : (
              <Sparkles size={14} aria-hidden="true" />
            )}
            {generating ? "Computing…" : "Compute feature attributions"}
          </Button>
        </Card>
      )}

      <Card className="flex items-start gap-3 p-4">
        <Info
          size={14}
          className="mt-0.5 shrink-0 text-primary"
          aria-hidden="true"
        />
        <div>
          <p className="text-[11px] font-bold text-foreground">Reading these scores</p>
          <p className="mt-0.5 text-[10px] leading-relaxed text-muted-foreground">
            Feature contributions reflect mathematical deviations and
            sensitivity gradients from the learned normal boundary. They provide
            statistical transparency into model scoring rather than causal
            proof. Frequency counts how often a feature appears in the ranked
            explanation for a flagged record.
          </p>
        </div>
      </Card>
    </div>
  );
};

"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FlaskConical,
  Info,
  RefreshCw,
  Repeat,
  ShieldCheck,
  TrendingDown,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import {
  api,
  type HoldoutComparison,
  type HoldoutMetrics,
  type StabilityReport,
} from "@/lib/api";
import { Card, CardHeader, SectionLabel } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonRows, Spinner } from "@/components/ui/Feedback";
import { StatCard } from "@/components/ui/StatCard";
import { useToast } from "@/components/ui/Toast";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

import {
  DeltaChip,
  InlineError,
  MetricRow,
  TEST_SIZES,
  TransferPanel,
  fmt,
  isNumber,
  pct,
  signed,
} from "./rigor/widgets";

export const RigorView: React.FC = () => {
  const { overview, dataset } = useApp();
  const { toast } = useToast();

  const [testSize, setTestSize] = useState(0.3);
  const [holdout, setHoldout] = useState<HoldoutMetrics | null>(null);
  const [comparison, setComparison] = useState<HoldoutComparison | null>(null);
  const [stability, setStability] = useState<StabilityReport | null>(null);

  const [busy, setBusy] = useState<"holdout" | "comparison" | "stability" | null>(
    null
  );
  const [holdoutError, setHoldoutError] = useState<string | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [stabilityError, setStabilityError] = useState<string | null>(null);

  const detectionReady = Boolean(overview?.ready);
  const runtimeKey = useMemo(
    () => `${dataset?.dataset_name ?? "none"}::${overview?.model_name ?? "none"}`,
    [dataset?.dataset_name, overview?.model_name]
  );

  // The report belongs to the dataset/model pair it was computed for. Tagging
  // it means a dataset swap never leaves stale numbers on screen.
  const [holdoutKey, setHoldoutKey] = useState<string | null>(null);
  const currentHoldout = holdoutKey === runtimeKey ? holdout : null;

  const autoLoaded = useRef<{ key: string; size: number } | null>(null);
  const holdoutRequest = useRef(0);

  const loadHoldout = useCallback(
    async (size: number, announce = false) => {
      const requestId = ++holdoutRequest.current;
      autoLoaded.current = { key: runtimeKey, size };
      setBusy("holdout");
      setHoldoutError(null);

      try {
        const data = await api.getHoldoutMetrics(size);
        if (requestId !== holdoutRequest.current) return;

        setHoldout(data);
        setHoldoutKey(runtimeKey);

        if (announce) {
          if (data.available) {
            toast(
              "success",
              "Held-out report updated",
              `Fitted on ${data.n_fit?.toLocaleString()} rows and scored on ${
                data.n_test?.toLocaleString() ?? "held-out"
              } unseen rows.`
            );
          } else {
            toast(
              "info",
              "Held-out report updated",
              data.reason || "Accuracy metrics are unavailable for this split."
            );
          }
        }
      } catch (error) {
        if (requestId === holdoutRequest.current) {
          setHoldoutError(errorMessage(error));
        }
      } finally {
        if (requestId === holdoutRequest.current) setBusy(null);
      }
    },
    [runtimeKey, toast]
  );

  const runComparison = useCallback(async () => {
    setBusy("comparison");
    setComparisonError(null);

    try {
      const data = await api.getHoldoutComparison(testSize);
      setComparison(data);

      let best: { model: string; value: number } | null = null;
      for (const row of data.table) {
        const model = typeof row.Model === "string" ? row.Model : null;
        const value = row["ROC-AUC (out-of-sample)"];
        if (
          model &&
          isNumber(value) &&
          (best === null || value > best.value)
        ) {
          best = { model, value };
        }
      }

      toast(
        "success",
        "Held-out comparison complete",
        best
          ? `${best.model} leads on held-out ROC-AUC at ${best.value.toFixed(3)}.`
          : "Every detector was fitted on the same training split."
      );
    } catch (error) {
      setComparisonError(errorMessage(error));
    } finally {
      setBusy(null);
    }
  }, [testSize, toast]);

  const runStability = useCallback(async () => {
    setBusy("stability");
    setStabilityError(null);

    try {
      const data = await api.getExplanationStability();
      setStability(data);

      if (data.n_fits < 2) {
        toast(
          "info",
          "Stability could not be measured",
          data.verdict || "At least two successful refits are required."
        );
      } else {
        toast(
          "success",
          "Stability measured",
          data.verdict || `${data.n_fits} refits compared.`
        );
      }
    } catch (error) {
      setStabilityError(errorMessage(error));
    } finally {
      setBusy(null);
    }
  }, [toast]);

  // Auto-load the held-out report once per dataset/model pair. All state
  // updates happen after the promise settles, never synchronously in the
  // effect body.
  useEffect(() => {
    if (!detectionReady) return;
    if (
      autoLoaded.current?.key === runtimeKey &&
      autoLoaded.current?.size === testSize
    ) {
      return;
    }

    autoLoaded.current = { key: runtimeKey, size: testSize };
    const requestId = ++holdoutRequest.current;

    // No cancellation flag here: React's development Strict Mode tears down and
    // re-runs the effect, and the second run is intentionally skipped because
    // the same split is already in flight. The request-id guard is what keeps
    // stale responses out, including after a dataset change or a manual rerun.
    void api
      .getHoldoutMetrics(testSize)
      .then((data) => {
        if (requestId !== holdoutRequest.current) return;
        setHoldout(data);
        setHoldoutKey(runtimeKey);
      })
      .catch((error) => {
        if (requestId !== holdoutRequest.current) return;
        setHoldoutError(errorMessage(error));
      });
  }, [detectionReady, runtimeKey, testSize]);

  const bestModel = useMemo(() => {
    let best: { model: string; value: number } | null = null;
    for (const row of comparison?.table ?? []) {
      const model = typeof row.Model === "string" ? row.Model : null;
      const value = row["ROC-AUC (out-of-sample)"];
      if (model && isNumber(value) && (best === null || value > best.value)) {
        best = { model, value };
      }
    }
    return best;
  }, [comparison]);

  if (!dataset?.loaded) {
    return (
      <EmptyState
        icon={ShieldCheck}
        title="No dataset loaded"
        description="Load a dataset and run detection before checking whether the reported metrics hold up out of sample."
      />
    );
  }

  if (!detectionReady) {
    return (
      <EmptyState
        icon={FlaskConical}
        title="Run detection first"
        description="Held-out metrics and explanation fidelity are only meaningful once a detector has scored the data."
      />
    );
  }

  const ins = currentHoldout?.in_sample;
  const oos = currentHoldout?.out_of_sample;
  const gaps = currentHoldout?.optimism_gap;
  const rocGap = gaps?.roc_auc;

  const gapTone =
    !isNumber(rocGap)
      ? "var(--subtle-foreground)"
      : rocGap <= 0.02
        ? "var(--success)"
        : rocGap <= 0.06
          ? "var(--caution)"
          : "var(--danger)";

  const flagDrift = currentHoldout?.flag_rate_drift;
  const flagDriftTone =
    !isNumber(flagDrift)
      ? "var(--subtle-foreground)"
      : Math.abs(flagDrift) <= 0.01
        ? "var(--success)"
        : Math.abs(flagDrift) <= 0.05
          ? "var(--caution)"
          : "var(--danger)";

  const stabilityTone =
    !isNumber(stability?.mean_rank_correlation)
      ? "border-border bg-surface-muted text-muted-foreground"
      : stability.mean_rank_correlation >= 0.8
        ? "border-success-line bg-success-soft text-muted-foreground"
        : stability.mean_rank_correlation >= 0.5
          ? "border-caution-line bg-caution-soft text-muted-foreground"
          : "border-danger-line bg-danger-soft text-muted-foreground";

  const comparisonStale =
    comparison !== null &&
    Math.abs(comparison.test_size - testSize) > 0.0001;

  return (
    <div className="space-y-5">
      {/* ── Why this page exists ───────────────────────────────────────── */}
      <Card className="relative overflow-hidden p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
          <span className="grid size-9 shrink-0 place-items-center rounded-lg border border-primary bg-primary-soft text-primary">
            <ShieldCheck size={17} aria-hidden="true" />
          </span>

          <div className="min-w-0 flex-1">
            <SectionLabel>Out-of-sample check</SectionLabel>
            <h2 className="mt-1 text-sm font-bold tracking-tight text-foreground">
              Every metric elsewhere is computed in-sample
            </h2>
            <p className="mt-1.5 max-w-3xl text-xs leading-relaxed text-muted-foreground">
              The detectors are fitted and then asked to score the very same
              rows, which flatters them. Here the model is fitted on one split
              and scored on rows it has never seen. Explanations get the same
              treatment: an attribution that does not move the score is worth
              less than one that does.
            </p>
          </div>

          <dl className="grid shrink-0 grid-cols-2 gap-x-6 gap-y-2 border-border sm:border-l sm:pl-5">
            <div>
              <dt className="text-[10px] uppercase tracking-wider text-subtle-foreground">
                Detector
              </dt>
              <dd className="max-w-40 truncate text-[11px] font-semibold text-foreground">
                {overview?.model_name || "Active run"}
              </dd>
            </div>
            <div>
              <dt className="text-[10px] uppercase tracking-wider text-subtle-foreground">
                Ground truth
              </dt>
              <dd className="max-w-40 truncate text-[11px] font-semibold text-foreground">
                {dataset.label_col ? dataset.label_col : "No label column"}
              </dd>
            </div>
          </dl>
        </div>
      </Card>

      {/* ── Headline numbers ───────────────────────────────────────────── */}
      {currentHoldout?.available && ins && oos && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatCard
            label="Held-out ROC-AUC"
            value={fmt(oos.roc_auc)}
            sub={`In-sample ${fmt(ins.roc_auc)}`}
            icon={ShieldCheck}
            accentColor="var(--primary)"
          />
          <StatCard
            label="Optimism gap"
            value={signed(rocGap, 4)}
            sub="In-sample minus held-out"
            icon={TrendingDown}
            accentColor={gapTone}
          />
          <StatCard
            label="Flag-rate drift"
            value={signed(flagDrift, 2, true)}
            sub={`${pct(currentHoldout.flag_rate_in_sample)} → ${pct(
              currentHoldout.flag_rate_out_of_sample
            )}`}
            icon={RefreshCw}
            accentColor={flagDriftTone}
          />
          <StatCard
            label="Held-out rows"
            value={(currentHoldout.n_test ?? 0).toLocaleString()}
            sub={`${(currentHoldout.n_fit ?? 0).toLocaleString()} fitted · ${pct(
              currentHoldout.test_size ?? testSize,
              0
            )} split`}
            icon={Database}
            accentColor="var(--info)"
          />
        </div>
      )}

      {/* ── Held-out metrics ───────────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Out-of-sample detection metrics"
          description={
            dataset.label_col
              ? `Fitted on a training split and scored on held-out rows, using "${dataset.label_col}" only for scoring.`
              : "No ground-truth label column was detected, so precision, recall and AUC cannot be computed."
          }
          icon={ShieldCheck}
          action={
            <Button
              onClick={() => void loadHoldout(testSize, true)}
              disabled={busy !== null}
              size="sm"
              variant="secondary"
            >
              {busy === "holdout" ? (
                <Spinner />
              ) : (
                <RefreshCw size={13} aria-hidden="true" />
              )}
              Recompute
            </Button>
          }
        />

        <div className="p-5">
          <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-3">
            <span
              id="test-size-label"
              className="text-[11px] font-semibold text-muted-foreground"
            >
              Held-out fraction
            </span>

            <div
              role="group"
              aria-labelledby="test-size-label"
              className="inline-flex rounded-lg border border-border bg-surface-muted p-0.5"
            >
              {TEST_SIZES.map((size) => (
                <button
                  key={size}
                  type="button"
                  onClick={() => {
                    if (size === testSize) return;
                    setTestSize(size);
                    void loadHoldout(size, true);
                  }}
                  disabled={busy !== null}
                  aria-pressed={size === testSize}
                  className={cn(
                    "rounded-md px-2.5 py-1 text-[11px] font-semibold tabular transition-colors disabled:opacity-50",
                    size === testSize
                      ? "bg-surface text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {Math.round(size * 100)}%
                </button>
              ))}
            </div>

            {busy === "holdout" ? (
              <span className="inline-flex items-center gap-1.5 text-[11px] text-subtle-foreground">
                <Spinner className="size-3" />
                Recomputing at {pct(testSize, 0)}…
              </span>
            ) : holdoutError ? (
              <span className="text-[11px] text-danger">
                Held-out report unavailable
              </span>
            ) : currentHoldout ? (
              <span className="text-[11px] tabular text-subtle-foreground">
                {currentHoldout.n_fit?.toLocaleString()} train ·{" "}
                {currentHoldout.n_test?.toLocaleString()} test rows ·{" "}
                {currentHoldout.model || overview?.model_name}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 text-[11px] text-subtle-foreground">
                <Spinner className="size-3" />
                Loading held-out report…
              </span>
            )}
          </div>

          {holdoutError && (
            <div className="mb-4">
              <InlineError
                title="Held-out report unavailable"
                message={holdoutError}
                onRetry={() => void loadHoldout(testSize)}
                busy={busy === "holdout"}
                retryLabel="Recompute"
              />
            </div>
          )}

          {!currentHoldout && !holdoutError ? (
            <div aria-busy="true">
              <SkeletonRows rows={5} />
              <p className="mt-4 text-center text-[10px] text-subtle-foreground">
                Fitting the detector on the training split and scoring unseen
                rows…
              </p>
            </div>
          ) : currentHoldout && !currentHoldout.available ? (
            <div className="space-y-3">
              <div className="flex items-start gap-3 rounded-xl border border-caution-line bg-caution-soft p-3.5">
                <AlertTriangle
                  size={14}
                  className="mt-0.5 shrink-0 text-caution"
                  aria-hidden="true"
                />
                <div>
                  <p className="text-xs font-bold text-foreground">
                    Accuracy metrics cannot be computed
                  </p>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">
                    {currentHoldout.reason ||
                      currentHoldout.summary ||
                      "This split has no ground-truth anomalies or labels."}
                  </p>
                </div>
              </div>
              <TransferPanel holdout={currentHoldout} />
            </div>
          ) : currentHoldout?.available && ins && oos ? (
            <>
              <div className="overflow-x-auto" data-scroll>
                <table className="w-full min-w-[30rem]">
                  <caption className="sr-only">
                    In-sample and held-out detection metrics with optimism gaps
                  </caption>
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-subtle-foreground">
                      <th scope="col" className="pb-2 text-left font-bold">
                        Metric
                      </th>
                      <th scope="col" className="pb-2 pr-3 text-right font-bold">
                        In-sample
                      </th>
                      <th scope="col" className="pb-2 pr-3 text-right font-bold">
                        Held-out
                      </th>
                      <th scope="col" className="pb-2 text-right font-bold">
                        Gap
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <MetricRow
                      label="ROC-AUC"
                      inSample={ins.roc_auc}
                      outOfSample={oos.roc_auc}
                      gap={gaps?.roc_auc}
                    />
                    <MetricRow
                      label="PR-AUC"
                      inSample={ins.pr_auc}
                      outOfSample={oos.pr_auc}
                      gap={gaps?.pr_auc}
                    />
                    <MetricRow
                      label="Precision"
                      inSample={ins.precision}
                      outOfSample={oos.precision}
                      gap={gaps?.precision}
                    />
                    <MetricRow
                      label="Recall"
                      inSample={ins.recall}
                      outOfSample={oos.recall}
                      gap={gaps?.recall}
                    />
                    <MetricRow
                      label="F1"
                      inSample={ins.f1}
                      outOfSample={oos.f1}
                      gap={gaps?.f1}
                    />
                  </tbody>
                </table>
              </div>

              <TransferPanel holdout={currentHoldout} />

              <p className="mt-4 flex items-start gap-2 text-[11px] leading-relaxed text-subtle-foreground">
                <Info size={12} className="mt-0.5 shrink-0" aria-hidden="true" />
                <span>{currentHoldout.note}</span>
              </p>
            </>
          ) : null}
        </div>
      </Card>

      {/* ── Cross-model held-out comparison ─────────────────────────────── */}
      <Card>
        <CardHeader
          title="Held-out comparison across detectors"
          description="Every model fitted on the same training split, so the ranking reflects generalisation rather than who memorised best."
          icon={FlaskConical}
          action={
            <Button
              onClick={() => void runComparison()}
              disabled={busy !== null}
              size="sm"
              variant="secondary"
            >
              {busy === "comparison" ? (
                <Spinner />
              ) : (
                <FlaskConical size={13} aria-hidden="true" />
              )}
              Run comparison
            </Button>
          }
        />

        <div className="p-5">
          {comparisonError && (
            <div className="mb-4">
              <InlineError
                title="Comparison unavailable"
                message={comparisonError}
                onRetry={() => void runComparison()}
                busy={busy === "comparison"}
                retryLabel="Run again"
              />
            </div>
          )}

          {busy === "comparison" && !comparison ? (
            <div aria-busy="true">
              <SkeletonRows rows={4} />
              <p className="mt-4 text-center text-[10px] text-subtle-foreground">
                Fitting every detector on the training split…
              </p>
            </div>
          ) : !comparison ? (
            !comparisonError && (
              <div className="py-8 text-center">
                <span className="mx-auto grid size-10 place-items-center rounded-xl border border-border bg-surface-muted text-subtle-foreground">
                  <FlaskConical size={18} aria-hidden="true" />
                </span>
                <p className="mt-3 text-xs font-bold text-foreground">
                  No comparison run yet
                </p>
                <p className="mx-auto mt-1 max-w-md text-[11px] leading-relaxed text-muted-foreground">
                  Fit all four detectors on the same training split to see which
                  one actually generalises, not which one memorised the training
                  rows.
                </p>
              </div>
            )
          ) : (
            <>
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full border border-border bg-surface-muted px-2.5 py-1 text-[10px] font-semibold text-muted-foreground">
                    {pct(comparison.test_size, 0)} held-out ·{" "}
                    {comparison.random_state} seed
                  </span>
                  {bestModel && (
                    <span className="inline-flex items-center gap-1 rounded-full border border-primary-line bg-primary-soft px-2.5 py-1 text-[10px] font-semibold text-primary">
                      <CheckCircle2 size={11} aria-hidden="true" />
                      Best held-out: {bestModel.model} ·{" "}
                      {bestModel.value.toFixed(3)}
                    </span>
                  )}
                </div>

                {comparisonStale && (
                  <span className="text-[10px] font-semibold text-caution">
                    Computed at {pct(comparison.test_size, 0)}; run again for{" "}
                    {pct(testSize, 0)}.
                  </span>
                )}
              </div>

              <div className="overflow-x-auto" data-scroll>
                <table className="data-table min-w-[52rem] text-[11px]">
                  <caption className="sr-only">
                    Held-out metrics for every detector, sorted as returned by
                    the backend
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col" className="text-right">
                        ROC-AUC held-out
                      </th>
                      <th scope="col" className="text-right">
                        ROC-AUC in-sample
                      </th>
                      <th scope="col" className="text-right">
                        PR-AUC held-out
                      </th>
                      <th scope="col" className="text-right">
                        F1
                      </th>
                      <th scope="col" className="text-right">
                        Recall
                      </th>
                      <th scope="col" className="text-right">
                        AUC gap
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.table.map((row, index) => {
                      const model =
                        typeof row.Model === "string"
                          ? row.Model
                          : `Detector ${index + 1}`;
                      const outRoc = row["ROC-AUC (out-of-sample)"];
                      const inRoc = row["ROC-AUC (in-sample)"];
                      const outPr = row["PR-AUC (out-of-sample)"];
                      const f1 = row["F1 (out-of-sample)"];
                      const recall = row["Recall (out-of-sample)"];
                      const gap = row["AUC gap"];
                      const reason =
                        typeof row.Reason === "string"
                          ? row.Reason
                          : typeof row.Error === "string"
                            ? row.Error
                            : null;
                      const isBest = bestModel?.model === model;

                      if (!isNumber(outRoc)) {
                        return (
                          <tr key={`${model}-${index}`}>
                            <th
                              scope="row"
                              className="font-sans font-bold text-foreground"
                            >
                              {model}
                            </th>
                            <td
                              colSpan={6}
                              className="text-muted-foreground"
                              title={reason ?? undefined}
                            >
                              {reason ?? "Held-out metrics unavailable."}
                            </td>
                          </tr>
                        );
                      }

                      return (
                        <tr
                          key={`${model}-${index}`}
                          className={isBest ? "bg-primary-soft/60" : undefined}
                        >
                          <th
                            scope="row"
                            className="font-sans font-bold text-foreground"
                          >
                            <span className="flex flex-wrap items-center gap-2">
                              {model}
                              {isBest && (
                                <span className="rounded-full border border-primary-line bg-surface px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-primary">
                                  Best
                                </span>
                              )}
                            </span>
                          </th>
                          <td className="text-right font-bold tabular text-foreground">
                            {outRoc.toFixed(3)}
                          </td>
                          <td className="text-right tabular text-muted-foreground">
                            {isNumber(inRoc) ? inRoc.toFixed(3) : "—"}
                          </td>
                          <td className="text-right tabular text-muted-foreground">
                            {isNumber(outPr) ? outPr.toFixed(3) : "—"}
                          </td>
                          <td className="text-right tabular text-muted-foreground">
                            {isNumber(f1) ? f1.toFixed(3) : "—"}
                          </td>
                          <td className="text-right tabular text-muted-foreground">
                            {isNumber(recall) ? recall.toFixed(3) : "—"}
                          </td>
                          <td className="text-right">
                            <DeltaChip
                              value={isNumber(gap) ? gap : null}
                              label={`${model} AUC gap`}
                              betterWhenLower
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <p className="mt-3 flex items-start gap-2 text-[11px] leading-relaxed text-subtle-foreground">
                <Info size={12} className="mt-0.5 shrink-0" aria-hidden="true" />
                <span>{comparison.note}</span>
              </p>
            </>
          )}
        </div>
      </Card>

      {/* ── Explanation stability ──────────────────────────────────────── */}
      <Card>
        <CardHeader
          title="Explanation stability across refits"
          description="Refits the detector under several random seeds and measures how much the feature ranking moves. An explanation that only holds for one seed should not be acted on."
          icon={Repeat}
          action={
            <Button
              onClick={() => void runStability()}
              disabled={busy !== null}
              size="sm"
              variant="secondary"
            >
              {busy === "stability" ? (
                <Spinner />
              ) : (
                <Repeat size={13} aria-hidden="true" />
              )}
              Measure
            </Button>
          }
        />

        <div className="p-5">
          {stabilityError && (
            <div className="mb-4">
              <InlineError
                title="Stability could not be measured"
                message={stabilityError}
                onRetry={() => void runStability()}
                busy={busy === "stability"}
                retryLabel="Measure again"
              />
            </div>
          )}

          {busy === "stability" && !stability ? (
            <div aria-busy="true">
              <SkeletonRows rows={3} />
              <p className="mt-4 text-center text-[10px] text-subtle-foreground">
                Refitting the detector under multiple seeds…
              </p>
            </div>
          ) : !stability ? (
            !stabilityError && (
              <div className="py-8 text-center">
                <span className="mx-auto grid size-10 place-items-center rounded-xl border border-border bg-surface-muted text-subtle-foreground">
                  <Repeat size={18} aria-hidden="true" />
                </span>
                <p className="mt-3 text-xs font-bold text-foreground">
                  Stability not measured yet
                </p>
                <p className="mx-auto mt-1 max-w-md text-[11px] leading-relaxed text-muted-foreground">
                  Re-fit the detector under several seeds and compare the
                  resulting feature rankings. Low agreement means the
                  explanation is partly noise.
                </p>
              </div>
            )
          ) : !isNumber(stability.mean_rank_correlation) ||
            stability.n_fits < 2 ? (
            <div className="rounded-xl border border-caution-line bg-caution-soft p-4">
              <p className="text-xs font-bold text-foreground">
                Not enough successful refits
              </p>
              <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                {stability.verdict ||
                  "At least two successful fits are required to compare feature rankings."}
              </p>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              className="space-y-4"
            >
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <div className="rounded-lg border border-border bg-surface-muted p-3">
                  <SectionLabel>Mean rank correlation</SectionLabel>
                  <p className="mt-1 text-lg font-extrabold tabular text-foreground">
                    {fmt(stability.mean_rank_correlation)}
                  </p>
                </div>
                <div className="rounded-lg border border-border bg-surface-muted p-3">
                  <SectionLabel>Worst pair</SectionLabel>
                  <p className="mt-1 text-lg font-extrabold tabular text-foreground">
                    {fmt(stability.min_rank_correlation)}
                  </p>
                </div>
                <div className="rounded-lg border border-border bg-surface-muted p-3">
                  <SectionLabel>Top-3 retention</SectionLabel>
                  <p className="mt-1 text-lg font-extrabold tabular text-foreground">
                    {pct(stability.top_k_retention)}
                  </p>
                </div>
                <div className="rounded-lg border border-border bg-surface-muted p-3">
                  <SectionLabel>Fits compared</SectionLabel>
                  <p className="mt-1 text-lg font-extrabold tabular text-foreground">
                    {stability.n_fits}
                  </p>
                  <p className="mt-0.5 text-[10px] text-subtle-foreground">
                    {stability.n_anomalies_probed
                      ? `${stability.n_anomalies_probed} anomalies probed`
                      : "Same rows in every fit"}
                  </p>
                </div>
              </div>

              <p
                className={cn(
                  "rounded-lg border p-3 text-xs leading-relaxed",
                  stabilityTone
                )}
              >
                {stability.verdict}
              </p>

              {stability.top_k_features && stability.top_k_features.length > 0 && (
                <div>
                  <SectionLabel>Recurring leading features</SectionLabel>
                  <ul className="mt-2 space-y-1.5">
                    {stability.top_k_features.slice(0, 5).map((feature) => (
                      <li
                        key={feature.feature}
                        className="flex items-center gap-3"
                      >
                        <span
                          className="w-36 truncate text-xs text-foreground sm:w-44"
                          title={feature.feature}
                        >
                          {feature.feature}
                        </span>
                        <span
                          className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-muted"
                          aria-hidden="true"
                        >
                          <motion.span
                            className="block h-full rounded-full bg-primary"
                            initial={{ width: 0 }}
                            animate={{
                              width: `${Math.max(
                                0,
                                Math.min(100, feature.share * 100)
                              )}%`,
                            }}
                            transition={{
                              duration: 0.4,
                              ease: [0.16, 1, 0.3, 1],
                            }}
                          />
                        </span>
                        <span className="w-14 text-right text-[11px] tabular text-muted-foreground">
                          {pct(feature.share, 0)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {stability.per_fit_leader && stability.per_fit_leader.length > 0 && (
                <div>
                  <SectionLabel>Leader under each seed</SectionLabel>
                  <ul className="mt-2 flex flex-wrap gap-1.5">
                    {stability.per_fit_leader.map((leader, index) => (
                      <li
                        key={`${leader}-${index}`}
                        className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-muted px-2.5 py-1 text-[10px] text-muted-foreground"
                      >
                        <span className="font-mono text-subtle-foreground">
                          {stability.seeds?.[index] ?? index + 1}
                        </span>
                        <span className="font-semibold text-foreground">
                          {leader}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </Card>

      {/* ── Reading guide ──────────────────────────────────────────────── */}
      <Card className="p-5">
        <SectionLabel>Reading these numbers</SectionLabel>
        <ul className="mt-3 grid gap-3 text-xs leading-relaxed text-muted-foreground sm:grid-cols-2">
          <li className="rounded-lg border border-border bg-surface-muted p-3">
            <span className="font-semibold text-foreground">Held-out gap</span>{" "}
            compares a model scored on its own training rows against one scored
            on unseen rows. A small gap means the headline figure is
            trustworthy.
          </li>
          <li className="rounded-lg border border-border bg-surface-muted p-3">
            <span className="font-semibold text-foreground">Precision vs recall</span>{" "}
            at a fixed contamination rate trades false alarms against missed
            anomalies. Held-out recall well below 1 means flagged conditions are
            being missed, whatever ROC-AUC suggests.
          </li>
          <li className="rounded-lg border border-border bg-surface-muted p-3">
            <span className="font-semibold text-foreground">Stability</span> answers a
            different question from accuracy: whether you would get the same
            story if you re-ran the pipeline tomorrow.
          </li>
          <li className="rounded-lg border border-border bg-surface-muted p-3">
            <span className="font-semibold text-foreground">Domain truth</span> is not
            measured here. These checks show whether the model behaves as
            reported, which is the part that can be verified.
          </li>
        </ul>
      </Card>
    </div>
  );
};

export default RigorView;

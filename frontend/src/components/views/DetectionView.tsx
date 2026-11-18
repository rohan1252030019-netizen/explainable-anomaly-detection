"use client";

import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  CircleDot,
  Cpu,
  Database,
  Gauge,
  Layers,
  Play,
  Sliders,
  Sparkles,
  Target,
  Trees,
  Zap,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card, SectionLabel } from "@/components/ui/Card";
import { Spinner } from "@/components/ui/Feedback";
import { useToast } from "@/components/ui/Toast";
import { cn } from "@/lib/cn";
import { errorMessage } from "@/lib/errors";

const MODELS = [
  {
    id: "Isolation Forest",
    short: "IF",
    icon: Trees,
    tagline: "Recursive random partitioning",
    desc: "Isolates anomalies via random splits — outliers need fewer partitions to isolate.",
    bestFor: "Global outliers, large datasets",
    complexity: "O(n log n)",
    accent: "var(--primary)",
  },
  {
    id: "Local Outlier Factor",
    short: "LOF",
    icon: Target,
    tagline: "Density-based local scoring",
    desc: "Compares each observation's local density with its k nearest neighbours.",
    bestFor: "Local outliers, varying density clusters",
    complexity: "O(n²)",
    accent: "var(--info)",
  },
  {
    id: "One-Class SVM",
    short: "OCSVM",
    icon: Zap,
    tagline: "Kernel max-margin boundary",
    desc: "Learns a kernel hypersphere around the dense normal region in feature space.",
    bestFor: "Non-linear distributions, novelty boundary",
    complexity: "O(n²–n³)",
    accent: "var(--caution)",
  },
  {
    id: "Elliptic Envelope",
    short: "EE",
    icon: CircleDot,
    tagline: "Gaussian + Mahalanobis distance",
    desc: "Fits a robust covariance estimate and flags high Mahalanobis distances.",
    bestFor: "Approximately Gaussian features",
    complexity: "O(n · p²)",
    accent: "var(--success)",
  },
  {
    id: "k-Nearest Neighbors",
    short: "k-NN",
    icon: Zap,
    tagline: "k-th distance density outlier",
    desc: "Scores observations by Euclidean distance to the k-th nearest neighbour in scaled space.",
    bestFor: "Clustered outliers, distance-based novelty",
    complexity: "O(n log n)",
    accent: "var(--info)",
  },
  {
    id: "PCA Reconstruction",
    short: "PCA",
    icon: Layers,
    tagline: "Subspace projection residual",
    desc: "Projects observations onto principal subspace and scores orthogonal reconstruction error.",
    bestFor: "Multivariate linear subspace violations",
    complexity: "O(n · p²)",
    accent: "var(--primary)",
  },
  {
    id: "Histogram-Based Outlier Score",
    short: "HBOS",
    icon: Sliders,
    tagline: "Linear-time inverse log density",
    desc: "Builds univariate histograms assuming feature independence; ultra-fast for streaming & massive data.",
    bestFor: "High-throughput tabular data, large n",
    complexity: "O(n · p)",
    accent: "var(--success)",
  },
  {
    id: "Gaussian Mixture Model",
    short: "GMM",
    icon: Gauge,
    tagline: "Multimodal EM log-likelihood",
    desc: "Fits an Expectation-Maximization mixture of Gaussians and flags low probability density observations.",
    bestFor: "Multi-cluster distributions, multimodal data",
    complexity: "O(n · k · p)",
    accent: "var(--caution)",
  },
];

const THRESHOLD_METHODS = [
  {
    value: "contamination",
    label: "Contamination rate",
    short: "Contamination",
    hint: "Flags the top share of scores, matching the expected outlier rate.",
  },
  {
    value: "quantile",
    label: "Quantile cutoff",
    short: "Quantile",
    hint: "Uses an empirical percentile of the normalised score distribution.",
  },
  {
    value: "fixed",
    label: "Fixed score cutoff",
    short: "Fixed cutoff",
    hint: "Flags every observation at or above an absolute normalised score.",
  },
];

function RangeField({
  id,
  label,
  value,
  display,
  min,
  max,
  step,
  onChange,
  hint,
  disabled = false,
}: {
  id: string;
  label: string;
  value: number;
  display: string;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint?: string;
  disabled?: boolean;
}) {
  const progress = ((value - min) / (max - min)) * 100;
  const hintId = hint ? `${id}-hint` : undefined;

  return (
    <div className={cn(disabled && "opacity-60")}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-xs font-semibold text-foreground">
          {label}
        </label>
        <span className="rounded bg-surface-muted px-1.5 py-0.5 font-mono text-[11px] font-bold tabular text-primary">
          {display}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-valuetext={display}
        aria-describedby={hintId}
        style={{ ["--range-progress" as string]: `${progress}%` }}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && (
        <p id={hintId} className="mt-1 text-[10px] leading-relaxed text-subtle-foreground">
          {hint}
        </p>
      )}
    </div>
  );
}

export const DetectionView: React.FC = () => {
  const { dataset, overview, refreshAll, setActiveTab } = useApp();
  const { toast } = useToast();

  const [selectedModel, setSelectedModel] = useState("Isolation Forest");
  const [thresholdMethod, setThresholdMethod] = useState("contamination");
  const [contamination, setContamination] = useState(0.05);
  const [quantileVal, setQuantileVal] = useState(0.95);
  const [fixedVal, setFixedVal] = useState(0.5);

  const [nEstimators, setNEstimators] = useState(100);
  const [nNeighbors, setNNeighbors] = useState(20);
  const [svmNu, setSvmNu] = useState(0.05);
  const [svmKernel, setSvmKernel] = useState("rbf");

  const [useEnsemble, setUseEnsemble] = useState(false);
  const [ensembleMethod] = useState<"rank_borda" | "weighted_score">("rank_borda");
  const [selectedEnsembleModels] = useState<string[]>([
    "Isolation Forest",
    "Local Outlier Factor",
    "k-Nearest Neighbors",
  ]);
  const [ensembleWeights, setEnsembleWeights] = useState({
    "Isolation Forest": 0.4,
    "Local Outlier Factor": 0.3,
    "One-Class SVM": 0.3,
  });

  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runCompleted, setRunCompleted] = useState(false);

  const model = MODELS.find((item) => item.id === selectedModel) ?? MODELS[0];
  const nRows = dataset?.n_rows ?? 0;
  const nFeatures = dataset?.n_preprocessed_features ?? 0;
  const matrixReady = dataset?.loaded && dataset.feature_matrix_ready === true;
  const canRun = matrixReady && !isRunning;

  const threshold = THRESHOLD_METHODS.find(
    (method) => method.value === thresholdMethod
  );

  const targetFlags = useMemo(() => {
    if (!nRows) return null;
    if (thresholdMethod === "contamination") {
      return Math.round(nRows * contamination);
    }
    if (thresholdMethod === "quantile") {
      return Math.round(nRows * (1 - quantileVal));
    }
    return null;
  }, [contamination, nRows, quantileVal, thresholdMethod]);

  const thresholdSetting =
    thresholdMethod === "contamination"
      ? `${(contamination * 100).toFixed(1)}% expected`
      : thresholdMethod === "quantile"
        ? `${(quantileVal * 100).toFixed(0)}th percentile`
        : `score ≥ ${fixedVal.toFixed(2)}`;

  const parameterSummary =
    selectedModel === "Isolation Forest"
      ? `${nEstimators} trees`
      : selectedModel === "Local Outlier Factor"
        ? `k = ${nNeighbors}`
        : selectedModel === "One-Class SVM"
          ? `${svmKernel.toUpperCase()} · ν ${svmNu.toFixed(2)}`
          : "robust covariance";

  const ensembleWeightsTotal = Object.values(ensembleWeights).reduce(
    (sum, weight) => sum + weight,
    0
  );

  const handleRunDetection = async () => {
    if (!matrixReady) {
      setError(
        !dataset?.loaded
          ? "Load a dataset before running detection."
          : "Build the feature matrix before running detection."
      );
      return;
    }
    if (isRunning) return;

    setIsRunning(true);
    setError(null);
    setRunCompleted(false);

    try {
      let params: Record<string, unknown> = {};
      if (selectedModel === "Isolation Forest") {
        params = { n_estimators: nEstimators, contamination };
      } else if (selectedModel === "Local Outlier Factor") {
        params = { n_neighbors: nNeighbors, contamination };
      } else if (selectedModel === "One-Class SVM") {
        params = { nu: svmNu, kernel: svmKernel };
      } else if (selectedModel === "Elliptic Envelope") {
        params = { contamination };
      }

      let result;
      if (useEnsemble) {
        const ensembleRes = await api.runEnsemble({
          models: selectedEnsembleModels.length > 0 ? selectedEnsembleModels : ["Isolation Forest", "k-Nearest Neighbors"],
          aggregation_method: ensembleMethod,
        });
        result = { success: true, overview: ensembleRes.overview };
      } else {
        result = await api.runDetection({
          model_name: selectedModel,
          params,
          threshold_method: thresholdMethod,
          contamination,
          quantile_val: quantileVal,
          fixed_val: fixedVal,
        });
      }

      await refreshAll();
      setRunCompleted(true);

      const flagged = result.overview?.n_anomalies ?? 0;
      toast(
        "success",
        "Detection complete",
        `${flagged.toLocaleString()} ${
          flagged === 1 ? "record" : "records"
        } flagged by ${selectedModel}.`
      );
    } catch (err) {
      const message = errorMessage(err);
      setError(message);
      toast("error", "Detection failed", message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Prerequisite notice */}
      {!matrixReady && (
        <div
          role="status"
          className="flex flex-col gap-3 rounded-xl border border-caution-line bg-caution-soft p-4 sm:flex-row sm:items-center sm:justify-between"
        >
          <div className="flex items-start gap-3">
            <Database
              size={16}
              className="mt-0.5 shrink-0 text-caution"
              aria-hidden="true"
            />
            <div>
              <p className="text-xs font-bold text-foreground">
                {dataset?.loaded
                  ? "The feature matrix is not ready"
                  : "A dataset is required before detection"}
              </p>
              <p className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">
                {dataset?.loaded
                  ? "Run preprocessing on the dataset screen to build the matrix that every detector scores."
                  : "Upload a CSV or load the benchmark sample, then build its feature matrix."}
              </p>
            </div>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setActiveTab("dataset")}
            className="shrink-0"
          >
            <Database size={13} aria-hidden="true" />
            {dataset?.loaded ? "Prepare features" : "Open dataset"}
          </Button>
        </div>
      )}

      {/* Detector selection */}
      <section className="space-y-3" aria-labelledby="detector-heading">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <SectionLabel>Step 1 · Detector</SectionLabel>
            <h2
              id="detector-heading"
              className="mt-1 text-sm font-bold tracking-tight text-foreground"
            >
              Choose an unsupervised algorithm
            </h2>
          </div>
          <span className="text-[10px] text-subtle-foreground">
            4 methods · no labels required
          </span>
        </div>

        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {MODELS.map((item) => {
            const Icon = item.icon;
            const selected = selectedModel === item.id;
            return (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => setSelectedModel(item.id)}
                  aria-pressed={selected}
                  disabled={isRunning}
                  className={cn(
                    "panel group relative flex h-full w-full flex-col justify-between overflow-hidden p-4 text-left transition-[border-color,box-shadow,transform] duration-200",
                    selected
                      ? "border-primary-line shadow-[0_0_0_1px_var(--primary)]"
                      : "hover:-translate-y-0.5 hover:border-border-strong",
                    isRunning && "cursor-not-allowed opacity-60"
                  )}
                >
                  <span
                    className={cn(
                      "absolute inset-x-0 top-0 h-0.5 transition-opacity",
                      selected ? "opacity-100" : "opacity-0"
                    )}
                    style={{ backgroundColor: item.accent }}
                    aria-hidden="true"
                  />
                  <span className="block">
                    <span className="mb-3 flex items-start justify-between gap-2">
                      <span
                        className="grid size-8 place-items-center rounded-lg text-primary-foreground"
                        style={{ backgroundColor: item.accent }}
                      >
                        <Icon size={16} aria-hidden="true" />
                      </span>
                      <span className="flex items-center gap-2">
                        <span className="rounded bg-surface-muted px-1.5 py-0.5 font-mono text-[10px] font-bold text-subtle-foreground">
                          {item.complexity}
                        </span>
                        <span
                          className={cn(
                            "grid size-5 place-items-center rounded-full border transition-colors",
                            selected
                              ? "border-primary bg-primary text-primary-foreground"
                              : "border-border text-transparent"
                          )}
                          aria-hidden="true"
                        >
                          <CheckCircle2 size={12} />
                        </span>
                      </span>
                    </span>
                    <span className="block text-sm font-bold text-foreground">
                      {item.id}
                    </span>
                    <span className="mt-0.5 block text-[11px] font-semibold text-primary">
                      {item.tagline}
                    </span>
                    <span className="mt-2 block text-[11px] leading-relaxed text-muted-foreground">
                      {item.desc}
                    </span>
                  </span>
                  <span className="mt-4 block border-t border-border pt-3 text-[10px] leading-relaxed text-subtle-foreground">
                    <span className="font-semibold text-muted-foreground">
                      Best for:
                    </span>{" "}
                    {item.bestFor}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="space-y-4">
          {/* Steps 2 and 3 */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Card className="p-5">
              <div className="flex items-start gap-3">
                <span className="grid size-6 shrink-0 place-items-center rounded-md border border-border bg-surface-muted font-mono text-[10px] font-bold text-subtle-foreground">
                  2
                </span>
                <div>
                  <h3 className="flex items-center gap-1.5 text-xs font-bold text-foreground">
                    <Sliders size={13} aria-hidden="true" />
                    Algorithm parameters
                  </h3>
                  <p className="mt-0.5 text-[10px] leading-relaxed text-subtle-foreground">
                    Specific to {model.short} · {parameterSummary}
                  </p>
                </div>
              </div>

              <div className="mt-4 space-y-4">
                {selectedModel === "Isolation Forest" && (
                  <RangeField
                    id="n-estimators"
                    label="n_estimators (trees)"
                    value={nEstimators}
                    display={`${nEstimators}`}
                    min={50}
                    max={300}
                    step={25}
                    onChange={setNEstimators}
                    hint="Number of decision trees grown to isolate observations."
                    disabled={isRunning}
                  />
                )}

                {selectedModel === "Local Outlier Factor" && (
                  <RangeField
                    id="n-neighbors"
                    label="n_neighbors (k)"
                    value={nNeighbors}
                    display={`${nNeighbors}`}
                    min={5}
                    max={50}
                    step={5}
                    onChange={setNNeighbors}
                    hint="Neighbourhood radius used to compare relative point densities."
                    disabled={isRunning}
                  />
                )}

                {selectedModel === "One-Class SVM" && (
                  <div className="space-y-4">
                    <div>
                      <label
                        htmlFor="svm-kernel"
                        className="mb-1.5 block text-xs font-semibold text-foreground"
                      >
                        Kernel function
                      </label>
                      <select
                        id="svm-kernel"
                        value={svmKernel}
                        onChange={(e) => setSvmKernel(e.target.value)}
                        className="field"
                        disabled={isRunning}
                      >
                        <option value="rbf">
                          RBF (radial basis — non-linear)
                        </option>
                        <option value="linear">Linear</option>
                        <option value="poly">Polynomial</option>
                      </select>
                    </div>
                    <RangeField
                      id="svm-nu"
                      label="nu (upper bound on margin error)"
                      value={svmNu}
                      display={svmNu.toFixed(2)}
                      min={0.01}
                      max={0.25}
                      step={0.01}
                      onChange={setSvmNu}
                      disabled={isRunning}
                    />
                  </div>
                )}

                {selectedModel === "Elliptic Envelope" && (
                  <div className="panel-inset p-3">
                    <p className="text-[11px] leading-relaxed text-muted-foreground">
                      Fits a robust minimum-covariance-determinant estimate and
                      scores observations by Mahalanobis distance. Unsupervised;
                      it uses the configured contamination rate shown in the
                      threshold card.
                    </p>
                  </div>
                )}
              </div>
            </Card>

            <Card className="p-5">
              <div className="flex items-start gap-3">
                <span className="grid size-6 shrink-0 place-items-center rounded-md border border-border bg-surface-muted font-mono text-[10px] font-bold text-subtle-foreground">
                  3
                </span>
                <div>
                  <h3 className="text-xs font-bold text-foreground">
                    Decision threshold
                  </h3>
                  <p className="mt-0.5 text-[10px] leading-relaxed text-subtle-foreground">
                    How scores become flagged records
                  </p>
                </div>
              </div>

              <div className="mt-4 space-y-4">
                <div>
                  <label
                    htmlFor="threshold-method"
                    className="mb-1.5 block text-xs font-semibold text-foreground"
                  >
                    Threshold method
                  </label>
                  <select
                    id="threshold-method"
                    value={thresholdMethod}
                    onChange={(e) => setThresholdMethod(e.target.value)}
                    className="field"
                    disabled={isRunning}
                  >
                    <option value="contamination">
                      Contamination rate (top expected %)
                    </option>
                    <option value="quantile">
                      Quantile cutoff (percentile threshold)
                    </option>
                    <option value="fixed">
                      Fixed cutoff (exact score cutoff)
                    </option>
                  </select>
                  <p className="mt-1 text-[10px] leading-relaxed text-subtle-foreground">
                    {threshold?.hint}
                  </p>
                </div>

                {thresholdMethod === "contamination" && (
                  <RangeField
                    id="contamination"
                    label="Expected contamination"
                    value={contamination}
                    display={`${(contamination * 100).toFixed(1)}%`}
                    min={0.01}
                    max={0.25}
                    step={0.01}
                    onChange={setContamination}
                    hint={
                      targetFlags !== null
                        ? `Targets about ${targetFlags.toLocaleString()} of ${nRows.toLocaleString()} records.`
                        : "Determines the fraction of observations flagged as outliers."
                    }
                    disabled={isRunning}
                  />
                )}

                {thresholdMethod === "quantile" && (
                  <RangeField
                    id="quantile"
                    label="Percentile threshold"
                    value={quantileVal}
                    display={`${(quantileVal * 100).toFixed(1)}th`}
                    min={0.8}
                    max={0.99}
                    step={0.01}
                    onChange={setQuantileVal}
                    hint={
                      targetFlags !== null
                        ? `Flags the top ${((1 - quantileVal) * 100).toFixed(
                            1
                          )}% — about ${targetFlags.toLocaleString()} records.`
                        : "Scores above this percentile are flagged."
                    }
                    disabled={isRunning}
                  />
                )}

                {thresholdMethod === "fixed" && (
                  <RangeField
                    id="fixed"
                    label="Fixed score cutoff"
                    value={fixedVal}
                    display={fixedVal.toFixed(2)}
                    min={0.1}
                    max={0.95}
                    step={0.05}
                    onChange={setFixedVal}
                    hint="Only records at or above this normalised score are flagged."
                    disabled={isRunning}
                  />
                )}

                <div className="rounded-lg border border-border bg-surface-muted px-3 py-2">
                  <div className="flex items-center justify-between gap-2 text-[10px]">
                    <span className="text-subtle-foreground">Configured cutoff</span>
                    <span className="font-mono font-bold text-foreground">
                      {thresholdSetting}
                    </span>
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* Ensemble */}
          <Card className="p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="flex items-start gap-3">
                <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-border bg-surface-muted text-primary">
                  <Layers size={15} aria-hidden="true" />
                </span>
                <div>
                  <h3 className="text-xs font-bold text-foreground">
                    Ensemble multi-model scoring
                  </h3>
                  <p className="mt-0.5 max-w-xl text-[11px] leading-relaxed text-muted-foreground">
                    Blends the selected detector with Isolation Forest, LOF and
                    One-Class SVM. Scores are normalised before weighting — useful
                    when no single method clearly fits the data.
                  </p>
                </div>
              </div>

              <label className="flex cursor-pointer select-none items-center gap-2.5">
                <span className="text-[10px] font-semibold text-muted-foreground">
                  {useEnsemble ? "Enabled" : "Disabled"}
                </span>
                <input
                  type="checkbox"
                  checked={useEnsemble}
                  onChange={(e) => setUseEnsemble(e.target.checked)}
                  disabled={isRunning}
                  className="peer sr-only"
                />
                <span
                  className={cn(
                    "relative h-5 w-9 rounded-full border border-border bg-surface-muted transition-colors",
                    "peer-checked:border-primary peer-checked:bg-primary",
                    "peer-focus-visible:ring-2 peer-focus-visible:ring-primary/40 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-surface",
                    "peer-disabled:cursor-not-allowed peer-disabled:opacity-50"
                  )}
                  aria-hidden="true"
                >
                  <span
                    className={cn(
                      "absolute left-0.5 top-0.5 size-3.5 rounded-full bg-surface shadow-sm transition-transform",
                      "peer-checked:translate-x-4"
                    )}
                  />
                </span>
                <span className="sr-only">Enable ensemble scoring</span>
              </label>
            </div>

            {useEnsemble && (
              <div className="mt-4 border-t border-border pt-4">
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  {(
                    Object.keys(
                      ensembleWeights
                    ) as (keyof typeof ensembleWeights)[]
                  ).map((key) => (
                    <RangeField
                      key={key}
                      id={`weight-${key.toLowerCase().replace(/[^a-z]+/g, "-")}`}
                      label={key === "Local Outlier Factor" ? "LOF" : key}
                      value={ensembleWeights[key]}
                      display={ensembleWeights[key].toFixed(1)}
                      min={0}
                      max={1}
                      step={0.1}
                      onChange={(v) =>
                        setEnsembleWeights((prev) => ({ ...prev, [key]: v }))
                      }
                      disabled={isRunning}
                    />
                  ))}
                </div>
                <p className="mt-3 text-[10px] text-subtle-foreground">
                  Relative weights sum to{" "}
                  <span className="font-mono font-semibold tabular text-muted-foreground">
                    {ensembleWeightsTotal.toFixed(1)}
                  </span>
                  . The backend renormalises them to 1.0; an all-zero mix falls
                  back to equal weights.
                </p>
              </div>
            )}
          </Card>
        </div>

        {/* Run rail */}
        <aside className="space-y-4 xl:sticky xl:top-24">
          <Card className="p-5" aria-busy={isRunning}>
            <div className="flex items-center justify-between gap-2">
              <SectionLabel>Run plan</SectionLabel>
              <span className="inline-flex items-center gap-1 rounded-full border border-border bg-surface-muted px-2 py-0.5 text-[9px] font-semibold text-subtle-foreground">
                <Cpu size={9} aria-hidden="true" />
                Unsupervised
              </span>
            </div>

            <dl className="mt-4 space-y-2.5 text-[11px]">
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-subtle-foreground">Dataset</dt>
                <dd
                  className="max-w-[11rem] truncate text-right font-semibold text-foreground"
                  title={dataset?.dataset_name || "No dataset loaded"}
                >
                  {dataset?.dataset_name || "None"}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-subtle-foreground">Records</dt>
                <dd className="font-semibold tabular text-foreground">
                  {nRows ? nRows.toLocaleString() : "—"}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-subtle-foreground">Features</dt>
                <dd className="font-semibold tabular text-foreground">
                  {matrixReady ? nFeatures.toLocaleString() : "—"}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-subtle-foreground">Detector</dt>
                <dd className="max-w-[11rem] truncate text-right font-semibold text-foreground">
                  {model.id}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-subtle-foreground">Parameters</dt>
                <dd className="text-right font-semibold text-foreground">
                  {parameterSummary}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-subtle-foreground">Threshold</dt>
                <dd className="text-right font-semibold text-foreground">
                  {thresholdSetting}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-subtle-foreground">Target flagged</dt>
                <dd className="font-semibold tabular text-foreground">
                  {targetFlags !== null
                    ? `≈ ${targetFlags.toLocaleString()}`
                    : "score-based"}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4 border-t border-border pt-2.5">
                <dt className="text-subtle-foreground">Ensemble</dt>
                <dd
                  className={cn(
                    "font-semibold",
                    useEnsemble ? "text-primary" : "text-muted-foreground"
                  )}
                >
                  {useEnsemble ? "Weighted blend" : "Off"}
                </dd>
              </div>
            </dl>

            <Button
              size="lg"
              onClick={handleRunDetection}
              disabled={!canRun}
              className="mt-4 w-full"
            >
              {isRunning ? <Spinner /> : <Play size={15} aria-hidden="true" />}
              {isRunning ? "Fitting and scoring…" : "Run detection"}
            </Button>

            {isRunning && (
              <div
                className="mt-3 space-y-2"
                role="status"
                aria-live="polite"
              >
                <div className="h-1 overflow-hidden rounded-full bg-surface-muted">
                  <span className="block h-full w-full animate-pulse rounded-full bg-primary/70" />
                </div>
                <p className="text-[10px] leading-relaxed text-subtle-foreground">
                  Fitting {model.short} and scoring{" "}
                  {nRows ? nRows.toLocaleString() : "the"} records. The backend
                  responds when scoring and thresholding finish.
                </p>
              </div>
            )}

            {!matrixReady && !isRunning && (
              <p className="mt-3 text-[10px] leading-relaxed text-caution">
                {dataset?.loaded
                  ? "Prepare the feature matrix on the dataset screen to enable this run."
                  : "Load a dataset to enable this run."}
              </p>
            )}

            {error && (
              <div
                role="alert"
                className="mt-3 flex items-start gap-2 rounded-lg border border-danger-line bg-danger-soft px-3 py-2.5"
              >
                <AlertTriangle
                  size={13}
                  className="mt-0.5 shrink-0 text-danger"
                  aria-hidden="true"
                />
                <div className="min-w-0">
                  <p className="text-[11px] font-bold text-foreground">
                    Detection failed
                  </p>
                  <p className="mt-0.5 break-words text-[10px] leading-relaxed text-muted-foreground">
                    {error}
                  </p>
                </div>
              </div>
            )}
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-2">
              <Gauge size={13} className="text-subtle-foreground" aria-hidden="true" />
              <SectionLabel>What the cutoff means</SectionLabel>
            </div>
            <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground">
              {thresholdMethod === "contamination" &&
                "Scores are ranked and the top share is flagged. The severity tier is assigned within that flagged set, so every tier stays meaningful."}
              {thresholdMethod === "quantile" &&
                "The empirical score percentile is used as the cutoff. Scores are normalised first, so the percentile reflects this run, not another dataset."}
              {thresholdMethod === "fixed" &&
                "An absolute normalised cutoff is applied. Use it when a specific operating point is required and you can accept the resulting flag count."}
            </p>
          </Card>
        </aside>
      </div>

      {/* Latest run */}
      {overview?.ready && !isRunning && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="panel flex flex-col gap-4 border-success-line bg-success-soft p-4 sm:flex-row sm:items-center sm:justify-between"
        >
          <div className="flex items-start gap-3">
            <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-success-line bg-surface text-success">
              <CheckCircle2 size={16} aria-hidden="true" />
            </span>
            <div>
              <p className="text-xs font-bold text-foreground">
                {runCompleted ? "Detection complete" : "Latest detection run"}:{" "}
                {(overview.n_anomalies ?? 0).toLocaleString()} anomalies
                identified
                {overview.anomaly_rate !== undefined
                  ? ` (${(overview.anomaly_rate * 100).toFixed(2)}%)`
                  : ""}
              </p>
              <p className="mt-0.5 text-[11px] text-muted-foreground">
                {overview.model_name}
                {overview.threshold_val !== undefined
                  ? ` · threshold ${overview.threshold_val.toFixed(4)}`
                  : ""}
                {overview.threshold_method
                  ? ` · ${overview.threshold_method}`
                  : ""}
              </p>
            </div>
          </div>

          <div className="flex shrink-0 flex-wrap items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setActiveTab("dashboard")}
            >
              View dashboard
            </Button>
            <Button size="sm" onClick={() => setActiveTab("anomalies")}>
              Explore anomalies
              <ArrowRight size={13} aria-hidden="true" />
            </Button>
          </div>
        </motion.div>
      )}

      {/* Quick path for first-time users */}
      {!dataset?.loaded && (
        <div className="flex items-center gap-2 text-[10px] text-subtle-foreground">
          <Sparkles size={11} aria-hidden="true" />
          New here? The benchmark sample loads 880 records and a ready feature
          matrix in one step from the dataset screen.
        </div>
      )}
    </div>
  );
};

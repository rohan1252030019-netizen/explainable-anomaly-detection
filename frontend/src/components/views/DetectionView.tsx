"use client";

import React, { useState } from "react";
import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import {
  Play,
  Trees,
  Target,
  Zap,
  CircleDot,
  CheckCircle2,
  Layers,
  ArrowRight,
  Sliders,
} from "lucide-react";

export const DetectionView: React.FC = () => {
  const { dataset, overview, refreshAll, setActiveTab } = useApp();

  const [selectedModel, setSelectedModel] = useState<string>("Isolation Forest");
  const [thresholdMethod, setThresholdMethod] = useState<string>("contamination");
  const [contamination, setContamination] = useState<number>(0.05);
  const [quantileVal, setQuantileVal] = useState<number>(0.95);
  const [fixedVal, setFixedVal] = useState<number>(0.5);

  // Model-specific params
  const [nEstimators, setNEstimators] = useState<number>(100);
  const [nNeighbors, setNNeighbors] = useState<number>(20);
  const [svmNu, setSvmNu] = useState<number>(0.05);
  const [svmKernel, setSvmKernel] = useState<string>("rbf");

  // Ensemble toggle
  const [useEnsemble, setUseEnsemble] = useState<boolean>(false);
  const [ensembleWeights, setEnsembleWeights] = useState({
    "Isolation Forest": 0.4,
    "Local Outlier Factor": 0.3,
    "One-Class SVM": 0.3,
  });

  // Running state
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(0);

  const modelsList = [
    {
      id: "Isolation Forest",
      name: "Isolation Forest",
      short: "IF",
      icon: Trees,
      tagline: "Recursive random partitioning. Fast and scalable.",
      desc: "Isolates anomalies by recursive random splits. Anomalies require fewer partitions to isolate.",
      bestFor: "Global outliers, large datasets",
      complexity: "O(n log n)",
      color: "#6366F1",
    },
    {
      id: "Local Outlier Factor",
      name: "Local Outlier Factor",
      short: "LOF",
      icon: Target,
      tagline: "Density-based. Detects sparse local regions.",
      desc: "Compares local density of an observation to its k nearest neighbours. Low-density regions are flagged.",
      bestFor: "Local outliers, clusters with varying densities",
      complexity: "O(n²)",
      color: "#3B82F6",
    },
    {
      id: "One-Class SVM",
      name: "One-Class SVM",
      short: "OCSVM",
      icon: Zap,
      tagline: "Kernel-based max-margin boundary.",
      desc: "Learns a kernel-based hypersphere surrounding the dense normal training points in feature space.",
      bestFor: "Non-linear distributions, novelty boundary",
      complexity: "O(n²–n³)",
      color: "#F59E0B",
    },
    {
      id: "Elliptic Envelope",
      name: "Elliptic Envelope",
      short: "EE",
      icon: CircleDot,
      tagline: "Multivariate Gaussian with Mahalanobis distance.",
      desc: "Fits a robust covariance estimate to detect observations with high Mahalanobis distances.",
      bestFor: "Unimodal, approximately Gaussian features",
      complexity: "O(n · p²)",
      color: "#10B981",
    },
  ];

  const handleRunDetection = async () => {
    if (!dataset?.loaded) {
      alert("Please load a dataset first.");
      return;
    }

    setIsRunning(true);
    setCurrentStep(1);

    try {
      // Step simulation for visual feedback
      await new Promise((r) => setTimeout(r, 200));
      setCurrentStep(2);

      let params: Record<string, any> = {};
      if (selectedModel === "Isolation Forest") {
        params = { n_estimators: nEstimators, contamination };
      } else if (selectedModel === "Local Outlier Factor") {
        params = { n_neighbors: nNeighbors, contamination };
      } else if (selectedModel === "One-Class SVM") {
        params = { nu: svmNu, kernel: svmKernel };
      } else if (selectedModel === "Elliptic Envelope") {
        params = { contamination };
      }

      setCurrentStep(3);

      await api.runDetection({
        model_name: selectedModel,
        params,
        threshold_method: thresholdMethod,
        contamination,
        quantile_val: quantileVal,
        fixed_val: fixedVal,
        use_ensemble: useEnsemble,
        ensemble_weights: useEnsemble ? ensembleWeights : undefined,
      });

      setCurrentStep(4);
      await new Promise((r) => setTimeout(r, 200));
      setCurrentStep(5);

      await refreshAll();
    } catch (err: any) {
      alert(`Detection failed: ${err.message}`);
    } finally {
      setIsRunning(false);
      setCurrentStep(0);
    }
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Visual Pipeline Banner */}
      <div className="saas-card p-5 bg-gradient-to-r from-slate-900 to-indigo-950/30 border-indigo-500/20">
        <div className="text-[11px] font-bold uppercase tracking-wider text-indigo-400 mb-3">
          End-to-End Detection Pipeline
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-200 border border-slate-700">
            1. Preprocessed X ({dataset?.n_preprocessed_features || 15} features)
          </span>
          <span className="text-slate-500">→</span>
          <span className="px-2.5 py-1 rounded bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 font-semibold">
            2. {selectedModel}
          </span>
          <span className="text-slate-500">→</span>
          <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-200 border border-slate-700">
            3. Score Normalization
          </span>
          <span className="text-slate-500">→</span>
          <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-200 border border-slate-700">
            4. {thresholdMethod} ({(contamination * 100).toFixed(0)}%)
          </span>
          <span className="text-slate-500">→</span>
          <span className="px-2.5 py-1 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30 font-semibold">
            5. Explainability & Ranking
          </span>
        </div>
      </div>

      {/* Model Selection Cards */}
      <div className="space-y-3">
        <div className="text-xs font-bold uppercase tracking-wider text-slate-400">
          Step 1 · Select Unsupervised Algorithm
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {modelsList.map((m) => {
            const Icon = m.icon;
            const isSelected = selectedModel === m.id;
            return (
              <div
                key={m.id}
                onClick={() => setSelectedModel(m.id)}
                className={`saas-card p-4 flex flex-col justify-between cursor-pointer transition-all ${
                  isSelected
                    ? "ring-2 ring-indigo-500 bg-indigo-500/5 dark:bg-indigo-500/10 border-transparent shadow-md"
                    : "hover:border-slate-400 dark:hover:border-slate-700"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center text-white"
                      style={{ backgroundColor: m.color }}
                    >
                      <Icon size={16} />
                    </div>
                    <span className="text-[10px] font-mono font-bold text-slate-400 px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800">
                      {m.complexity}
                    </span>
                  </div>

                  <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                    {m.name}
                  </h4>
                  <p className="text-[11px] text-indigo-400 font-medium mt-0.5">
                    {m.tagline}
                  </p>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-2 leading-relaxed">
                    {m.desc}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 text-[10px] text-slate-400">
                  <span className="font-semibold text-slate-500">Best for:</span> {m.bestFor}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Model Hyperparameters + Thresholding Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Parameters Column */}
        <div className="saas-card p-5 space-y-4">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Sliders size={14} />
            <span>Step 2 · Algorithm Parameters ({selectedModel})</span>
          </div>

          {selectedModel === "Isolation Forest" && (
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-700 dark:text-slate-300 font-semibold">n_estimators (Trees)</span>
                  <span className="font-mono text-indigo-400">{nEstimators}</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="300"
                  step="25"
                  value={nEstimators}
                  onChange={(e) => setNEstimators(Number(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400">
                  Number of decision trees grown to isolate observations.
                </span>
              </div>
            </div>
          )}

          {selectedModel === "Local Outlier Factor" && (
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-700 dark:text-slate-300 font-semibold">n_neighbors (k)</span>
                  <span className="font-mono text-indigo-400">{nNeighbors}</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="50"
                  step="5"
                  value={nNeighbors}
                  onChange={(e) => setNNeighbors(Number(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer"
                />
                <span className="text-[10px] text-slate-400">
                  Local neighborhood radius used to calculate relative point densities.
                </span>
              </div>
            </div>
          )}

          {selectedModel === "One-Class SVM" && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                  Kernel Function
                </label>
                <select
                  value={svmKernel}
                  onChange={(e) => setSvmKernel(e.target.value)}
                  className="saas-input w-full p-2 text-xs"
                >
                  <option value="rbf">RBF (Radial Basis Function - Non-linear)</option>
                  <option value="linear">Linear</option>
                  <option value="poly">Polynomial</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-700 dark:text-slate-300 font-semibold">nu (Margin Error Upper Bound)</span>
                  <span className="font-mono text-indigo-400">{svmNu}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.25"
                  step="0.01"
                  value={svmNu}
                  onChange={(e) => setSvmNu(Number(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer"
                />
              </div>
            </div>
          )}

          {selectedModel === "Elliptic Envelope" && (
            <div className="text-xs text-slate-400">
              Fits a robust minimum covariance determinant to evaluate Mahalanobis distance. Unsupervised; uses estimated contamination rate.
            </div>
          )}
        </div>

        {/* Thresholding Strategy Column */}
        <div className="saas-card p-5 space-y-4">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Step 3 · Thresholding Strategy
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
              Threshold Decision Method
            </label>
            <select
              value={thresholdMethod}
              onChange={(e) => setThresholdMethod(e.target.value)}
              className="saas-input w-full p-2 text-xs"
            >
              <option value="contamination">Contamination Rate (Top expected % outliers)</option>
              <option value="quantile">Quantile Cutoff (Percentile threshold)</option>
              <option value="fixed">Fixed Cutoff (Exact score cutoff)</option>
            </select>
          </div>

          {thresholdMethod === "contamination" && (
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-700 dark:text-slate-300 font-semibold">Expected Contamination</span>
                <span className="font-mono text-indigo-400">{(contamination * 100).toFixed(1)}%</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.25"
                step="0.01"
                value={contamination}
                onChange={(e) => setContamination(Number(e.target.value))}
                className="w-full accent-indigo-600 cursor-pointer"
              />
              <span className="text-[10px] text-slate-400">
                Determines the fraction of observations flagged as outliers.
              </span>
            </div>
          )}

          {thresholdMethod === "quantile" && (
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-700 dark:text-slate-300 font-semibold">Percentile Threshold</span>
                <span className="font-mono text-indigo-400">{(quantileVal * 100).toFixed(1)}th</span>
              </div>
              <input
                type="range"
                min="0.80"
                max="0.99"
                step="0.01"
                value={quantileVal}
                onChange={(e) => setQuantileVal(Number(e.target.value))}
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>
          )}

          {thresholdMethod === "fixed" && (
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-700 dark:text-slate-300 font-semibold">Fixed Score Cutoff</span>
                <span className="font-mono text-indigo-400">{fixedVal.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.95"
                step="0.05"
                value={fixedVal}
                onChange={(e) => setFixedVal(Number(e.target.value))}
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>
          )}
        </div>
      </div>

      {/* Ensemble Scoring Option */}
      <div className="saas-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers size={16} className="text-indigo-400" />
            <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
              Ensemble Multi-Model Scoring
            </span>
          </div>
          <input
            type="checkbox"
            checked={useEnsemble}
            onChange={(e) => setUseEnsemble(e.target.checked)}
            className="w-4 h-4 accent-indigo-600 rounded cursor-pointer"
          />
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          Combines normalized outlier scores from Isolation Forest, LOF, and One-Class SVM via weighted average to produce more robust detections.
        </p>

        {useEnsemble && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 border-t border-slate-800">
            <div>
              <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                <span>Isolation Forest Weight</span>
                <span>{ensembleWeights["Isolation Forest"]}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={ensembleWeights["Isolation Forest"]}
                onChange={(e) =>
                  setEnsembleWeights({
                    ...ensembleWeights,
                    "Isolation Forest": Number(e.target.value),
                  })
                }
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>
            <div>
              <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                <span>LOF Weight</span>
                <span>{ensembleWeights["Local Outlier Factor"]}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={ensembleWeights["Local Outlier Factor"]}
                onChange={(e) =>
                  setEnsembleWeights({
                    ...ensembleWeights,
                    "Local Outlier Factor": Number(e.target.value),
                  })
                }
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>
            <div>
              <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                <span>One-Class SVM Weight</span>
                <span>{ensembleWeights["One-Class SVM"]}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={ensembleWeights["One-Class SVM"]}
                onChange={(e) =>
                  setEnsembleWeights({
                    ...ensembleWeights,
                    "One-Class SVM": Number(e.target.value),
                  })
                }
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>
          </div>
        )}
      </div>

      {/* Progress Execution Feedback */}
      {isRunning && (
        <div className="saas-card p-5 space-y-3 bg-indigo-950/20 border-indigo-500/30">
          <div className="text-xs font-bold text-indigo-400 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
            <span>Running Detection Pipeline...</span>
          </div>
          <div className="space-y-2 text-xs text-slate-300">
            <div className={`flex items-center gap-2 ${currentStep >= 1 ? "text-emerald-400 font-semibold" : "text-slate-500"}`}>
              <CheckCircle2 size={14} /> 1. Preparing feature matrix and reference distributions
            </div>
            <div className={`flex items-center gap-2 ${currentStep >= 2 ? "text-emerald-400 font-semibold" : "text-slate-500"}`}>
              <CheckCircle2 size={14} /> 2. Fitting unsupervised detector ({selectedModel})
            </div>
            <div className={`flex items-center gap-2 ${currentStep >= 3 ? "text-emerald-400 font-semibold" : "text-slate-500"}`}>
              <CheckCircle2 size={14} /> 3. Computing continuous anomaly scores
            </div>
            <div className={`flex items-center gap-2 ${currentStep >= 4 ? "text-emerald-400 font-semibold" : "text-slate-500"}`}>
              <CheckCircle2 size={14} /> 4. Calibrating threshold & assigning severity tiers
            </div>
            <div className={`flex items-center gap-2 ${currentStep >= 5 ? "text-emerald-400 font-semibold" : "text-slate-500"}`}>
              <CheckCircle2 size={14} /> 5. Building ranked results table
            </div>
          </div>
        </div>
      )}

      {/* Run Action Button */}
      <div className="flex items-center justify-between pt-2">
        <div className="text-xs text-slate-400">
          Dataset: <span className="font-semibold text-slate-200">{dataset?.dataset_name || "None"}</span> · Features: <span className="font-semibold text-slate-200">{dataset?.n_preprocessed_features || 0}</span>
        </div>

        <button
          onClick={handleRunDetection}
          disabled={isRunning || !dataset?.loaded}
          className="flex items-center gap-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-bold rounded-xl shadow-lg shadow-indigo-600/20 transition-all cursor-pointer"
        >
          <Play size={15} />
          <span>{isRunning ? "Training & Scoring..." : "Execute Anomaly Detection"}</span>
        </button>
      </div>

      {/* Post-run prompt */}
      {overview?.ready && !isRunning && (
        <div className="saas-card p-4 flex items-center justify-between border-emerald-500/30 bg-emerald-500/5">
          <div className="flex items-center gap-3">
            <CheckCircle2 size={18} className="text-emerald-500" />
            <div>
              <div className="text-xs font-bold text-slate-900 dark:text-slate-100">
                Detection Complete: {overview.n_anomalies} anomalies identified ({( (overview.anomaly_rate || 0) * 100).toFixed(2)}%)
              </div>
              <div className="text-[11px] text-slate-400">
                Threshold: {overview.threshold_val?.toFixed(4)} · Model: {overview.model_name}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab("anomalies")}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 text-white text-xs font-semibold rounded-lg shadow-sm cursor-pointer"
            >
              <span>Explore Anomalies</span>
              <ArrowRight size={13} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

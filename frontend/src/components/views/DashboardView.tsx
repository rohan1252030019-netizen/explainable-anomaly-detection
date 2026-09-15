"use client";

import React from "react";
import { useApp } from "@/context/AppContext";
import { StatCard } from "@/components/ui/StatCard";
import { ScoreDistributionChart } from "@/components/charts/ScoreDistributionChart";
import { SeverityDonutChart } from "@/components/charts/SeverityDonutChart";
import { ObservationTimelineChart } from "@/components/charts/ObservationTimelineChart";
import { SeverityBadge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import {
  ShieldAlert,
  Database,
  BarChart3,
  Percent,
  Play,
  ArrowRight,
  TrendingUp,
  Sparkles,
} from "lucide-react";
import { api } from "@/lib/api";

export const DashboardView: React.FC = () => {
  const { dataset, overview, setActiveTab, refreshAll } = useApp();

  const handleRunDemo = async () => {
    try {
      await api.loadSample();
      await api.runDetection({
        model_name: "Isolation Forest",
        contamination: 0.05,
      });
      await refreshAll();
    } catch (e) {
      console.error(e);
    }
  };

  // State 1: No dataset loaded
  if (!dataset?.loaded) {
    return (
      <div className="space-y-8 max-w-5xl mx-auto py-8">
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 text-xs font-semibold">
            <Sparkles size={14} />
            <span>Unsupervised Machine Learning & Transparent XAI</span>
          </div>
          <h2 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-slate-50">
            Explainable Anomaly Detection Platform
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-xl mx-auto leading-relaxed">
            Detect subtle anomalies without labeled training data, understand feature-level contributions, and investigate outlier records with transparent explainability.
          </p>
        </div>

        <EmptyState
          icon={Database}
          title="No Dataset Loaded"
          description="Upload your tabular CSV dataset or launch the benchmark sample dataset (880 records with financial & behavioral features) to begin."
          actionText="Load Benchmark Sample Dataset"
          onAction={handleRunDemo}
        />

        {/* Feature pillars */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
          <div className="saas-card p-5 space-y-2">
            <div className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
              1. Unsupervised Learning
            </div>
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              Zero Label Dependency
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              Models learn the manifold of normal behavior autonomously using Isolation Forest, LOF, One-Class SVM, and Elliptic Envelope.
            </p>
          </div>

          <div className="saas-card p-5 space-y-2">
            <div className="text-xs font-bold text-rose-400 uppercase tracking-wider">
              2. Precise Scoring
            </div>
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              Calibrated Thresholds
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              Continuous scores normalized in [0, 1] with contamination quantiles and severity tiering (Critical, High, Medium, Low).
            </p>
          </div>

          <div className="saas-card p-5 space-y-2">
            <div className="text-xs font-bold text-purple-400 uppercase tracking-wider">
              3. Transparent XAI
            </div>
            <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              Plain-English Reasons
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              Decompose anomalies into directional deviations, robust z-score variations, perturbation importance, and tree-exact SHAP values.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // State 2: Dataset loaded, detection not run
  if (!overview?.ready) {
    return (
      <div className="space-y-6 max-w-5xl mx-auto py-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">
              Dataset Ready for Analysis
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {dataset.dataset_name} ({dataset.n_rows?.toLocaleString()} rows × {dataset.n_cols} columns)
            </p>
          </div>
          <button
            onClick={() => setActiveTab("detection")}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-sm transition-all cursor-pointer"
          >
            <Play size={14} />
            <span>Configure & Run Detection</span>
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <StatCard label="Total Records" value={dataset.n_rows || 0} icon={Database} />
          <StatCard label="Total Columns" value={dataset.n_cols || 0} icon={BarChart3} />
          <StatCard label="Preprocessed Features" value={dataset.n_preprocessed_features || 0} icon={Percent} />
          <StatCard label="Missing Values" value={dataset.total_missing || 0} icon={ShieldAlert} />
        </div>

        <EmptyState
          icon={Play}
          title="Run Anomaly Detection"
          description="Select an unsupervised algorithm (Isolation Forest, LOF, OCSVM, or Elliptic Envelope) to score and rank anomalies."
          actionText="Go to Detection Workflow"
          onAction={() => setActiveTab("detection")}
        />
      </div>
    );
  }

  // State 3: Full Dashboard with real data
  const total = overview.n_total || 0;
  const anomalies = overview.n_anomalies || 0;
  const ratePct = ((overview.anomaly_rate || 0) * 100).toFixed(2);
  const critCount = overview.severity_counts?.CRITICAL || 0;
  const highCount = overview.severity_counts?.HIGH || 0;

  return (
    <div className="space-y-6">
      {/* Top KPI row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4">
        <StatCard
          label="Total Records"
          value={total.toLocaleString()}
          sub="Analyzed instances"
          icon={Database}
        />
        <StatCard
          label="Anomalies Flagged"
          value={anomalies.toLocaleString()}
          sub={`${ratePct}% anomaly rate`}
          icon={ShieldAlert}
          accentColor="#EF4444"
        />
        <StatCard
          label="Critical + High"
          value={(critCount + highCount).toLocaleString()}
          sub="Requires priority review"
          icon={TrendingUp}
          accentColor="#F97316"
        />
        <StatCard
          label="Active Model"
          value={overview.model_name || "Isolation Forest"}
          sub="Unsupervised detector"
          icon={Play}
        />
        <StatCard
          label="Median Score"
          value={overview.median_anomaly_score?.toFixed(3) || "0.000"}
          sub="Anomalous records"
          icon={BarChart3}
        />
        <StatCard
          label="Threshold Cutoff"
          value={overview.threshold_val?.toFixed(3) || "0.500"}
          sub={overview.threshold_method || "contamination"}
          icon={Percent}
        />
      </div>

      {/* Main Visuals Row: Score distribution + Severity donut */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="saas-card p-5 lg:col-span-2">
          <ScoreDistributionChart
            data={overview.score_histogram || []}
            threshold={overview.threshold_val}
            modelName={overview.model_name}
          />
        </div>

        <div className="saas-card p-5 flex flex-col justify-between">
          <div className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-2">
            Severity Classification
          </div>
          <SeverityDonutChart
            counts={
              overview.severity_counts || {
                CRITICAL: 0,
                HIGH: 0,
                MEDIUM: 0,
                LOW: 0,
              }
            }
          />
          <div className="text-[11px] text-slate-400 text-center pt-2 border-t border-slate-100 dark:border-slate-800">
            Categorized by score distribution quantiles
          </div>
        </div>
      </div>

      {/* Observation Timeline */}
      <div className="saas-card p-5">
        <ObservationTimelineChart
          data={overview.score_timeline || []}
          threshold={overview.threshold_val}
        />
      </div>

      {/* Quick Actions Footer Banner */}
      <div className="saas-card p-5 flex flex-col sm:flex-row items-center justify-between gap-4 bg-gradient-to-r from-slate-900 to-indigo-950/40 border-indigo-500/20">
        <div>
          <h4 className="text-sm font-bold text-slate-100">
            Ready to investigate flagged anomalies?
          </h4>
          <p className="text-xs text-slate-400 mt-0.5">
            Inspect individual records, view directional feature contributions, and analyze why the model made each decision.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={() => setActiveTab("anomalies")}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-sm transition-all cursor-pointer"
          >
            <span>View All Anomalies</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
};

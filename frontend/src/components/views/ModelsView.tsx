"use client";

import React, { useState } from "react";
import { useApp } from "@/context/AppContext";
import { api, ModelBenchmarkResponse } from "@/lib/api";
import { StatCard } from "@/components/ui/StatCard";
import { Layers, Play, CheckCircle2, AlertTriangle, Info } from "lucide-react";

export const ModelsView: React.FC = () => {
  const { dataset, setActiveTab } = useApp();

  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ModelBenchmarkResponse | null>(null);
  const [contamination, setContamination] = useState(0.05);

  const handleBenchmark = async () => {
    if (!dataset?.loaded) return;
    setLoading(true);
    try {
      const res = await api.benchmarkModels(contamination);
      setData(res);
    } catch (err: any) {
      alert(`Benchmark failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!dataset?.loaded) {
    return (
      <div className="saas-card p-12 text-center text-slate-400 space-y-3">
        <p className="text-sm">Load a dataset to benchmark anomaly detectors.</p>
        <button
          onClick={() => setActiveTab("dataset")}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg"
        >
          Go to Dataset Management
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Benchmark Trigger Header */}
      <div className="saas-card p-5 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="space-y-1">
          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Layers size={16} className="text-indigo-400" />
            <span>Multi-Model Benchmark & Consensus Lab</span>
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Train Isolation Forest, LOF, One-Class SVM, and Elliptic Envelope simultaneously on the exact same preprocessed data to examine consensus and divergence.
          </p>
        </div>

        <div className="flex items-center gap-4 shrink-0">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span>Contamination:</span>
            <span className="font-mono text-indigo-400 font-bold">{(contamination * 100).toFixed(0)}%</span>
          </div>

          <button
            onClick={handleBenchmark}
            disabled={loading}
            className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-bold rounded-lg shadow-sm transition-all cursor-pointer flex items-center gap-2"
          >
            <Play size={14} />
            <span>{loading ? "Training All Detectors..." : "Run Multi-Model Benchmark"}</span>
          </button>
        </div>
      </div>

      {data ? (
        <>
          {/* Consensus KPIs */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <StatCard
              label="All-Model Consensus"
              value={data.all_model_consensus}
              sub="Anomalies flagged by every detector"
              accentColor="#EF4444"
              icon={CheckCircle2}
            />
            <StatCard
              label="Model Disagreements"
              value={data.disagreed_count}
              sub="Flagged by some models, but not all"
              accentColor="#F59E0B"
              icon={AlertTriangle}
            />
            <StatCard
              label="Agreed Normal"
              value={data.agreed_normal}
              sub="Unanimously considered inliers"
              accentColor="#10B981"
              icon={CheckCircle2}
            />
          </div>

          {/* Comparative Metrics Table */}
          <div className="saas-card overflow-hidden">
            <div className="p-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/20 text-xs font-bold text-slate-800 dark:text-slate-200">
              Detector Comparison Metrics
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 uppercase text-[10px] font-bold">
                    <th className="py-2.5 px-4">Model</th>
                    <th className="py-2.5 px-4">Score Separation</th>
                    <th className="py-2.5 px-4">Anomalies Detected</th>
                    <th className="py-2.5 px-4">Training Time (s)</th>
                    <th className="py-2.5 px-4">F1 Score (Evaluation)</th>
                    <th className="py-2.5 px-4">ROC-AUC (Evaluation)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono text-[11px]">
                  {data.comparison_table.map((row, i) => (
                    <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/30">
                      <td className="py-3 px-4 font-bold text-slate-800 dark:text-slate-200 font-sans">
                        {row.Model}
                      </td>
                      <td className="py-3 px-4 font-bold text-indigo-400">
                        {typeof row["Score Sep."] === "number" ? row["Score Sep."].toFixed(4) : row["Score Sep."]}
                      </td>
                      <td className="py-3 px-4 text-slate-300">
                        {row.Anomalies}
                      </td>
                      <td className="py-3 px-4 text-slate-400">
                        {typeof row["Fit Time (s)"] === "number" ? row["Fit Time (s)"].toFixed(3) : row["Fit Time (s)"]}s
                      </td>
                      <td className="py-3 px-4 text-slate-400">
                        {row.F1 !== undefined && row.F1 !== null && typeof row.F1 === "number" ? row.F1.toFixed(3) : "N/A"}
                      </td>
                      <td className="py-3 px-4 text-slate-400">
                        {row["ROC-AUC"] !== undefined && row["ROC-AUC"] !== null && typeof row["ROC-AUC"] === "number" ? row["ROC-AUC"].toFixed(3) : "N/A"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pairwise Jaccard Similarity */}
          <div className="saas-card p-5 space-y-4">
            <div>
              <div className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Pairwise Jaccard Overlap
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                Intersection over union of outlier sets (1.0 = identical anomaly sets, 0.0 = completely disjoint).
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
              {data.pairwise_jaccard.map((item, i) => (
                <div key={i} className="p-3 rounded-lg border border-slate-200 dark:border-slate-800 space-y-2">
                  <div className="text-xs font-bold text-slate-800 dark:text-slate-200">
                    {item.model_a} × {item.model_b}
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400">Similarity Index:</span>
                    <span className="font-mono font-bold text-indigo-400">
                      {item.jaccard.toFixed(3)}
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-indigo-500"
                      style={{ width: `${item.jaccard * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="saas-card p-12 text-center text-slate-400 space-y-2">
          <p className="text-xs">Click "Run Multi-Model Benchmark" to train and compare all detectors.</p>
        </div>
      )}
    </div>
  );
};

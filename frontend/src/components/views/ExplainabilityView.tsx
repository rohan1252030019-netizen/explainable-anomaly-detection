"use client";

import React, { useState, useEffect } from "react";
import { useApp } from "@/context/AppContext";
import { api, GlobalImportanceResponse } from "@/lib/api";
import { StatCard } from "@/components/ui/StatCard";
import { Sparkles, BarChart2, CheckCircle2, Info, ArrowUpRight } from "lucide-react";

export const ExplainabilityView: React.FC = () => {
  const { overview, dataset, setActiveTab } = useApp();

  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<GlobalImportanceResponse | null>(null);
  const [useShap, setUseShap] = useState(false);
  const [topK, setTopK] = useState(8);

  // Deep dive selected feature
  const [selectedFeature, setSelectedFeature] = useState<string>("");
  const [distData, setDistData] = useState<any | null>(null);

  const loadGlobal = async () => {
    if (!overview?.ready) return;
    try {
      const res = await api.getGlobalImportance();
      setData(res);
      if (res.global_importance && res.global_importance.length > 0) {
        setSelectedFeature(res.global_importance[0].feature);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);
    try {
      await api.generateExplanations({ use_shap: useShap, top_k: topK });
      await loadGlobal();
    } catch (err: any) {
      alert(`Generation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadFeatureDist = async (feat: string) => {
    try {
      const res = await api.getFeatureDistribution(feat);
      setDistData(res);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadGlobal();
  }, [overview?.ready]);

  useEffect(() => {
    if (selectedFeature) {
      loadFeatureDist(selectedFeature);
    }
  }, [selectedFeature]);

  if (!overview?.ready) {
    return (
      <div className="saas-card p-12 text-center text-slate-400 space-y-3">
        <p className="text-sm">Run anomaly detection before analyzing explainability.</p>
        <button
          onClick={() => setActiveTab("detection")}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg"
        >
          Go to Detection Workflow
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Configuration & Trigger Card */}
      <div className="saas-card p-5 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="space-y-1">
          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Sparkles size={16} className="text-indigo-400" />
            <span>Global Anomaly Driver Attribution</span>
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Decompose model decisions across the entire population to find which features most strongly separate outliers from normal observations.
          </p>
        </div>

        <div className="flex items-center gap-4 shrink-0">
          <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useShap}
              onChange={(e) => setUseShap(e.target.checked)}
              className="w-3.5 h-3.5 accent-indigo-600 rounded"
            />
            <span>Use SHAP (TreeExact)</span>
          </label>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-bold rounded-lg shadow-sm transition-all cursor-pointer flex items-center gap-2"
          >
            <Sparkles size={14} />
            <span>{loading ? "Computing Attributions..." : "Generate Explanations"}</span>
          </button>
        </div>
      </div>

      {data?.ready && data.global_importance ? (
        <>
          {/* Summary KPIs */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <StatCard
              label="Anomalies Analyzed"
              value={data.n_explained || overview.n_anomalies || 0}
              sub="Individual explanations generated"
              icon={CheckCircle2}
            />
            <StatCard
              label="Primary Anomaly Driver"
              value={data.top_driver || "—"}
              sub="Highest combined impact score"
              accentColor="#EF4444"
              icon={Sparkles}
            />
            <StatCard
              label="XAI Method"
              value={useShap ? "SHAP TreeExplainer" : "Perturbation & Robust Z"}
              sub="Model-agnostic feature deviation"
              icon={BarChart2}
            />
          </div>

          {/* Main Content: Global Feature Importance Table & Bar */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Global Ranking Table */}
            <div className="saas-card p-5 lg:col-span-2 space-y-4">
              <div className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Top Anomaly-Driving Features
              </div>

              <div className="space-y-3">
                {data.global_importance.slice(0, 8).map((item, i) => {
                  const maxScore = data.global_importance?.[0]?.mean_combined_score || 1;
                  const pct = (item.mean_combined_score / maxScore) * 100;
                  const isSelected = selectedFeature === item.feature;

                  return (
                    <div
                      key={item.feature}
                      onClick={() => setSelectedFeature(item.feature)}
                      className={`p-3 rounded-lg border transition-all cursor-pointer ${
                        isSelected
                          ? "border-indigo-500 bg-indigo-500/10"
                          : "border-slate-200 dark:border-slate-800 hover:border-slate-700 bg-slate-50/50 dark:bg-slate-900/40"
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs mb-1.5">
                        <div className="flex items-center gap-2 font-bold text-slate-800 dark:text-slate-200">
                          <span className="text-indigo-400">#{i + 1}</span>
                          <span>{item.feature}</span>
                        </div>
                        <div className="flex items-center gap-3 text-[11px] text-slate-400">
                          <span>{item.frequency} anomalies</span>
                          <span className="font-mono font-bold text-indigo-400">
                            score {item.mean_combined_score.toFixed(3)}
                          </span>
                        </div>
                      </div>

                      <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-rose-500"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Feature Distribution Separation Deep-Dive */}
            <div className="saas-card p-5 space-y-4 flex flex-col justify-between">
              <div>
                <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
                  Distribution Separation
                </div>
                <h4 className="text-sm font-bold text-slate-800 dark:text-slate-100">
                  {selectedFeature}
                </h4>
                <p className="text-[11px] text-slate-400 mt-1">
                  Comparison of empirical medians between normal cluster and outlier records.
                </p>
              </div>

              {distData && (
                <div className="space-y-4 py-2">
                  <div className="grid grid-cols-2 gap-3 text-center">
                    <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                      <div className="text-[10px] uppercase font-bold text-blue-400">
                        Normal Median
                      </div>
                      <div className="text-lg font-bold text-blue-300 mt-0.5">
                        {distData.normal_median}
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20">
                      <div className="text-[10px] uppercase font-bold text-rose-400">
                        Anomaly Median
                      </div>
                      <div className="text-lg font-bold text-rose-400 mt-0.5">
                        {distData.anomaly_median}
                      </div>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950/40 border border-slate-800 text-center">
                    <div className="text-[10px] uppercase font-bold text-slate-400">
                      Distribution Shift (Separation)
                    </div>
                    <div className="text-xl font-extrabold text-amber-400 mt-0.5">
                      Δ {distData.separation}
                    </div>
                  </div>
                </div>
              )}

              <div className="text-[10px] text-slate-500 border-t border-slate-800 pt-3">
                Select any feature on the left to inspect its empirical separation.
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="saas-card p-12 text-center text-slate-400 space-y-3">
          <p className="text-xs">
            Global feature attributions have not been computed yet.
          </p>
          <button
            onClick={handleGenerate}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-sm"
          >
            Compute Feature Attributions Now
          </button>
        </div>
      )}

      {/* Academic Methodology Box */}
      <div className="saas-card p-4 text-xs text-slate-400 space-y-1 bg-slate-950/20 border-slate-800">
        <div className="font-bold text-slate-300 flex items-center gap-1.5">
          <Info size={14} className="text-indigo-400" />
          <span>Academic Evaluation Note:</span>
        </div>
        <p className="leading-relaxed text-[11px]">
          Feature contributions reflect mathematical deviations and sensitivity gradients from the learned normal boundary. They provide statistical transparency into model scoring rather than causal proofs.
        </p>
      </div>
    </div>
  );
};

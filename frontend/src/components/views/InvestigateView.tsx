"use client";

import React, { useState, useEffect } from "react";
import { useApp } from "@/context/AppContext";
import { api, InvestigationDetail } from "@/lib/api";
import { ScoreGauge } from "@/components/ui/ScoreGauge";
import { SeverityBadge } from "@/components/ui/Badge";
import { FeatureContributionChart } from "@/components/charts/FeatureContributionChart";
import { ObservedVsTypicalRange } from "@/components/charts/ObservedVsTypicalRange";
import {
  ArrowLeft,
  Sparkles,
  CheckCircle2,
  FileCode,
  Info,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

export const InvestigateView: React.FC = () => {
  const { overview, setActiveTab } = useApp();

  const [rowId, setRowId] = useState<number>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("xaud-selected-anomaly");
      if (saved) return Number(saved);
    }
    return 0;
  });

  const [detail, setDetail] = useState<InvestigationDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveDetailTab] = useState<"features" | "comparison" | "raw">("features");

  const loadDetail = async (id: number) => {
    setLoading(true);
    try {
      const res = await api.investigateAnomaly(id);
      setDetail(res);
      setRowId(id);
      localStorage.setItem("xaud-selected-anomaly", String(id));
    } catch (err: any) {
      console.error("Failed to load anomaly investigation detail:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (overview?.ready) {
      loadDetail(rowId);
    }
  }, [overview?.ready]);

  if (!overview?.ready) {
    return (
      <div className="saas-card p-12 text-center text-slate-400 space-y-3">
        <p className="text-sm">Run anomaly detection to begin investigating outliers.</p>
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
      {/* Top Breadcrumb & ID Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <button
          onClick={() => setActiveTab("anomalies")}
          className="inline-flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
        >
          <ArrowLeft size={14} />
          <span>Back to Anomalies Table</span>
        </button>

        {/* Observation Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Observation Index:</span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => loadDetail(Math.max(0, rowId - 1))}
              disabled={rowId <= 0}
              className="p-1 rounded border border-slate-700 disabled:opacity-30 hover:bg-slate-800 text-xs"
            >
              <ChevronLeft size={14} />
            </button>
            <input
              type="number"
              min="0"
              value={rowId}
              onChange={(e) => {
                const val = Number(e.target.value);
                if (val >= 0) loadDetail(val);
              }}
              className="saas-input w-20 px-2 py-1 text-xs text-center font-mono font-bold"
            />
            <button
              onClick={() => loadDetail(rowId + 1)}
              className="p-1 rounded border border-slate-700 hover:bg-slate-800 text-xs"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="saas-card p-16 text-center text-slate-400">
          <span className="animate-pulse font-medium">Decomposing anomaly & computing explanations...</span>
        </div>
      ) : detail ? (
        <>
          {/* Header Banner: Risk Tier, Score Gauge, Why was it flagged */}
          <div className="saas-card p-6 border-l-4 border-l-rose-500 bg-gradient-to-r from-slate-900 to-indigo-950/20">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
              {/* Score Gauge */}
              <div className="flex flex-col items-center justify-center border-b md:border-b-0 md:border-r border-slate-800 pb-4 md:pb-0 md:pr-4">
                <div className="mb-2">
                  <SeverityBadge severity={detail.severity} />
                </div>
                <ScoreGauge
                  score={detail.normalized_score}
                  severity={detail.severity}
                  threshold={detail.threshold}
                />
                <div className="text-[11px] text-slate-400 mt-2">
                  Observation #{detail.row_id} {detail.rank !== null && `· Rank #${detail.rank}`}
                </div>
              </div>

              {/* Plain-English Explanation Card */}
              <div className="md:col-span-2 space-y-3">
                <div className="flex items-center gap-2 text-xs font-bold text-indigo-400 uppercase tracking-wider">
                  <Sparkles size={14} />
                  <span>Transparent Explainability (XAI)</span>
                </div>

                <h3 className="text-base font-extrabold text-slate-100">
                  Why was this record flagged as anomalous?
                </h3>

                <p className="text-xs text-slate-300 leading-relaxed bg-slate-950/50 p-4 rounded-xl border border-slate-800">
                  {detail.explanation_text || "Record exhibits statistical deviation from the baseline normal cluster."}
                </p>

                {detail.confidence_note && (
                  <div className="flex items-center gap-2 text-[11px] text-slate-400">
                    <Info size={13} className="text-indigo-400 shrink-0" />
                    <span>{detail.confidence_note}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Tabbed Analysis Sections */}
          <div className="saas-card overflow-hidden">
            <div className="flex border-b border-slate-200 dark:border-slate-800 px-5 pt-3 bg-slate-50 dark:bg-slate-950/40">
              <button
                onClick={() => setActiveDetailTab("features")}
                className={`pb-3 text-xs font-semibold border-b-2 cursor-pointer transition-colors mr-6 ${
                  activeTab === "features"
                    ? "border-indigo-500 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Feature Contributions
              </button>
              <button
                onClick={() => setActiveDetailTab("comparison")}
                className={`pb-3 text-xs font-semibold border-b-2 cursor-pointer transition-colors mr-6 ${
                  activeTab === "comparison"
                    ? "border-indigo-500 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Observed vs Baseline Normal
              </button>
              <button
                onClick={() => setActiveDetailTab("raw")}
                className={`pb-3 text-xs font-semibold border-b-2 cursor-pointer transition-colors ${
                  activeTab === "raw"
                    ? "border-indigo-500 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                Raw Record Attributes
              </button>
            </div>

            <div className="p-5">
              {/* Tab 1: Feature Contributions */}
              {activeTab === "features" && (
                <FeatureContributionChart features={detail.top_features} />
              )}

              {/* Tab 2: Observed vs Typical Baseline Range */}
              {activeTab === "comparison" && (
                <ObservedVsTypicalRange items={detail.feature_comparison} />
              )}

              {/* Tab 3: Raw Record Data */}
              {activeTab === "raw" && (
                <div className="space-y-3">
                  <div className="text-xs font-semibold text-slate-400">
                    Observation #{detail.row_id} Attribute Profile
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 font-mono text-xs">
                    {Object.entries(detail.raw_record).map(([k, v]) => (
                      <div
                        key={k}
                        className="p-2.5 rounded bg-slate-950/40 border border-slate-800 space-y-1"
                      >
                        <span className="text-[10px] text-slate-500 block truncate uppercase font-sans">
                          {k}
                        </span>
                        <span className="font-bold text-slate-200 block truncate">
                          {v !== null ? String(v) : "null"}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      ) : (
        <div className="saas-card p-12 text-center text-slate-400">
          Observation #{rowId} not found.
        </div>
      )}
    </div>
  );
};

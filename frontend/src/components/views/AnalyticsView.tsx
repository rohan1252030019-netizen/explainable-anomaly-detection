"use client";

import React, { useState, useEffect } from "react";
import { useApp } from "@/context/AppContext";
import { api, PcaAnalyticsResponse } from "@/lib/api";
import { PcaScatterChart } from "@/components/charts/PcaScatterChart";
import { CorrelationHeatmap } from "@/components/charts/CorrelationHeatmap";
import { LineChart, Sparkles } from "lucide-react";

export const AnalyticsView: React.FC = () => {
  const { overview, dataset, setActiveTab } = useApp();

  const [pcaData, setPcaData] = useState<PcaAnalyticsResponse | null>(null);
  const [corrData, setCorrData] = useState<{ features: string[]; matrix: number[][] } | null>(null);
  const [loading, setLoading] = useState(false);

  const loadAnalytics = async () => {
    if (!dataset?.loaded) return;
    setLoading(true);
    try {
      const [pcaRes, corrRes] = await Promise.all([
        api.getPca().catch(() => null),
        api.getCorrelations(12).catch(() => null),
      ]);
      setPcaData(pcaRes);
      setCorrData(corrRes);
    } catch (err) {
      console.error("Failed to load analytics data", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, [dataset?.loaded, overview?.ready]);

  if (!dataset?.loaded) {
    return (
      <div className="saas-card p-12 text-center text-slate-400 space-y-3">
        <p className="text-sm">Load a dataset to explore visual analytics.</p>
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <LineChart size={18} className="text-indigo-400" />
            <span>Visual Analytics & Dimensionality Reduction</span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Examine cluster separability, high-dimensional projections, and cross-feature interactions.
          </p>
        </div>
      </div>

      {loading ? (
        <div className="saas-card p-16 text-center text-slate-400">
          <span className="animate-pulse">Computing PCA projections and correlation matrices...</span>
        </div>
      ) : (
        <>
          {/* PCA 2D Scatter */}
          {pcaData && (
            <div className="saas-card p-5">
              <PcaScatterChart data={pcaData} />
            </div>
          )}

          {/* Correlation Matrix */}
          {corrData && (
            <div className="saas-card p-5">
              <CorrelationHeatmap
                features={corrData.features}
                matrix={corrData.matrix}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
};

"use client";

import React, { useState, useEffect } from "react";
import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import {
  Settings,
  Sun,
  Moon,
  Download,
  RotateCcw,
  Clock,
  CheckCircle2,
} from "lucide-react";

export const SettingsView: React.FC = () => {
  const { theme, toggleTheme, refreshAll, setActiveTab } = useApp();

  const [experiments, setExperiments] = useState<any[]>([]);
  const [resetting, setResetting] = useState(false);

  const loadExperiments = async () => {
    try {
      const res = await api.getExperiments();
      setExperiments(res.experiments || []);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadExperiments();
  }, []);

  const handleReset = async () => {
    if (!confirm("Are you sure you want to reset all dataset and detection state?")) {
      return;
    }
    setResetting(true);
    try {
      await api.resetState();
      await refreshAll();
      setActiveTab("dataset");
    } catch (e) {
      console.error(e);
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Theme Settings */}
      <div className="saas-card p-6 space-y-4">
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Settings size={16} className="text-indigo-400" />
          <span>Interface & Appearance</span>
        </h3>

        <div className="flex items-center justify-between p-4 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
          <div>
            <div className="text-xs font-semibold text-slate-800 dark:text-slate-200">
              Color Theme (Current: {theme === "dark" ? "Dark Mode" : "Light Mode"})
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
              High-contrast typography with calibrated colors for data analytics.
            </p>
          </div>

          <button
            onClick={toggleTheme}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm transition-colors cursor-pointer"
          >
            {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
            <span>Switch to {theme === "dark" ? "Light" : "Dark"} Mode</span>
          </button>
        </div>
      </div>

      {/* Export Suite */}
      <div className="saas-card p-6 space-y-4">
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Download size={16} className="text-indigo-400" />
          <span>Data & Model Artifact Exports</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <a
            href={api.exportPredictionsUrl}
            target="_blank"
            rel="noreferrer"
            className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all text-xs font-semibold space-y-1 block"
          >
            <div className="text-slate-800 dark:text-slate-200 flex items-center justify-between">
              <span>Full Predictions CSV</span>
              <Download size={14} />
            </div>
            <p className="text-[10px] text-slate-400 font-normal">
              All records with scores, severity, and ranks.
            </p>
          </a>

          <a
            href={api.exportAnomaliesUrl}
            target="_blank"
            rel="noreferrer"
            className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all text-xs font-semibold space-y-1 block"
          >
            <div className="text-slate-800 dark:text-slate-200 flex items-center justify-between">
              <span>Anomalies Only CSV</span>
              <Download size={14} />
            </div>
            <p className="text-[10px] text-slate-400 font-normal">
              Filtered outliers sorted by rank.
            </p>
          </a>

          <a
            href={api.exportExplanationsUrl}
            target="_blank"
            rel="noreferrer"
            className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all text-xs font-semibold space-y-1 block"
          >
            <div className="text-slate-800 dark:text-slate-200 flex items-center justify-between">
              <span>Explanations JSON</span>
              <Download size={14} />
            </div>
            <p className="text-[10px] text-slate-400 font-normal">
              Structured feature contribution attributions.
            </p>
          </a>
        </div>
      </div>

      {/* Experiment History */}
      <div className="saas-card overflow-hidden">
        <div className="p-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/40 text-xs font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
          <Clock size={14} />
          <span>Session Experiment History ({experiments.length} runs)</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse font-mono text-[11px]">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 uppercase text-[10px] font-bold font-sans">
                <th className="py-2.5 px-4">#</th>
                <th className="py-2.5 px-4">Model</th>
                <th className="py-2.5 px-4">Threshold</th>
                <th className="py-2.5 px-4">Anomalies</th>
                <th className="py-2.5 px-4">Rate</th>
                <th className="py-2.5 px-4">Fit Time</th>
                <th className="py-2.5 px-4">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60">
              {experiments.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-500 font-sans">
                    No experiments logged yet.
                  </td>
                </tr>
              ) : (
                experiments.map((exp) => (
                  <tr key={exp.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/30">
                    <td className="py-2.5 px-4 text-slate-500 font-sans">#{exp.id}</td>
                    <td className="py-2.5 px-4 font-bold text-indigo-400 font-sans">{exp.model}</td>
                    <td className="py-2.5 px-4 text-slate-300 font-sans">{exp.threshold_method}</td>
                    <td className="py-2.5 px-4 text-rose-400 font-bold">{exp.n_anomalies}</td>
                    <td className="py-2.5 px-4 text-slate-400">{((exp.anomaly_rate || 0) * 100).toFixed(1)}%</td>
                    <td className="py-2.5 px-4 text-slate-400">{exp.fit_time_sec?.toFixed(3)}s</td>
                    <td className="py-2.5 px-4 text-slate-500 text-[10px]">{exp.timestamp}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Danger Zone / Reset */}
      <div className="saas-card p-6 border-rose-500/30 bg-rose-500/5 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-xs font-bold text-rose-400 uppercase tracking-wider">
              Reset Session State
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Clear loaded dataset, preprocessed matrix, trained models, and all analysis results from memory.
            </p>
          </div>

          <button
            onClick={handleReset}
            disabled={resetting}
            className="flex items-center gap-2 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold rounded-lg shadow-sm transition-colors cursor-pointer"
          >
            <RotateCcw size={14} />
            <span>{resetting ? "Resetting..." : "Reset All State"}</span>
          </button>
        </div>
      </div>
    </div>
  );
};

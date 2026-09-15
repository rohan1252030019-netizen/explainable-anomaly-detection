"use client";

import React from "react";
import { useApp } from "@/context/AppContext";
import {
  LayoutDashboard,
  Database,
  Play,
  AlertTriangle,
  Sparkles,
  LineChart,
  Layers,
  FileText,
  Settings,
  ShieldCheck,
} from "lucide-react";

interface NavItem {
  id: string;
  label: string;
  icon: any;
  badge?: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "dataset", label: "Dataset", icon: Database },
  { id: "detection", label: "Detection", icon: Play },
  { id: "anomalies", label: "Anomalies", icon: AlertTriangle },
  { id: "explainability", label: "Explainability", icon: Sparkles },
  { id: "analytics", label: "Visual Analytics", icon: LineChart },
  { id: "models", label: "Model Lab", icon: Layers },
  { id: "reports", label: "Audit Reports", icon: FileText },
  { id: "settings", label: "Settings", icon: Settings },
];

export const Sidebar: React.FC = () => {
  const { activeTab, setActiveTab, dataset, overview } = useApp();

  const isDatasetLoaded = !!dataset?.loaded;
  const isFeatureReady = !!dataset?.feature_matrix_ready;
  const isDetectionDone = !!overview?.ready;

  return (
    <aside className="w-64 border-r border-slate-200 dark:border-slate-800/80 bg-white dark:bg-slate-900/60 backdrop-blur flex flex-col h-screen sticky top-0 shrink-0 select-none">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-200 dark:border-slate-800/80 flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-indigo-500 flex items-center justify-center text-white font-black text-lg shadow-sm shadow-indigo-500/20">
          X
        </div>
        <div>
          <div className="text-base font-extrabold tracking-tight text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
            XAUD <span className="text-[10px] px-1.5 py-0.2 bg-indigo-500/10 text-indigo-400 font-semibold rounded">v2.0</span>
          </div>
          <div className="text-[10px] text-slate-400 dark:text-slate-500 tracking-wider font-semibold uppercase">
            Anomaly Intelligence
          </div>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        <div className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider px-3 py-2">
          Platform
        </div>

        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                isActive
                  ? "bg-indigo-600 text-white shadow-sm shadow-indigo-600/30"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/50"
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon size={16} />
                <span>{item.label}</span>
              </div>
              {item.id === "anomalies" && overview?.n_anomalies !== undefined && overview.n_anomalies > 0 && (
                <span
                  className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                    isActive ? "bg-white/20 text-white" : "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                  }`}
                >
                  {overview.n_anomalies}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Live Pipeline Status */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-800/80 text-[11px] bg-slate-50/50 dark:bg-slate-950/20">
        <div className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2 flex items-center justify-between">
          <span>Pipeline Status</span>
          <ShieldCheck size={12} className="text-emerald-500" />
        </div>
        <div className="space-y-1.5 text-slate-500 dark:text-slate-400">
          <div className="flex items-center gap-2">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isDatasetLoaded ? "bg-emerald-500" : "bg-slate-400 dark:bg-slate-700"
              }`}
            />
            <span>Dataset Loaded</span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isFeatureReady ? "bg-emerald-500" : "bg-slate-400 dark:bg-slate-700"
              }`}
            />
            <span>Features Processed</span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isDetectionDone ? "bg-emerald-500" : "bg-slate-400 dark:bg-slate-700"
              }`}
            />
            <span>Detection Fitted</span>
          </div>
        </div>

        {dataset?.dataset_name && (
          <div className="mt-3 pt-2 border-t border-slate-200 dark:border-slate-800 text-[10px] text-slate-400 truncate">
            {dataset.dataset_name}
          </div>
        )}
      </div>
    </aside>
  );
};

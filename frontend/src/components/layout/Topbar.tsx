"use client";

import React from "react";
import { useApp } from "@/context/AppContext";
import { Search, Sun, Moon, Database, Play, Sparkles } from "lucide-react";
import { api } from "@/lib/api";

export const Topbar: React.FC = () => {
  const {
    activeTab,
    setCommandPaletteOpen,
    theme,
    toggleTheme,
    dataset,
    overview,
    refreshAll,
    setActiveTab,
  } = useApp();

  const handleQuickDemo = async () => {
    try {
      await api.loadSample();
      await api.runDetection({
        model_name: "Isolation Forest",
        contamination: 0.05,
      });
      await refreshAll();
      setActiveTab("dashboard");
    } catch (err) {
      console.error("Quick demo failed", err);
    }
  };

  return (
    <header className="h-16 border-b border-slate-200 dark:border-slate-800/80 bg-white/80 dark:bg-slate-900/60 backdrop-blur sticky top-0 z-30 px-6 flex items-center justify-between">
      {/* Page Title & Breadcrumb */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          XAUD
        </span>
        <span className="text-slate-300 dark:text-slate-700">/</span>
        <h1 className="text-sm font-bold text-slate-800 dark:text-slate-100 capitalize">
          {activeTab.replace("_", " ")}
        </h1>
      </div>

      {/* Global Center Search Trigger */}
      <button
        onClick={() => setCommandPaletteOpen(true)}
        className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/40 text-xs text-slate-400 hover:border-indigo-500/40 hover:text-slate-200 transition-all cursor-pointer w-72 justify-between"
      >
        <div className="flex items-center gap-2">
          <Search size={14} />
          <span>Quick actions & search...</span>
        </div>
        <kbd className="text-[10px] font-mono px-1.5 py-0.5 bg-slate-200 dark:bg-slate-800 rounded border border-slate-300 dark:border-slate-700">
          Ctrl K
        </kbd>
      </button>

      {/* Right Controls */}
      <div className="flex items-center gap-3">
        {/* Active Dataset Chip */}
        {dataset?.loaded ? (
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-indigo-500/10 text-indigo-500 dark:text-indigo-400 border border-indigo-500/20">
            <Database size={12} />
            <span className="truncate max-w-[120px]">{dataset.dataset_name}</span>
          </div>
        ) : (
          <button
            onClick={handleQuickDemo}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm transition-all cursor-pointer"
          >
            <Sparkles size={13} />
            <span>1-Click Demo</span>
          </button>
        )}

        {/* Model Status Chip */}
        {overview?.ready && (
          <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
            <Play size={11} className="text-emerald-500" />
            <span>{overview.model_name}</span>
            <span className="text-rose-400 font-bold ml-1">
              ({overview.n_anomalies} anomalies)
            </span>
          </div>
        )}

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          aria-label="Toggle Theme"
          className="w-9 h-9 rounded-lg border border-slate-200 dark:border-slate-800 flex items-center justify-center text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100 transition-colors cursor-pointer"
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  );
};

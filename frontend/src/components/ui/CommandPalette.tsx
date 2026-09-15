"use client";

import React, { useState, useEffect, useRef } from "react";
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
  Sun,
  Moon,
  Upload,
  Search,
  X,
} from "lucide-react";
import { api } from "@/lib/api";

export const CommandPalette: React.FC = () => {
  const {
    commandPaletteOpen,
    setCommandPaletteOpen,
    setActiveTab,
    theme,
    toggleTheme,
    refreshAll,
  } = useApp();

  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (commandPaletteOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
    } else {
      setQuery("");
    }
  }, [commandPaletteOpen]);

  if (!commandPaletteOpen) return null;

  const actions = [
    {
      id: "overview",
      name: "Go to Dashboard Overview",
      category: "Navigation",
      icon: LayoutDashboard,
      run: () => setActiveTab("dashboard"),
    },
    {
      id: "dataset",
      name: "Go to Dataset Management",
      category: "Navigation",
      icon: Database,
      run: () => setActiveTab("dataset"),
    },
    {
      id: "detection",
      name: "Go to Anomaly Detection Workflow",
      category: "Navigation",
      icon: Play,
      run: () => setActiveTab("detection"),
    },
    {
      id: "anomalies",
      name: "Go to Anomalies Explorer",
      category: "Navigation",
      icon: AlertTriangle,
      run: () => setActiveTab("anomalies"),
    },
    {
      id: "explainability",
      name: "Go to Explainability Center",
      category: "Navigation",
      icon: Sparkles,
      run: () => setActiveTab("explainability"),
    },
    {
      id: "analytics",
      name: "Go to Visual Analytics (PCA)",
      category: "Navigation",
      icon: LineChart,
      run: () => setActiveTab("analytics"),
    },
    {
      id: "models",
      name: "Go to Model Lab (Benchmark)",
      category: "Navigation",
      icon: Layers,
      run: () => setActiveTab("models"),
    },
    {
      id: "reports",
      name: "Go to Audit Reports",
      category: "Navigation",
      icon: FileText,
      run: () => setActiveTab("reports"),
    },
    {
      id: "settings",
      name: "Go to Settings & Export",
      category: "Navigation",
      icon: Settings,
      run: () => setActiveTab("settings"),
    },
    {
      id: "sample",
      name: "Load Benchmark Sample Dataset",
      category: "Quick Actions",
      icon: Upload,
      run: async () => {
        await api.loadSample();
        await refreshAll();
        setActiveTab("dashboard");
      },
    },
    {
      id: "theme",
      name: `Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`,
      category: "System",
      icon: theme === "dark" ? Sun : Moon,
      run: toggleTheme,
    },
  ];

  const filtered = actions.filter((item) =>
    item.name.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-24 bg-black/60 backdrop-blur-sm p-4 animate-in fade-in duration-150">
      <div
        className="w-full max-w-xl saas-card shadow-2xl overflow-hidden border border-slate-700/60 bg-slate-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center px-4 py-3 border-b border-slate-800">
          <Search size={18} className="text-slate-400 mr-2" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Type a command or search page..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
          />
          <button
            onClick={() => setCommandPaletteOpen(false)}
            className="text-slate-500 hover:text-slate-300 p-1"
          >
            <X size={16} />
          </button>
        </div>

        <div className="max-h-80 overflow-y-auto p-2">
          {filtered.length === 0 ? (
            <div className="text-center py-8 text-xs text-slate-500">
              No matching commands found.
            </div>
          ) : (
            filtered.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    item.run();
                    setCommandPaletteOpen(false);
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left text-xs font-medium text-slate-200 hover:bg-indigo-600/20 hover:text-indigo-300 transition-colors group cursor-pointer"
                >
                  <div className="w-7 h-7 rounded-md bg-slate-800 flex items-center justify-center text-slate-400 group-hover:text-indigo-400">
                    <Icon size={14} />
                  </div>
                  <span className="flex-1">{item.name}</span>
                  <span className="text-[10px] text-slate-500 font-normal uppercase">
                    {item.category}
                  </span>
                </button>
              );
            })
          )}
        </div>

        <div className="px-4 py-2 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-500 bg-slate-950/50">
          <span>Navigate with mouse or enter</span>
          <span className="flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 bg-slate-800 rounded border border-slate-700 text-[10px]">
              ESC
            </kbd>{" "}
            to close
          </span>
        </div>
      </div>
    </div>
  );
};

"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { api, DatasetSummary, DetectionOverview, HealthResponse } from "@/lib/api";

interface AppContextType {
  dataset: DatasetSummary | null;
  overview: DetectionOverview | null;
  health: HealthResponse | null;
  theme: "dark" | "light";
  toggleTheme: () => void;
  isLoading: boolean;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  commandPaletteOpen: boolean;
  setCommandPaletteOpen: (open: boolean) => void;
  refreshDataset: () => Promise<void>;
  refreshOverview: () => Promise<void>;
  refreshAll: () => Promise<void>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [overview, setOverview] = useState<DetectionOverview | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

  // Read theme from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("xaud-theme") as "dark" | "light" | null;
    if (saved) {
      setTheme(saved);
      document.documentElement.classList.toggle("dark", saved === "dark");
    } else {
      document.documentElement.classList.add("dark");
    }
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("xaud-theme", next);
    document.documentElement.classList.toggle("dark", next === "dark");
  };

  const refreshDataset = async () => {
    try {
      const d = await api.getDatasetSummary();
      setDataset(d);
    } catch (e) {
      console.warn("Could not load dataset summary", e);
    }
  };

  const refreshOverview = async () => {
    try {
      const o = await api.getResults();
      setOverview(o);
    } catch (e) {
      console.warn("Could not load detection results", e);
    }
  };

  const refreshAll = async () => {
    setIsLoading(true);
    try {
      const h = await api.getHealth().catch(() => null);
      setHealth(h);
      await refreshDataset();
      await refreshOverview();
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    refreshAll();
  }, []);

  // Global Ctrl+K keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <AppContext.Provider
      value={{
        dataset,
        overview,
        health,
        theme,
        toggleTheme,
        isLoading,
        activeTab,
        setActiveTab,
        commandPaletteOpen,
        setCommandPaletteOpen,
        refreshDataset,
        refreshOverview,
        refreshAll,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useApp must be used within an AppProvider");
  }
  return context;
}

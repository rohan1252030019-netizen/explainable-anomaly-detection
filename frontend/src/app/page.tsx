"use client";

import React, { useEffect, useRef } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useApp } from "@/context/AppContext";
import { Sidebar } from "@/components/layout/Sidebar";
import { Topbar } from "@/components/layout/Topbar";
import { CommandPalette } from "@/components/ui/CommandPalette";
import { PageHeader, type PageStatusTone } from "@/components/ui/PageHeader";
import { getNavGroup, getNavItem } from "@/lib/navigation";
import { DashboardView } from "@/components/views/DashboardView";
import { DatasetView } from "@/components/views/DatasetView";
import { DetectionView } from "@/components/views/DetectionView";
import { AnomaliesView } from "@/components/views/AnomaliesView";
import { InvestigateView } from "@/components/views/InvestigateView";
import { ExplainabilityView } from "@/components/views/ExplainabilityView";
import { AnalyticsView } from "@/components/views/AnalyticsView";
import { ModelsView } from "@/components/views/ModelsView";
import { RigorView } from "@/components/views/RigorView";
import { DriftMonitorView } from "@/components/views/DriftMonitorView";
import { BenchmarkLabView } from "@/components/views/BenchmarkLabView";
import { ExperimentsView } from "@/components/views/ExperimentsView";
import { ReportsView } from "@/components/views/ReportsView";
import { SettingsView } from "@/components/views/SettingsView";

interface ViewProps {
  onSelectAnomaly?: (rowId: number) => void;
}

const VIEWS: Record<string, React.ComponentType<ViewProps>> = {
  dashboard: DashboardView,
  dataset: DatasetView,
  detection: DetectionView,
  anomalies: AnomaliesView,
  investigate: InvestigateView,
  explainability: ExplainabilityView,
  analytics: AnalyticsView,
  models: ModelsView,
  drift: DriftMonitorView,
  benchmarks: BenchmarkLabView,
  rigor: RigorView,
  experiments: ExperimentsView,
  reports: ReportsView,
  settings: SettingsView,
};

export default function Home() {
  const { activeTab, setActiveTab, dataset, overview } = useApp();
  const reducedMotion = useReducedMotion();
  const ActiveView = VIEWS[activeTab] ?? DashboardView;

  const navItem = getNavItem(activeTab);
  const navGroup = getNavGroup(activeTab);

  const announcerRef = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    if (!announcerRef.current) return;
    announcerRef.current.textContent = `${navItem?.label ?? activeTab} view loaded`;
  }, [activeTab, navItem?.label]);

  const status: { tone: PageStatusTone; label: string } = overview?.ready
    ? {
        tone: "ready",
        label: `${overview.n_anomalies?.toLocaleString() ?? 0} anomalies`,
      }
    : dataset?.loaded
      ? { tone: "active", label: "Dataset ready" }
      : { tone: "idle", label: "Awaiting dataset" };

  const handleSelectAnomaly = (rowId: number) => {
    localStorage.setItem("xaud-selected-anomaly", String(rowId));
    setActiveTab("investigate");
  };

  return (
    <div className="flex min-h-svh bg-canvas">
      <a
        href="#main-content"
        className="sr-only-focusable fixed left-4 top-4 z-50 rounded-lg bg-accent px-3 py-2 text-xs font-semibold text-on-accent"
      >
        Skip to main content
      </a>

      <p
        ref={announcerRef}
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      />

      <Sidebar />

      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />

        <main
          id="main-content"
          tabIndex={-1}
          className="mx-auto w-full max-w-[1440px] flex-1 px-4 py-5 focus:outline-none sm:px-6 sm:py-7 lg:px-8"
        >
          <PageHeader
            group={navGroup}
            title={navItem?.label ?? activeTab}
            description={navItem?.description}
            status={status}
          />

          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={reducedMotion ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reducedMotion ? { opacity: 1 } : { opacity: 0, y: -4 }}
              transition={{
                duration: reducedMotion ? 0 : 0.18,
                ease: [0.16, 1, 0.3, 1],
              }}
            >
              <ActiveView onSelectAnomaly={handleSelectAnomaly} />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      <CommandPalette />
    </div>
  );
}

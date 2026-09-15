"use client";

import React from "react";
import { useApp } from "@/context/AppContext";
import { Sidebar } from "@/components/layout/Sidebar";
import { Topbar } from "@/components/layout/Topbar";
import { CommandPalette } from "@/components/ui/CommandPalette";

// View components
import { DashboardView } from "@/components/views/DashboardView";
import { DatasetView } from "@/components/views/DatasetView";
import { DetectionView } from "@/components/views/DetectionView";
import { AnomaliesView } from "@/components/views/AnomaliesView";
import { InvestigateView } from "@/components/views/InvestigateView";
import { ExplainabilityView } from "@/components/views/ExplainabilityView";
import { AnalyticsView } from "@/components/views/AnalyticsView";
import { ModelsView } from "@/components/views/ModelsView";
import { ReportsView } from "@/components/views/ReportsView";
import { SettingsView } from "@/components/views/SettingsView";

export default function Home() {
  const { activeTab, setActiveTab } = useApp();

  const handleSelectAnomaly = (rowId: number) => {
    localStorage.setItem("xaud-selected-anomaly", String(rowId));
    setActiveTab("investigate");
  };

  return (
    <div className="flex min-h-screen bg-slate-50 dark:bg-[#090D16]">
      {/* Sidebar Navigation */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Topbar />

        <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto">
          {activeTab === "dashboard" && <DashboardView />}
          {activeTab === "dataset" && <DatasetView />}
          {activeTab === "detection" && <DetectionView />}
          {activeTab === "anomalies" && (
            <AnomaliesView onSelectAnomaly={handleSelectAnomaly} />
          )}
          {activeTab === "investigate" && <InvestigateView />}
          {activeTab === "explainability" && <ExplainabilityView />}
          {activeTab === "analytics" && <AnalyticsView />}
          {activeTab === "models" && <ModelsView />}
          {activeTab === "reports" && <ReportsView />}
          {activeTab === "settings" && <SettingsView />}
        </main>
      </div>

      {/* Global Quick Action Palette (Ctrl+K) */}
      <CommandPalette />
    </div>
  );
}

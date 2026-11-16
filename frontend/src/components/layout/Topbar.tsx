"use client";

import React, { useState } from "react";
import {
  Cpu,
  Database,
  Menu,
  Moon,
  RefreshCw,
  Search,
  Sparkles,
  Sun,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import { getNavItem, getNavGroup } from "@/lib/navigation";
import { Button } from "@/components/ui/Button";
import { IconButton } from "@/components/ui/IconButton";
import { useToast } from "@/components/ui/Toast";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

export const Topbar: React.FC = () => {
  const {
    activeTab,
    setActiveTab,
    setCommandPaletteOpen,
    theme,
    toggleTheme,
    dataset,
    overview,
    refreshAll,
    setMobileNavOpen,
  } = useApp();
  const { toast } = useToast();

  const [busy, setBusy] = useState<"demo" | "refresh" | null>(null);

  const navItem = getNavItem(activeTab);
  const navGroup = getNavGroup(activeTab);
  const pageLabel = navItem?.label ?? activeTab;

  const handleQuickDemo = async () => {
    setBusy("demo");
    try {
      await api.loadSample();
      await api.runDetection({
        model_name: "Isolation Forest",
        contamination: 0.05,
      });
      await refreshAll();
      setActiveTab("dashboard");
      toast(
        "success",
        "Demo analysis ready",
        "Sample dataset loaded and scored with Isolation Forest."
      );
    } catch (err) {
      toast("error", "Could not run the demo", errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const handleRefresh = async () => {
    setBusy("refresh");
    try {
      await refreshAll();
    } finally {
      setBusy(null);
    }
  };

  return (
    <header
      className="sticky top-0 z-30 flex h-14 items-center justify-between gap-3 border-b border-edge bg-surface/90 px-4 backdrop-blur-xl sm:h-16 sm:px-6"
      data-print-hide
    >
      {/* Left: mobile nav trigger + breadcrumb */}
      <div className="flex min-w-0 items-center gap-3">
        <IconButton
          onClick={() => setMobileNavOpen(true)}
          variant="ghost"
          size="sm"
          className="lg:hidden"
          aria-label="Open navigation"
        >
          <Menu size={16} />
        </IconButton>

        <div className="hidden min-w-0 items-center gap-2 text-[11px] sm:flex">
          <span className="font-semibold text-ink-faint">XAUD</span>
          {navGroup && (
            <>
              <span className="text-edge-strong" aria-hidden="true">
                /
              </span>
              <span className="text-ink-faint">{navGroup}</span>
            </>
          )}
          <span className="text-edge-strong" aria-hidden="true">
            /
          </span>
          <span className="truncate font-semibold text-ink">{pageLabel}</span>
        </div>

        <span className="truncate text-sm font-bold tracking-tight text-ink sm:hidden">
          {pageLabel}
        </span>
      </div>

      {/* Center: command palette trigger */}
      <div className="flex flex-1 justify-end md:justify-center">
        <button
          onClick={() => setCommandPaletteOpen(true)}
          className={cn(
            "hidden items-center justify-between gap-2 rounded-lg border border-edge bg-inset px-3 py-1.5 text-[11px] text-ink-faint",
            "transition-colors duration-150 hover:border-accent-line hover:text-ink",
            "md:flex md:w-64 lg:w-72"
          )}
          aria-label="Search views and actions"
        >
          <span className="flex items-center gap-2">
            <Search size={13} aria-hidden="true" />
            <span>Search or jump to…</span>
          </span>
          <kbd className="rounded border border-edge bg-surface px-1.5 py-0.5 font-mono text-[10px]">
            ⌘K
          </kbd>
        </button>
      </div>

      {/* Right controls */}
      <div className="flex shrink-0 items-center gap-1.5">
        <IconButton
          onClick={() => setCommandPaletteOpen(true)}
          variant="ghost"
          size="sm"
          className="md:hidden"
          aria-label="Search views and actions"
        >
          <Search size={16} />
        </IconButton>

        {dataset?.loaded ? (
          <button
            onClick={() => setActiveTab("dataset")}
            className="hidden max-w-[180px] items-center gap-1.5 rounded-full border border-edge bg-inset px-2.5 py-1 text-[11px] font-medium text-ink-muted transition-colors hover:border-accent-line hover:text-ink md:flex"
            title={`Active dataset: ${dataset.dataset_name ?? "Untitled"}`}
          >
            <Database size={11} className="text-accent" aria-hidden="true" />
            <span className="truncate">{dataset.dataset_name}</span>
          </button>
        ) : (
          <Button
            size="sm"
            onClick={handleQuickDemo}
            disabled={busy !== null}
            className="hidden sm:inline-flex"
          >
            <Sparkles size={13} aria-hidden="true" />
            {busy === "demo" ? "Preparing…" : "Load demo"}
          </Button>
        )}

        {overview?.ready && (
          <button
            onClick={() => setActiveTab("anomalies")}
            title={`${overview.n_anomalies} anomalies flagged by ${overview.model_name}`}
            className="hidden items-center gap-1.5 rounded-full border border-edge bg-inset px-2.5 py-1 text-[11px] font-medium text-ink-muted transition-colors hover:border-critical/30 hover:text-ink lg:flex"
          >
            <Cpu size={11} className="text-good" aria-hidden="true" />
            <span className="max-w-[130px] truncate">{overview.model_name}</span>
            <span className="h-3 w-px bg-edge" aria-hidden="true" />
            <span className="font-bold tabular text-critical">
              {(overview.n_anomalies ?? 0).toLocaleString()}
            </span>
            <span className="sr-only">anomalies flagged</span>
          </button>
        )}

        <IconButton
          onClick={handleRefresh}
          disabled={busy !== null}
          aria-label="Refresh data"
          title="Refresh data"
          size="sm"
        >
          <RefreshCw
            size={14}
            className={cn(busy === "refresh" && "animate-spin")}
          />
        </IconButton>

        <IconButton
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          aria-pressed={theme === "dark"}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          size="sm"
        >
          {theme === "dark" ? (
            <Sun size={15} aria-hidden="true" />
          ) : (
            <Moon size={15} aria-hidden="true" />
          )}
        </IconButton>
      </div>
    </header>
  );
};

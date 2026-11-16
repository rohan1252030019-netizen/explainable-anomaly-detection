"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  Clock,
  Database,
  Download,
  FileJson,
  Moon,
  PanelLeft,
  RefreshCw,
  RotateCcw,
  Server,
  Settings,
  ShieldCheck,
  Sun,
  Table2,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api, type ExperimentRecord } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, SectionLabel } from "@/components/ui/Card";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { useToast } from "@/components/ui/Toast";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

const EXPORTS = [
  {
    href: api.exportPredictionsUrl,
    label: "Full predictions CSV",
    hint: "All records with scores, severity and ranks.",
    requirement: "detection" as const,
    icon: Table2,
  },
  {
    href: api.exportAnomaliesUrl,
    label: "Anomalies only CSV",
    hint: "Filtered outliers sorted by rank.",
    requirement: "detection" as const,
    icon: Database,
  },
  {
    href: api.exportExplanationsUrl,
    label: "Explanations JSON",
    hint: "Structured feature contribution attributions.",
    requirement: "explanations" as const,
    icon: FileJson,
  },
];

export const SettingsView: React.FC = () => {
  const {
    theme,
    toggleTheme,
    sidebarCollapsed,
    toggleSidebar,
    refreshAll,
    setActiveTab,
    health,
    dataset,
    overview,
  } = useApp();
  const { toast } = useToast();

  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyToken, setHistoryToken] = useState(0);
  const [explanationsReady, setExplanationsReady] = useState<boolean | null>(
    null
  );

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    let cancelled = false;

    api
      .getExperiments()
      .then((response) => {
        if (cancelled) return;
        setExperiments(response.experiments || []);
        setHistoryError(null);
        setHistoryLoading(false);
      })
      .catch((error) => {
        if (cancelled) return;
        setHistoryError(errorMessage(error));
        setHistoryLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [historyToken]);

  useEffect(() => {
    let cancelled = false;

    api
      .getGlobalImportance()
      .then((response) => {
        if (!cancelled) setExplanationsReady(response.ready);
      })
      .catch(() => {
        if (!cancelled) setExplanationsReady(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const reloadHistory = useCallback(() => {
    setHistoryLoading(true);
    setHistoryError(null);
    setHistoryToken((token) => token + 1);
  }, []);

  const handleReset = async () => {
    setResetting(true);

    try {
      await api.resetState();
      await refreshAll();
      setActiveTab("dataset");
      toast(
        "success",
        "Session reset",
        "Dataset, models and analysis results were cleared."
      );
      setConfirmOpen(false);
    } catch (error) {
      toast("error", "Reset failed", errorMessage(error));
    } finally {
      setResetting(false);
    }
  };

  const detectionReady = Boolean(overview?.ready);

  const exportClick = (
    event: React.MouseEvent<HTMLAnchorElement>,
    requirement: "detection" | "explanations"
  ) => {
    if (requirement === "detection" && !detectionReady) {
      event.preventDefault();
      toast(
        "info",
        "Nothing to export yet",
        "Run detection before downloading predictions or anomalies."
      );
    }
    if (requirement === "explanations" && explanationsReady !== true) {
      event.preventDefault();
      toast(
        "info",
        "No explanations to export",
        "Generate global explanations before downloading the attribution JSON."
      );
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-5">
      {/* Session snapshot */}
      <Card className="p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <span className="grid size-9 shrink-0 place-items-center rounded-lg border border-edge bg-inset text-accent">
              <Settings size={16} aria-hidden="true" />
            </span>
            <div>
              <SectionLabel>Current session</SectionLabel>
              <p className="mt-1 text-sm font-bold tracking-tight text-ink">
                {dataset?.loaded
                  ? dataset.dataset_name || "Untitled dataset"
                  : "No dataset loaded"}
              </p>
              <p className="mt-0.5 text-[11px] text-ink-muted">
                {detectionReady
                  ? `${overview?.model_name || "Detector"} · ${
                      overview?.n_anomalies?.toLocaleString() ?? "0"
                    } anomalies flagged`
                  : "Detection has not run in this session."}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-edge bg-inset px-2.5 py-1 text-[10px] font-semibold text-ink-muted">
              <Server size={11} aria-hidden="true" />
              {health?.status === "ok" || health?.status === "healthy"
                ? "Backend online"
                : health
                  ? `Backend ${health.status}`
                  : "Backend unchecked"}
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-edge bg-inset px-2.5 py-1 text-[10px] font-semibold text-ink-muted">
              <ShieldCheck size={11} aria-hidden="true" />
              {detectionReady ? "Results available" : "Awaiting detection"}
            </span>
          </div>
        </div>

        <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            ["Dataset rows", dataset?.n_rows?.toLocaleString() ?? "—"],
            ["Features", dataset?.n_preprocessed_features?.toLocaleString() ?? "—"],
            ["Framework", health?.framework || "—"],
            ["Version", health?.version || "—"],
          ].map(([label, value]) => (
            <div key={label} className="rounded-lg border border-edge bg-inset p-3">
              <dt className="text-[10px] uppercase tracking-wider text-ink-faint">
                {label}
              </dt>
              <dd className="mt-0.5 text-[11px] font-bold leading-tight text-ink" title={value}>
                {value}
              </dd>
            </div>
          ))}
        </dl>
      </Card>

      {/* Appearance */}
      <Card>
        <CardHeader
          title="Interface & appearance"
          description="Preferences are stored on this device and applied before the first paint."
          icon={Sun}
        />

        <div className="space-y-3 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-edge bg-inset p-4">
            <div className="min-w-0">
              <p className="text-xs font-semibold text-ink">Colour theme</p>
              <p className="mt-0.5 text-[11px] text-ink-muted">
                Both themes share the same semantic tokens, including severity
                colours and chart series.
              </p>
            </div>

            <div
              role="group"
              aria-label="Colour theme"
              className="inline-flex rounded-lg border border-edge bg-surface p-0.5"
            >
              {[
                { id: "light" as const, label: "Light", icon: Sun },
                { id: "dark" as const, label: "Dark", icon: Moon },
              ].map((option) => {
                const Icon = option.icon;
                const active = theme === option.id;
                return (
                  <button
                    key={option.id}
                    type="button"
                    aria-pressed={active}
                    onClick={() => {
                      if (!active) toggleTheme();
                    }}
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[11px] font-semibold transition-colors",
                      active
                        ? "bg-inset text-ink shadow-sm"
                        : "text-ink-muted hover:text-ink"
                    )}
                  >
                    <Icon size={12} aria-hidden="true" />
                    {option.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-edge bg-inset p-4">
            <div className="flex min-w-0 items-start gap-2.5">
              <PanelLeft
                size={15}
                className="mt-0.5 shrink-0 text-ink-faint"
                aria-hidden="true"
              />
              <div>
                <p className="text-xs font-semibold text-ink">
                  Collapsed navigation
                </p>
                <p className="mt-0.5 text-[11px] text-ink-muted">
                  Keep the sidebar as an icon rail. Shortcut: Ctrl/Cmd + Shift + B.
                </p>
              </div>
            </div>

            <button
              type="button"
              role="switch"
              aria-checked={sidebarCollapsed}
              aria-label="Collapse navigation"
              onClick={toggleSidebar}
              className={cn(
                "relative h-6 w-10 shrink-0 rounded-full border transition-colors",
                sidebarCollapsed
                  ? "border-accent bg-accent"
                  : "border-edge-strong bg-inset"
              )}
            >
              <span
                className={cn(
                  "absolute left-0.5 top-1 size-4 rounded-full border border-edge-strong bg-elevated shadow-sm transition-transform",
                  sidebarCollapsed ? "translate-x-5" : "translate-x-0"
                )}
                aria-hidden="true"
              />
            </button>
          </div>
        </div>
      </Card>

      {/* Exports */}
      <Card>
        <CardHeader
          title="Data & model artifact exports"
          description="Downloads are generated by the backend from the active session, not reconstructed from cached browser state."
          icon={Download}
        />

        <div className="p-5">
          <ul className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {EXPORTS.map((item) => {
              const Icon = item.icon;
              const unavailable =
                item.requirement === "detection"
                  ? !detectionReady
                  : explanationsReady !== true;
              return (
                <li key={item.label}>
                  <a
                    href={item.href}
                    target="_blank"
                    rel="noreferrer"
                    onClick={(event) => exportClick(event, item.requirement)}
                    aria-disabled={unavailable || undefined}
                    className={cn(
                      "panel-flush group flex h-full flex-col gap-1.5 p-4 transition-all duration-200",
                      unavailable
                        ? "opacity-65 hover:border-edge"
                        : "hover:border-accent-line hover:bg-accent-soft/40"
                    )}
                  >
                    <span className="flex items-start justify-between gap-2 text-xs font-semibold text-ink">
                      {item.label}
                      <Icon
                        size={14}
                        className="mt-0.5 shrink-0 text-accent"
                        aria-hidden="true"
                      />
                    </span>
                    <span className="text-[10px] leading-relaxed text-ink-faint">
                      {item.hint}
                    </span>
                    <span className="mt-auto pt-2 text-[9px] font-bold uppercase tracking-wider text-ink-faint">
                      {item.requirement === "explanations"
                        ? explanationsReady === null
                          ? "Checking availability…"
                          : explanationsReady
                            ? "Ready to download"
                            : "Run explainability first"
                        : unavailable
                          ? "Run detection first"
                          : "Ready to download"}
                    </span>
                  </a>
                </li>
              );
            })}
          </ul>
        </div>
      </Card>

      {/* Experiment history */}
      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge bg-elevated px-4 py-3">
          <div className="flex items-center gap-2 text-xs font-bold text-ink">
            <Clock size={14} className="text-accent" aria-hidden="true" />
            Session experiment history
            <span className="rounded-full border border-edge bg-inset px-2 py-0.5 text-[10px] font-semibold tabular text-ink-faint">
              {experiments.length} runs
            </span>
          </div>

          <Button
            variant="ghost"
            size="xs"
            onClick={reloadHistory}
            disabled={historyLoading}
            aria-label="Refresh experiment history"
          >
            <RefreshCw
              size={11}
              className={historyLoading ? "animate-spin" : undefined}
              aria-hidden="true"
            />
            Refresh
          </Button>
        </div>

        {historyError && (
          <div
            role="alert"
            className="flex flex-col gap-2 border-b border-critical/25 bg-critical-soft px-4 py-3 sm:flex-row sm:items-center"
          >
            <AlertTriangle
              size={13}
              className="shrink-0 text-critical"
              aria-hidden="true"
            />
            <p className="min-w-0 flex-1 break-words text-[11px] text-ink-muted">
              {historyError}
            </p>
            <Button variant="ghost" size="xs" onClick={reloadHistory}>
              Retry
            </Button>
          </div>
        )}

        <div className="overflow-x-auto" data-scroll>
          <table className="data-table text-[11px]">
            <caption className="sr-only">Detection runs in this session</caption>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Model</th>
                <th scope="col">Threshold</th>
                <th scope="col" className="text-right">
                  Anomalies
                </th>
                <th scope="col" className="text-right">
                  Rate
                </th>
                <th scope="col" className="text-right">
                  Fit time
                </th>
                <th scope="col">Timestamp</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {historyLoading ? (
                [0, 1, 2].map((row) => (
                  <tr key={row}>
                    <td colSpan={7}>
                      <div className="skeleton h-4 w-full" />
                    </td>
                  </tr>
                ))
              ) : experiments.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center">
                    <span className="font-sans text-xs font-semibold text-ink">
                      No experiments logged yet
                    </span>
                    <span className="mt-0.5 block font-sans text-[11px] text-ink-faint">
                      Each detection run is recorded here for the current session.
                    </span>
                  </td>
                </tr>
              ) : (
                experiments.map((experiment) => (
                  <tr key={experiment.id}>
                    <td className="text-ink-faint">#{experiment.id}</td>
                    <td className="font-sans font-bold text-accent">
                      {experiment.model}
                    </td>
                    <td className="text-ink-muted">
                      {experiment.threshold_method}
                    </td>
                    <td className="text-right font-bold tabular text-critical">
                      {experiment.n_anomalies.toLocaleString()}
                    </td>
                    <td className="text-right tabular text-ink-muted">
                      {((experiment.anomaly_rate || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="text-right tabular text-ink-muted">
                      {typeof experiment.fit_time_sec === "number"
                        ? `${experiment.fit_time_sec.toFixed(3)}s`
                        : "—"}
                    </td>
                    <td className="whitespace-nowrap text-[10px] text-ink-faint">
                      {experiment.timestamp}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Danger zone */}
      <Card className="border-critical/25 p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex min-w-0 items-start gap-3">
            <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-critical/25 bg-critical-soft text-critical">
              <RotateCcw size={14} aria-hidden="true" />
            </span>
            <div className="min-w-0">
              <h2 className="text-xs font-bold uppercase tracking-[0.14em] text-critical">
                Reset session state
              </h2>
              <p className="mt-1 max-w-xl text-xs leading-relaxed text-ink-muted">
                Clear the loaded dataset, preprocessed matrix, trained models,
                explanations and all analysis results. Exports and reports stop
                working until a new dataset is loaded.
              </p>
            </div>
          </div>

          <Button
            variant="danger"
            onClick={() => setConfirmOpen(true)}
            disabled={resetting}
          >
            <RotateCcw size={14} aria-hidden="true" />
            {resetting ? "Resetting…" : "Reset all state"}
          </Button>
        </div>
      </Card>

      <ConfirmDialog
        open={confirmOpen}
        title="Reset the entire session?"
        description={
          <>
            This clears the dataset, feature matrix, trained models,
            explanations, reports and experiment history for the current
            backend session. It cannot be undone.
          </>
        }
        confirmLabel="Reset everything"
        cancelLabel="Keep session"
        tone="danger"
        busy={resetting}
        onConfirm={() => void handleReset()}
        onClose={() => {
          if (!resetting) setConfirmOpen(false);
        }}
      />
    </div>
  );
};

export default SettingsView;

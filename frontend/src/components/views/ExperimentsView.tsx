"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Check,
  Clock,
  Copy,
  FileCode2,
  Fingerprint,
  History,
  RefreshCw,
  Search,
} from "lucide-react";
import { api, type PersistentExperimentRecord } from "@/lib/api";
import { Card, SectionLabel } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Feedback";
import { StatCard } from "@/components/ui/StatCard";
import { Badge } from "@/components/ui/Badge";
import { useToast } from "@/components/ui/Toast";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

export const ExperimentsView: React.FC = () => {
  const { toast } = useToast();

  const [loading, setLoading] = useState(true);
  const [experiments, setExperiments] = useState<PersistentExperimentRecord[]>([]);
  const [search, setSearch] = useState("");
  const [selectedExp, setSelectedExp] = useState<PersistentExperimentRecord | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // State is written only from promise callbacks: a synchronous setState inside
  // an effect body triggers a cascading render (react-hooks/set-state-in-effect),
  // and the cancellation flag stops a late response from writing into a stale
  // mount.
  useEffect(() => {
    let cancelled = false;
    api
      .getExperimentHistory()
      .then((res) => {
        if (!cancelled) setExperiments(res ?? []);
      })
      .catch((err) => {
        if (!cancelled) {
          toast(
            "error",
            "Failed to load experiment history",
            errorMessage(err)
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [toast]);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.getExperimentHistory();
      setExperiments(res ?? []);
    } catch (err) {
      toast(
        "error",
        "Failed to load experiment history",
        errorMessage(err)
      );
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
    toast(
      "info",
      "Copied to clipboard",
      text
    );
  };

  const filteredExperiments = experiments.filter((exp) => {
    if (!search) return true;
    const term = search.toLowerCase();
    return (
      exp.experiment_id?.toLowerCase().includes(term) ||
      exp.model_name?.toLowerCase().includes(term) ||
      exp.dataset_name?.toLowerCase().includes(term) ||
      exp.dataset_fingerprint?.toLowerCase().includes(term)
    );
  });

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-edge bg-surface p-4">
        <div>
          <h2 className="text-sm font-semibold text-ink flex items-center gap-2">
            <History size={16} className="text-accent" />
            Reproducible Experiment Registry
          </h2>
          <p className="text-xs text-ink-faint mt-0.5">
            Every detection run and benchmark is cryptographically fingerprinted with SHA-256 dataset hashes, hyperparameters, and execution audits.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-faint" />
            <input
              type="search"
              placeholder="Search experiments..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="rounded-lg border border-edge bg-canvas pl-8 pr-3 py-1.5 text-xs text-ink placeholder:text-ink-faint w-56"
            />
          </div>

          <Button
            variant="outline"
            onClick={fetchHistory}
            disabled={loading}
            className="gap-2"
          >
            {loading ? <Spinner /> : <RefreshCw size={13} />}
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total Tracked Runs"
          value={String(experiments.length)}
          sub="Persistent across sessions"
        />
        <StatCard
          label="Unique Models Evaluated"
          value={String(new Set(experiments.map((e) => e.model_name)).size)}
          sub="From single runs & ensembles"
        />
        <StatCard
          label="Integrity Verification"
          value="SHA-256 Fingerprint"
          sub="Dataset version hashing"
        />
        <StatCard
          label="Storage Format"
          value="Structured JSON Logs"
          sub="Audit trail in outputs/experiments"
        />
      </div>

      {/* Experiments Table */}
      {filteredExperiments.length > 0 ? (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-edge bg-canvas text-ink-faint">
                  <th className="py-2.5 px-4 font-semibold">Run ID</th>
                  <th className="py-2.5 px-4 font-semibold">Timestamp</th>
                  <th className="py-2.5 px-4 font-semibold">Model / Pipeline</th>
                  <th className="py-2.5 px-4 font-semibold">Dataset Fingerprint</th>
                  <th className="py-2.5 px-4 font-semibold">Anomalies</th>
                  <th className="py-2.5 px-4 font-semibold">Runtime</th>
                  <th className="py-2.5 px-4 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-edge">
                {filteredExperiments.map((exp) => {
                  const isCopied = copiedId === exp.experiment_id;
                  const isSelected = selectedExp?.experiment_id === exp.experiment_id;

                  return (
                    <tr
                      key={exp.experiment_id}
                      className={cn(
                        "transition-colors hover:bg-surface-elevated/40",
                        isSelected && "bg-accent/5"
                      )}
                    >
                      <td className="py-3 px-4 font-mono font-medium text-ink">
                        <span className="flex items-center gap-1.5">
                          {exp.experiment_id.slice(0, 8)}...
                          <button
                            type="button"
                            onClick={() => copyToClipboard(exp.experiment_id, exp.experiment_id)}
                            className="text-ink-faint hover:text-ink transition-colors"
                            title="Copy full experiment ID"
                          >
                            {isCopied ? <Check size={11} className="text-good" /> : <Copy size={11} />}
                          </button>
                        </span>
                      </td>
                      <td className="py-3 px-4 text-ink-muted">
                        <span className="flex items-center gap-1 text-[11px]">
                          <Clock size={11} className="text-ink-faint" />
                          {new Date(exp.timestamp).toLocaleString()}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className="font-semibold text-ink">{exp.model_name}</span>
                        <div className="text-[10px] text-ink-faint truncate max-w-xs font-mono">
                          {JSON.stringify(exp.parameters)}
                        </div>
                      </td>
                      <td className="py-3 px-4 font-mono text-ink-faint">
                        <span className="flex items-center gap-1" title={exp.dataset_fingerprint}>
                          <Fingerprint size={12} className="text-accent" />
                          {exp.dataset_fingerprint ? exp.dataset_fingerprint.slice(0, 10) : "N/A"}...
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <Badge tone={exp.results?.n_anomalies ? "warning" : "neutral"}>
                          {exp.results?.n_anomalies ?? 0} (
                          {((exp.results?.anomaly_fraction ?? 0) * 100).toFixed(1)}%)
                        </Badge>
                      </td>
                      <td className="py-3 px-4 font-mono tabular-nums text-ink-muted">
                        {typeof exp.results?.fit_time_sec === "number"
                          ? `${exp.results.fit_time_sec.toFixed(3)}s`
                          : "—"}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          type="button"
                          onClick={() => setSelectedExp(isSelected ? null : exp)}
                          className="text-xs font-medium text-accent hover:underline"
                        >
                          {isSelected ? "Close" : "Inspect"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      ) : !loading ? (
        <Card className="p-8 text-center">
          <History size={28} className="mx-auto text-ink-faint" />
          <h3 className="mt-2 text-sm font-semibold text-ink">No Recorded Experiments</h3>
          <p className="mt-1 text-xs text-ink-muted max-w-md mx-auto">
            Run a detection algorithm or benchmark to start tracking reproducible experiments with cryptographic dataset fingerprints.
          </p>
        </Card>
      ) : null}

      {/* Selected Experiment Inspector Detail Card */}
      {selectedExp && (
        <Card className="p-5 border-accent">
          <div className="flex items-center justify-between border-b border-edge pb-3">
            <div className="flex items-center gap-2">
              <FileCode2 size={16} className="text-accent" />
              <SectionLabel>Experiment Inspection: {selectedExp.experiment_id}</SectionLabel>
            </div>
            <button
              type="button"
              onClick={() => setSelectedExp(null)}
              className="text-xs text-ink-faint hover:text-ink"
            >
              ✕ Dismiss
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div>
              <h4 className="text-xs font-semibold text-ink mb-1">Hyperparameters & Config</h4>
              <pre className="p-3 rounded bg-canvas border border-edge text-[11px] font-mono overflow-x-auto text-ink">
                {JSON.stringify(selectedExp.parameters, null, 2)}
              </pre>
            </div>
            <div>
              <h4 className="text-xs font-semibold text-ink mb-1">Audit Results & Metrics</h4>
              <pre className="p-3 rounded bg-canvas border border-edge text-[11px] font-mono overflow-x-auto text-ink">
                {JSON.stringify(selectedExp.results, null, 2)}
              </pre>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

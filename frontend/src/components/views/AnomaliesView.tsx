"use client";

import React, { useState, useEffect } from "react";
import { useApp } from "@/context/AppContext";
import { api, AnomalyRecord } from "@/lib/api";
import { SeverityBadge } from "@/components/ui/Badge";
import {
  Search,
  Filter,
  Download,
  ChevronLeft,
  ChevronRight,
  Eye,
  Sparkles,
} from "lucide-react";

interface AnomaliesViewProps {
  onSelectAnomaly?: (rowId: number) => void;
}

export const AnomaliesView: React.FC<AnomaliesViewProps> = ({ onSelectAnomaly }) => {
  const { overview, setActiveTab } = useApp();

  const [search, setSearch] = useState("");
  const [severityFilter, setSeverityFilter] = useState<string[]>([
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
  ]);
  const [minScore, setMinScore] = useState(0.0);
  const [anomaliesOnly, setAnomaliesOnly] = useState(true);
  const [sortBy, setSortBy] = useState("rank");
  const [ascending, setAscending] = useState(true);
  const [page, setPage] = useState(1);
  const pageSize = 20;

  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<{
    total: number;
    page: number;
    total_pages: number;
    items: AnomalyRecord[];
  }>({ total: 0, page: 1, total_pages: 1, items: [] });

  const fetchAnomalies = async () => {
    if (!overview?.ready) return;
    setLoading(true);
    try {
      const res = await api.getAnomalies({
        search: search || undefined,
        severity: severityFilter,
        min_score: minScore,
        anomalies_only: anomaliesOnly,
        sort_by: sortBy,
        ascending,
        page,
        page_size: pageSize,
      });
      setData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnomalies();
  }, [search, severityFilter, minScore, anomaliesOnly, sortBy, ascending, page, overview?.ready]);

  const toggleSeverity = (sev: string) => {
    setSeverityFilter((prev) =>
      prev.includes(sev) ? prev.filter((s) => s !== sev) : [...prev, sev]
    );
    setPage(1);
  };

  const handleInspect = (item: AnomalyRecord, index: number) => {
    // Determine the row id
    const rowId = item.index !== undefined ? Number(item.index) : index;
    if (onSelectAnomaly) {
      onSelectAnomaly(rowId);
    } else {
      localStorage.setItem("xaud-selected-anomaly", String(rowId));
      setActiveTab("investigate");
    }
  };

  if (!overview?.ready) {
    return (
      <div className="saas-card p-12 text-center text-slate-400 space-y-3">
        <p className="text-sm">Anomaly detection has not been run yet.</p>
        <button
          onClick={() => setActiveTab("detection")}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg"
        >
          Go to Detection Workflow
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header & Filter Controls Bar */}
      <div className="saas-card p-4 space-y-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Search box */}
          <div className="relative w-full md:w-80">
            <Search size={15} className="absolute left-3 top-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search by ID or feature values..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              className="saas-input w-full pl-9 pr-3 py-1.5 text-xs"
            />
          </div>

          {/* Severity Badges Filter */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-slate-400 font-semibold flex items-center gap-1">
              <Filter size={13} /> Severity:
            </span>
            {["CRITICAL", "HIGH", "MEDIUM", "LOW"].map((sev) => {
              const active = severityFilter.includes(sev);
              return (
                <button
                  key={sev}
                  onClick={() => toggleSeverity(sev)}
                  className={`text-xs px-2.5 py-1 rounded-full font-semibold border transition-colors cursor-pointer ${
                    active
                      ? sev === "CRITICAL"
                        ? "bg-rose-500/20 text-rose-400 border-rose-500/40"
                        : sev === "HIGH"
                        ? "bg-orange-500/20 text-orange-400 border-orange-500/40"
                        : sev === "MEDIUM"
                        ? "bg-amber-500/20 text-amber-400 border-amber-500/40"
                        : "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
                      : "bg-slate-100 dark:bg-slate-800 text-slate-400 border-transparent opacity-50"
                  }`}
                >
                  {sev}
                </button>
              );
            })}
          </div>

          {/* Export Action */}
          <div className="flex items-center gap-2">
            <a
              href={api.exportAnomaliesUrl}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 text-xs font-semibold text-slate-700 dark:text-slate-300 transition-colors"
            >
              <Download size={13} />
              <span>Export CSV</span>
            </a>
          </div>
        </div>

        {/* Secondary Filter Row: Min score & Anomalies only */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-slate-100 dark:border-slate-800 text-xs text-slate-500 dark:text-slate-400">
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={anomaliesOnly}
                onChange={(e) => {
                  setAnomaliesOnly(e.target.checked);
                  setPage(1);
                }}
                className="w-3.5 h-3.5 accent-indigo-600 rounded cursor-pointer"
              />
              <span>Outliers Only ({overview.n_anomalies})</span>
            </label>

            <div className="flex items-center gap-2">
              <span>Min Score:</span>
              <input
                type="range"
                min="0"
                max="0.9"
                step="0.05"
                value={minScore}
                onChange={(e) => {
                  setMinScore(Number(e.target.value));
                  setPage(1);
                }}
                className="w-24 accent-indigo-600 cursor-pointer"
              />
              <span className="font-mono text-indigo-400">{minScore.toFixed(2)}</span>
            </div>
          </div>

          <div className="text-[11px] text-slate-400">
            Showing {data.items.length} of {data.total} records
          </div>
        </div>
      </div>

      {/* Main Anomalies Table */}
      <div className="saas-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/40 text-slate-400 uppercase text-[10px] font-bold">
                <th className="py-3 px-4">Rank</th>
                <th className="py-3 px-4">Anomaly Score</th>
                <th className="py-3 px-4">Severity Tier</th>
                <th className="py-3 px-4">Detector</th>
                <th className="py-3 px-4">Feature Samples</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono text-[11px]">
              {loading ? (
                <tr>
                  <td colSpan={6} className="py-12 text-center text-slate-400 font-sans">
                    Loading anomaly records...
                  </td>
                </tr>
              ) : data.items.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-12 text-center text-slate-400 font-sans">
                    No anomalies match the current filter criteria.
                  </td>
                </tr>
              ) : (
                data.items.map((item, idx) => {
                  const score = item.normalized_score ?? 0;
                  const rowId = item.index !== undefined ? Number(item.index) : idx;

                  // Extract 2 key numerical attributes to show
                  const ignoreKeys = [
                    "anomaly_score",
                    "normalized_score",
                    "prediction",
                    "is_anomaly",
                    "severity",
                    "rank",
                    "model",
                    "label",
                    "index",
                  ];
                  const previewEntries = Object.entries(item)
                    .filter(([k, v]) => !ignoreKeys.includes(k) && typeof v === "number")
                    .slice(0, 3);

                  return (
                    <tr
                      key={idx}
                      className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors group"
                    >
                      <td className="py-3 px-4 font-bold text-slate-500 font-sans">
                        #{item.rank !== null ? item.rank : "—"}
                      </td>
                      <td className="py-3 px-4">
                        <span className="font-bold text-slate-900 dark:text-slate-100 text-xs">
                          {score.toFixed(4)}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-sans">
                        <SeverityBadge severity={item.severity} size="sm" />
                      </td>
                      <td className="py-3 px-4 text-slate-400 font-sans text-xs">
                        {item.model || overview.model_name}
                      </td>
                      <td className="py-3 px-4 text-slate-500 text-[10px]">
                        <div className="flex items-center gap-3">
                          {previewEntries.map(([k, v]) => (
                            <span key={k} className="truncate max-w-[120px]">
                              <span className="text-slate-400 font-normal">{k}:</span>{" "}
                              <span className="font-semibold text-slate-700 dark:text-slate-300">
                                {Number(v).toFixed(2)}
                              </span>
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="py-3 px-4 text-right font-sans">
                        <button
                          onClick={() => handleInspect(item, rowId)}
                          className="px-2.5 py-1 bg-indigo-600/10 hover:bg-indigo-600 text-indigo-400 hover:text-white rounded text-xs font-semibold transition-all cursor-pointer inline-flex items-center gap-1"
                        >
                          <Eye size={12} />
                          <span>Investigate</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <div>
            Page <span className="font-semibold text-slate-200">{data.page}</span> of{" "}
            <span className="font-semibold text-slate-200">{data.total_pages}</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="p-1.5 rounded border border-slate-200 dark:border-slate-800 disabled:opacity-30 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
              disabled={page >= data.total_pages}
              className="p-1.5 rounded border border-slate-200 dark:border-slate-800 disabled:opacity-30 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

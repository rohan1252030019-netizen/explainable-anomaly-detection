"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  Search,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api, type AnomalyRecord } from "@/lib/api";
import { SeverityBadge, severityStyle } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, SectionLabel } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonRows } from "@/components/ui/Feedback";
import { errorMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";

interface AnomaliesViewProps {
  onSelectAnomaly?: (rowId: number) => void;
}

const SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const;
const PAGE_SIZE = 20;

const IGNORED_KEYS = [
  "anomaly_score",
  "normalized_score",
  "prediction",
  "is_anomaly",
  "severity",
  "rank",
  "model",
  "label",
  "index",
  "original_index",
];

const EMPTY_RESULT = {
  total: 0,
  page: 1,
  total_pages: 1,
  items: [] as AnomalyRecord[],
};

function rowIdOf(item: AnomalyRecord): number | null {
  const value = item.original_index;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return null;
}

export const AnomaliesView: React.FC<AnomaliesViewProps> = ({
  onSelectAnomaly,
}) => {
  const { dataset, overview, setActiveTab } = useApp();

  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [severityFilter, setSeverityFilter] = useState<string[]>([
    ...SEVERITIES,
  ]);
  const [minScore, setMinScore] = useState(0);
  const [anomaliesOnly, setAnomaliesOnly] = useState(true);
  const [sortBy, setSortBy] = useState("rank");
  const [ascending, setAscending] = useState(true);
  const [page, setPage] = useState(1);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [data, setData] = useState(EMPTY_RESULT);

  const ready = !!overview?.ready;

  /* Commit typing after a pause so every keystroke does not hit the API. */
  useEffect(() => {
    const next = searchInput.trim();
    if (next === search) return;
    const timer = setTimeout(() => {
      setLoading(true);
      setError(null);
      setSearch(next);
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [search, searchInput]);

  useEffect(() => {
    if (!ready) return;

    if (severityFilter.length === 0) return;

    const controller = new AbortController();

    api
      .getAnomalies({
        search: search || undefined,
        /* Normal rows have severity NORMAL, so a severity filter would hide
           them. Only apply the tier filter while viewing flagged records. */
        severity: anomaliesOnly ? severityFilter : undefined,
        min_score: minScore,
        anomalies_only: anomaliesOnly,
        sort_by: sortBy,
        ascending,
        page,
        page_size: PAGE_SIZE,
      })
      .then((res) => {
        if (controller.signal.aborted) return;
        setData(res);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(errorMessage(err));
        setData(EMPTY_RESULT);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [
    ready,
    search,
    severityFilter,
    minScore,
    anomaliesOnly,
    sortBy,
    ascending,
    page,
    reloadToken,
    overview?.model_name,
    overview?.n_total,
  ]);

  const applyQuery = useCallback(
    (patch: Partial<{
      severityFilter: string[];
      minScore: number;
      anomaliesOnly: boolean;
      sortBy: string;
      ascending: boolean;
      page: number;
    }>) => {
      setLoading(true);
      setError(null);
      setPage(patch.page !== undefined ? patch.page : 1);
      if (patch.severityFilter !== undefined)
        setSeverityFilter(patch.severityFilter);
      if (patch.minScore !== undefined) setMinScore(patch.minScore);
      if (patch.anomaliesOnly !== undefined)
        setAnomaliesOnly(patch.anomaliesOnly);
      if (patch.sortBy !== undefined) setSortBy(patch.sortBy);
      if (patch.ascending !== undefined) setAscending(patch.ascending);
    },
    []
  );

  const toggleSeverity = (severity: string) => {
    if (!anomaliesOnly) return;
    applyQuery({
      severityFilter: severityFilter.includes(severity)
        ? severityFilter.filter((item) => item !== severity)
        : [...severityFilter, severity],
    });
  };

  const clearFilters = () => {
    setSearchInput("");
    setSearch("");
    applyQuery({
      severityFilter: [...SEVERITIES],
      minScore: 0,
    });
  };

  const handleInspect = useCallback(
    (item: AnomalyRecord) => {
      const rowId = rowIdOf(item);
      if (rowId === null) return;
      if (onSelectAnomaly) {
        onSelectAnomaly(rowId);
      } else {
        localStorage.setItem("xaud-selected-anomaly", String(rowId));
        setActiveTab("investigate");
      }
    },
    [onSelectAnomaly, setActiveTab]
  );

  const activeFilters =
    severityFilter.length !== SEVERITIES.length ||
    minScore > 0 ||
    !!search ||
    !!searchInput;

  const severityCounts = overview?.severity_counts;
  const noSeveritySelected = severityFilter.length === 0;
  const displayLoading = loading && !noSeveritySelected;

  const rangePct = (minScore / 1) * 100;
  const firstRecord =
    data.total === 0 ? 0 : (data.page - 1) * PAGE_SIZE + 1;
  const lastRecord =
    data.total === 0 ? 0 : Math.min(data.page * PAGE_SIZE, data.total);

  if (!ready) {
    return (
      <EmptyState
        icon={AlertTriangle}
        title="No detection results yet"
        description={
          dataset?.loaded
            ? "The dataset is ready. Configure a detector and run detection to populate the anomaly queue."
            : "Load a dataset and run detection to search, rank and investigate flagged records."
        }
        actionText={dataset?.loaded ? "Configure detection" : "Open dataset"}
        onAction={() =>
          setActiveTab(dataset?.loaded ? "detection" : "dataset")
        }
      />
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter toolbar */}
      <Card className="p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <form
            role="search"
            onSubmit={(event) => {
              event.preventDefault();
              setLoading(true);
              setError(null);
              setSearch(searchInput.trim());
              setPage(1);
            }}
            className="relative w-full lg:max-w-sm"
          >
            <label htmlFor="anomaly-search" className="sr-only">
              Search anomalies
            </label>
            <Search
              size={15}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-faint"
              aria-hidden="true"
            />
            <input
              id="anomaly-search"
              type="search"
              placeholder="Search rows, IDs or feature values…"
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
              className="field pl-9 pr-9 [&::-webkit-search-cancel-button]:hidden"
            />
            {searchInput && (
              <button
                type="button"
                onClick={() => {
                  setLoading(true);
                  setError(null);
                  setSearchInput("");
                  setSearch("");
                  setPage(1);
                }}
                aria-label="Clear search"
                className="absolute right-1 top-1/2 grid size-7 -translate-y-1/2 place-items-center rounded-md text-ink-faint transition-colors hover:bg-inset hover:text-ink"
              >
                <X size={13} />
              </button>
            )}
          </form>

          <div className="flex flex-wrap items-center gap-2">
            <span className="hidden items-center gap-1.5 rounded-full border border-edge bg-inset px-2.5 py-1 text-[10px] font-semibold text-ink-muted xl:inline-flex">
              Detector · {overview?.model_name ?? "—"}
            </span>
            <a
              href={api.exportAnomaliesUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-edge bg-inset px-3 text-xs font-semibold text-ink transition-colors hover:border-edge-strong hover:bg-elevated"
            >
              <Download size={13} aria-hidden="true" />
              Export CSV
            </a>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-edge pt-3">
          <SlidersHorizontal
            size={13}
            className="text-ink-faint"
            aria-hidden="true"
          />
          <SectionLabel className="mr-1">Severity</SectionLabel>
          {SEVERITIES.map((severity) => {
            const active = severityFilter.includes(severity);
            const count = severityCounts?.[severity] ?? 0;
            return (
              <button
                key={severity}
                type="button"
                onClick={() => toggleSeverity(severity)}
                aria-pressed={active}
                disabled={!anomaliesOnly}
                title={
                  anomaliesOnly
                    ? `Toggle ${severity.toLowerCase()} records`
                    : "Severity tiers only apply to flagged records"
                }
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold transition-[opacity,border-color,background-color]",
                  active
                    ? severityStyle(severity).chip
                    : "border-edge bg-inset text-ink-faint opacity-55 hover:opacity-100",
                  !anomaliesOnly && "cursor-not-allowed opacity-40"
                )}
              >
                <span
                  className={cn(
                    "size-1.5 rounded-full",
                    active ? severityStyle(severity).dot : "bg-edge-strong"
                  )}
                  aria-hidden="true"
                />
                {severity}
                <span className="tabular opacity-70">{count}</span>
              </button>
            );
          })}
          {!anomaliesOnly && (
            <span className="text-[10px] text-ink-faint">
              Tier filters apply to flagged records only
            </span>
          )}
          {anomaliesOnly && severityFilter.length === 0 && (
            <span className="text-[10px] font-medium text-caution">
              Select at least one severity tier
            </span>
          )}
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-x-5 gap-y-3 border-t border-edge pt-3 text-xs text-ink-muted">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
            <label className="flex cursor-pointer select-none items-center gap-2">
              <input
                type="checkbox"
                checked={anomaliesOnly}
                onChange={(event) =>
                  applyQuery({ anomaliesOnly: event.target.checked })
                }
                className="size-3.5 rounded"
              />
              Flagged records only
              <span className="rounded bg-inset px-1.5 py-0.5 text-[10px] font-semibold tabular text-ink-faint">
                {overview?.n_anomalies ?? 0}
              </span>
            </label>

            <div className="flex items-center gap-2">
              <label htmlFor="min-score" className="whitespace-nowrap">
                Min score
              </label>
              <input
                id="min-score"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={minScore}
                aria-valuetext={`${minScore.toFixed(2)} minimum score`}
                style={{ ["--range-progress" as string]: `${rangePct}%` }}
                onChange={(event) =>
                  applyQuery({ minScore: Number(event.target.value) })
                }
                className="w-24"
              />
              <span className="w-8 font-mono text-xs font-bold tabular text-accent">
                {minScore.toFixed(2)}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <label htmlFor="sort-by" className="whitespace-nowrap">
                Sort
              </label>
              <select
                id="sort-by"
                value={sortBy}
                onChange={(event) =>
                  applyQuery({ sortBy: event.target.value })
                }
                className="field h-8 w-auto py-1"
              >
                <option value="rank">Rank</option>
                <option value="normalized_score">Score</option>
                <option value="severity">Severity</option>
                <option value="original_index">Row index</option>
              </select>
              <Button
                variant="ghost"
                size="xs"
                onClick={() => applyQuery({ ascending: !ascending })}
                aria-label={
                  ascending ? "Switch to descending order" : "Switch to ascending order"
                }
                title={ascending ? "Ascending" : "Descending"}
              >
                {ascending ? "ASC" : "DESC"}
              </Button>
            </div>

            {activeFilters && (
              <button
                type="button"
                onClick={clearFilters}
                className="inline-flex items-center gap-1 font-semibold text-ink-faint transition-colors hover:text-ink"
              >
                <X size={12} aria-hidden="true" />
                Clear filters
              </button>
            )}
          </div>

          <p aria-live="polite" className="tabular text-[11px] text-ink-faint">
            {displayLoading ? (
              "Updating results…"
            ) : (
              <>
                Showing {firstRecord.toLocaleString()}–
                {lastRecord.toLocaleString()} of{" "}
                {data.total.toLocaleString()}
              </>
            )}
          </p>
        </div>
      </Card>

      {error ? (
        <Card
          role="alert"
          className="flex flex-col items-center gap-3 p-10 text-center"
        >
          <span className="grid size-10 place-items-center rounded-xl border border-critical/30 bg-critical-soft text-critical">
            <AlertTriangle size={18} aria-hidden="true" />
          </span>
          <div>
            <p className="text-sm font-bold text-ink">
              Could not load anomalies
            </p>
            <p className="mx-auto mt-1 max-w-lg break-words text-xs leading-relaxed text-ink-muted">
              {error}
            </p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              setLoading(true);
              setError(null);
              setReloadToken((token) => token + 1);
            }}
          >
            Retry
          </Button>
        </Card>
      ) : (
        <Card className="overflow-hidden">
          <p
            data-print-hide
            className="border-b border-edge bg-inset px-4 py-2 text-[10px] leading-relaxed text-ink-faint sm:hidden"
          >
            Swipe the table sideways for detector, feature samples and the
            Investigate action.
          </p>
          <div
            className="max-h-[36rem] overflow-auto"
            data-scroll
            tabIndex={0}
            role="region"
            aria-label="Anomaly results table"
            aria-busy={displayLoading}
          >
            <table className="data-table min-w-[54rem] text-[11px]">
              <caption className="sr-only">
                Ranked records with row index, score, severity, detector and
                numeric feature samples
              </caption>
              <thead>
                <tr>
                  <th scope="col">Rank</th>
                  <th scope="col">Row</th>
                  <th scope="col">Score</th>
                  <th scope="col">Severity</th>
                  <th scope="col">Detector</th>
                  <th scope="col">Feature samples</th>
                  <th scope="col" className="text-right">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {displayLoading ? (
                  <tr>
                    <td colSpan={7} className="p-8">
                      <SkeletonRows rows={5} />
                    </td>
                  </tr>
                ) : noSeveritySelected ? (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-6 py-12 text-center font-sans text-ink-muted"
                    >
                      <p className="text-xs font-semibold text-ink">
                        No severity tier selected
                      </p>
                      <p className="mt-1 text-[11px] text-ink-faint">
                        Choose at least one tier above, or clear the filters to
                        restore all four.
                      </p>
                    </td>
                  </tr>
                ) : data.items.length === 0 ? (
                  <tr>
                    <td
                      colSpan={7}
                      className="px-6 py-12 text-center font-sans text-ink-muted"
                    >
                      <p className="text-xs font-semibold text-ink">
                        No records match these filters
                      </p>
                      <p className="mt-1 text-[11px] text-ink-faint">
                        Lower the minimum score, widen the severity tiers, or
                        clear the search.
                      </p>
                    </td>
                  </tr>
                ) : (
                  data.items.map((item, index) => {
                    const score = item.normalized_score ?? 0;
                    const rowId = rowIdOf(item);
                    const severity = severityStyle(item.severity);
                    const previewEntries = Object.entries(item)
                      .filter(
                        ([key, value]) =>
                          !IGNORED_KEYS.includes(key) &&
                          typeof value === "number" &&
                          Number.isFinite(value)
                      )
                      .slice(0, 3);

                    return (
                      <tr key={`${rowId ?? "row"}-${index}`}>
                        <td className="font-sans font-bold text-ink-faint">
                          {item.rank !== null && item.rank !== undefined
                            ? `#${item.rank}`
                            : "—"}
                        </td>
                        <td className="text-ink-faint">
                          {rowId !== null ? `#${rowId}` : "—"}
                        </td>
                        <td>
                          <span className="text-xs font-bold tabular text-ink">
                            {score.toFixed(4)}
                          </span>
                          <span
                            className="ml-2 inline-block h-1.5 w-14 overflow-hidden rounded-full bg-inset align-middle"
                            aria-hidden="true"
                          >
                            <span
                              className="block h-full rounded-full"
                              style={{
                                width: `${Math.min(100, Math.max(0, score * 100))}%`,
                                background: severity.hex,
                              }}
                            />
                          </span>
                        </td>
                        <td>
                          <SeverityBadge severity={item.severity} size="sm" />
                        </td>
                        <td className="text-ink-faint">
                          {item.model || overview?.model_name || "—"}
                        </td>
                        <td className="text-ink-faint">
                          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
                            {previewEntries.length > 0 ? (
                              previewEntries.map(([key, value]) => (
                                <span key={key} className="truncate">
                                  <span className="font-sans text-ink-faint">
                                    {key}:
                                  </span>{" "}
                                  <span className="font-semibold text-ink-muted">
                                    {Number(value).toLocaleString(undefined, {
                                      maximumFractionDigits: 2,
                                    })}
                                  </span>
                                </span>
                              ))
                            ) : (
                              <span className="font-sans">—</span>
                            )}
                          </span>
                        </td>
                        <td className="text-right">
                          <Button
                            size="xs"
                            variant="secondary"
                            onClick={() => handleInspect(item)}
                            disabled={rowId === null}
                            aria-label={
                              rowId !== null
                                ? `Investigate row ${rowId}`
                                : "Row index unavailable"
                            }
                          >
                            <Eye size={12} aria-hidden="true" />
                            Investigate
                          </Button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-edge px-4 py-3 text-xs text-ink-muted">
            <p className="tabular">
              Page <span className="font-bold text-ink">{data.page}</span> of{" "}
              <span className="font-bold text-ink">{data.total_pages}</span>
              <span className="ml-2 text-ink-faint">
                · {PAGE_SIZE} per page
              </span>
            </p>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  applyQuery({ page: Math.max(1, data.page - 1) })
                }
                disabled={displayLoading || data.page <= 1}
                aria-label="Previous page"
              >
                <ChevronLeft size={14} aria-hidden="true" />
                Previous
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  applyQuery({
                    page: Math.min(data.total_pages, data.page + 1),
                  })
                }
                disabled={displayLoading || data.page >= data.total_pages}
                aria-label="Next page"
              >
                Next
                <ChevronRight size={14} aria-hidden="true" />
              </Button>
            </div>
          </div>
        </Card>
      )}

      {data.total > 0 && !displayLoading && !noSeveritySelected && !error && (
        <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-[10px] text-ink-faint">
          <span>
            Rank 1 is the most anomalous record. Scores are normalised to this
            run, not shared across datasets.
          </span>
          <button
            type="button"
            onClick={() => setActiveTab("investigate")}
            className="inline-flex items-center gap-1 font-semibold text-ink-muted transition-colors hover:text-ink"
          >
            Open the investigation workspace
            <ArrowRight size={11} aria-hidden="true" />
          </button>
        </div>
      )}

    </div>
  );
};

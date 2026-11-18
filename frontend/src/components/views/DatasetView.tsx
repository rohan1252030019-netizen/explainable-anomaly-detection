"use client";

import React, { useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Columns3,
  Database,
  FileSpreadsheet,
  Layers,
  Play,
  Settings2,
  ShieldAlert,
  Sliders,
  Sparkles,
  Table2,
  Upload,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { Button } from "@/components/ui/Button";
import { Card, SectionLabel } from "@/components/ui/Card";
import { Tabs } from "@/components/ui/Tabs";
import { Spinner } from "@/components/ui/Feedback";
import { useToast } from "@/components/ui/Toast";
import { api, type DatasetProfileResponse } from "@/lib/api";
import { cn } from "@/lib/cn";
import { errorMessage } from "@/lib/errors";

type ExplorerTab = "preview" | "columns" | "profile" | "config";
type Busy = "upload" | "sample" | "preprocess" | null;

export const DatasetView: React.FC = () => {
  const { dataset, refreshAll, setActiveTab } = useApp();
  const { toast } = useToast();
  const [busy, setBusy] = useState<Busy>(null);
  const [dragging, setDragging] = useState(false);
  const [scalerType, setScalerType] = useState("standard");
  const [labelCol, setLabelCol] = useState<string>("");
  const [selectedTab, setSelectedTab] = useState<ExplorerTab>("preview");
  const [tableSearch, setTableSearch] = useState("");
  const [profile, setProfile] = useState<DatasetProfileResponse | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(false);

  // `fetchProfile` stays free of synchronous setState so the effect below never
  // cascades renders; the effect drives its own request through promise
  // callbacks with a cancellation flag, and the handler-driven refresh keeps
  // using `fetchProfile` (called from a real event).
  const fetchProfile = React.useCallback(async () => {
    if (!dataset?.loaded) return;
    try {
      const res = await api.getDatasetProfile();
      setProfile(res);
    } catch {
      // Ignored
    } finally {
      setLoadingProfile(false);
    }
  }, [dataset?.loaded]);

  React.useEffect(() => {
    if (selectedTab !== "profile" || profile || !dataset?.loaded) return;
    let cancelled = false;
    api
      .getDatasetProfile()
      .then((res) => {
        if (!cancelled) setProfile(res);
      })
      .catch(() => {
        // The profile tab offers a retry button if this fails.
      })
      .finally(() => {
        if (!cancelled) setLoadingProfile(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTab, profile, dataset?.loaded]);

  const handleFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      toast(
        "error",
        "Unsupported file type",
        "Upload a comma-separated .csv file."
      );
      return;
    }

    setBusy("upload");
    try {
      await api.uploadCsv(file);
      await refreshAll();
      setSelectedTab("preview");
      toast(
        "success",
        "Dataset uploaded",
        `${file.name} was profiled and is ready for preprocessing.`
      );
    } catch (err) {
      toast("error", "Upload failed", errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) await handleFile(file);
  };

  const handleDrop = async (event: React.DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    if (busy) return;
    const file = event.dataTransfer.files?.[0];
    if (file) await handleFile(file);
  };

  const handleLoadSample = async () => {
    setBusy("sample");
    try {
      await api.loadSample();
      await refreshAll();
      setSelectedTab("preview");
      toast("success", "Benchmark dataset loaded");
    } catch (err) {
      toast("error", "Could not load the sample", errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const handleApplyPreprocessing = async () => {
    setBusy("preprocess");
    try {
      await api.preprocess({
        scaler_type: scalerType,
        label_col: labelCol || null,
      });
      await refreshAll();
      toast(
        "success",
        "Feature matrix rebuilt",
        "The preprocessing configuration was applied to the current dataset."
      );
    } catch (err) {
      toast("error", "Preprocessing failed", errorMessage(err));
    } finally {
      setBusy(null);
    }
  };

  const filteredPreview = (dataset?.preview || []).filter((row) => {
    if (!tableSearch) return true;
    const term = tableSearch.toLowerCase();
    return Object.values(row).some((value) =>
      String(value).toLowerCase().includes(term)
    );
  });

  const previewColumns = Object.keys(dataset?.preview?.[0] || {});
  const maxMissingPct = Math.max(
    1,
    ...(dataset?.columns?.map((column) => column.null_pct) ?? [1])
  );

  return (
    <div className="space-y-5">
      {/* Sources */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card className="flex flex-col p-5">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="flex items-center gap-2 text-sm font-bold text-foreground">
                <Upload size={15} className="text-primary" aria-hidden="true" />
                Upload a dataset
              </h2>
              <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                Comma-separated tabular data. Column profiling runs immediately;
                categorical encoding, imputation and scaling are handled during
                preprocessing.
              </p>
            </div>
          </div>

          <label
            htmlFor="csv-file-input"
            onDragOver={(event) => {
              event.preventDefault();
              if (!busy) setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className={cn(
              "mt-4 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-5 py-7 text-center transition-colors",
              dragging
                ? "border-primary bg-primary-soft"
                : "border-border-strong bg-surface-muted/60 hover:border-primary hover:bg-primary-soft",
              busy && "pointer-events-none opacity-60"
            )}
          >
            <span
              className={cn(
                "grid size-10 place-items-center rounded-lg border",
                dragging
                  ? "border-primary bg-primary-soft text-primary"
                  : "border-border bg-surface text-subtle-foreground"
              )}
            >
              {busy === "upload" ? (
                <Spinner className="text-primary" />
              ) : (
                <FileSpreadsheet size={18} aria-hidden="true" />
              )}
            </span>
            <span className="text-xs font-semibold text-foreground">
              {busy === "upload"
                ? "Uploading and profiling…"
                : dragging
                  ? "Drop the CSV to upload"
                  : "Drop a CSV here, or browse"}
            </span>
            <span className="text-[10px] text-subtle-foreground">
              CSV only · the current analysis is replaced on upload
            </span>
            <input
              type="file"
              accept=".csv,text/csv"
              id="csv-file-input"
              onChange={handleFileUpload}
              disabled={busy !== null}
              className="sr-only"
            />
          </label>
        </Card>

        <Card className="flex flex-col p-5">
          <h2 className="flex items-center gap-2 text-sm font-bold text-foreground">
            <Database size={15} className="text-success" aria-hidden="true" />
            Benchmark sample
          </h2>
          <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
            A deterministic sample for exploring the full pipeline without
            preparing data.
          </p>

          <div className="panel-inset mt-4 flex flex-1 flex-col justify-between gap-4 p-4">
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-mono text-xs font-semibold text-foreground">
                  sample_anomaly_data.csv
                </span>
                <span className="rounded border border-border bg-surface px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-subtle-foreground">
                  Included
                </span>
              </div>
              <ul className="mt-2.5 space-y-1 text-[11px] text-muted-foreground">
                <li>• 880 records (800 normal, 80 synthetic outliers)</li>
                <li>• 8 numerical attributes + 2 categorical tags</li>
                <li>• Ground-truth label reserved for post-hoc evaluation</li>
              </ul>
            </div>
            <Button
              variant="secondary"
              className="w-full"
              onClick={handleLoadSample}
              disabled={busy !== null}
            >
              {busy === "sample" ? (
                <Spinner />
              ) : (
                <Database size={14} aria-hidden="true" />
              )}
              {busy === "sample"
                ? "Loading sample…"
                : dataset?.loaded
                  ? "Reload benchmark sample"
                  : "Load benchmark sample"}
            </Button>
          </div>
        </Card>
      </div>

      {!dataset?.loaded && (
        <p className="text-center text-[11px] text-subtle-foreground">
          No dataset loaded. Upload a file or use the sample to unlock profiling,
          detection and reporting.
        </p>
      )}

      {dataset?.loaded && (
        <>
          {/* Dataset summary */}
          <Card className="p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="truncate text-sm font-bold tracking-tight text-foreground">
                    {dataset.dataset_name}
                  </h2>
                  <span className="inline-flex items-center gap-1 rounded-full border border-success-line bg-success-soft px-2 py-0.5 text-[9.5px] font-semibold text-success">
                    <CheckCircle2 size={10} aria-hidden="true" />
                    Loaded
                  </span>
                  <span
                    className={cn(
                      "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9.5px] font-semibold",
                      dataset.feature_matrix_ready
                        ? "border-primary-line bg-primary-soft text-primary"
                        : "border-border bg-surface-muted text-subtle-foreground"
                    )}
                  >
                    <Layers size={10} aria-hidden="true" />
                    {dataset.feature_matrix_ready
                      ? "Feature matrix ready"
                      : "Feature matrix pending"}
                  </span>
                </div>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  {dataset.n_rows?.toLocaleString()} rows · {dataset.n_cols}{" "}
                  columns
                  {dataset.memory_mb !== undefined
                    ? ` · ${dataset.memory_mb.toFixed(1)} MB in memory`
                    : ""}
                  {dataset.label_col
                    ? ` · label column: ${dataset.label_col}`
                    : " · no label column (purely unsupervised)"}
                </p>
              </div>
              <Button
                size="sm"
                onClick={() => setActiveTab("detection")}
                className="shrink-0"
              >
                <Play size={13} aria-hidden="true" />
                Continue to detection
              </Button>
            </div>

            <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-border pt-4 sm:grid-cols-4 lg:grid-cols-6">
              {[
                ["Rows", dataset.n_rows?.toLocaleString() ?? "0"],
                ["Columns", dataset.n_cols ?? 0],
                ["Numerical", dataset.numerical_cols?.length ?? 0],
                ["Categorical", dataset.categorical_cols?.length ?? 0],
                [
                  "Missing cells",
                  dataset.total_missing?.toLocaleString() ?? "0",
                  (dataset.total_missing ?? 0) > 0,
                ],
                ["Modelling features", dataset.n_preprocessed_features ?? 0],
              ].map(([label, value, warning]) => (
                <div key={String(label)}>
                  <dt className="text-[9.5px] font-semibold uppercase tracking-[0.1em] text-subtle-foreground">
                    {label}
                  </dt>
                  <dd
                    className={cn(
                      "mt-0.5 text-sm font-bold tabular",
                      warning ? "text-caution" : "text-foreground"
                    )}
                  >
                    {value}
                  </dd>
                </div>
              ))}
            </dl>
          </Card>

          {dataset.warnings && dataset.warnings.length > 0 && (
            <div
              role="alert"
              className="flex items-start gap-3 rounded-xl border border-caution-line bg-caution-soft p-4 text-xs"
            >
              <AlertCircle
                size={16}
                className="mt-0.5 shrink-0 text-caution"
                aria-hidden="true"
              />
              <div>
                <p className="font-bold text-foreground">
                  Dataset quality warnings ({dataset.warnings.length})
                </p>
                <ul className="mt-1 list-inside list-disc space-y-0.5 text-[11px] text-muted-foreground">
                  {dataset.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          <Card className="overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border">
              <Tabs
                value={selectedTab}
                onChange={(tab) => {
                  if (tab === "profile" && !profile && dataset?.loaded) {
                    setLoadingProfile(true);
                  }
                  setSelectedTab(tab);
                }}
                ariaLabel="Dataset explorer"
                panelId="dataset-explorer"
                tabs={[
                  { id: "preview", label: "Data preview" },
                  {
                    id: "columns",
                    label: "Column profile",
                    count: dataset.columns?.length || 0,
                  },
                  { id: "profile", label: "Health & Profiler" },
                  { id: "config", label: "Preprocessing" },
                ]}
                className="border-b-0"
              />
              {selectedTab === "preview" && (
                <div className="flex items-center gap-2 px-3">
                  <Table2 size={13} className="text-subtle-foreground" aria-hidden="true" />
                  <label htmlFor="preview-filter" className="sr-only">
                    Filter preview rows
                  </label>
                  <input
                    id="preview-filter"
                    type="search"
                    placeholder="Filter rows…"
                    value={tableSearch}
                    onChange={(event) => setTableSearch(event.target.value)}
                    className="field h-8 w-40 py-1"
                  />
                </div>
              )}
            </div>

            <div
              id={`dataset-explorer-${selectedTab}`}
              role="tabpanel"
              aria-labelledby={`tab-dataset-explorer-${selectedTab}`}
              className="p-4 sm:p-5"
            >
              {selectedTab === "preview" && (
                <>
                  <div className="data-scroll max-h-[26rem] overflow-auto rounded-lg border border-border">
                    <table className="data-table text-[11px]">
                      <caption className="sr-only">
                        Preview of the first {filteredPreview.length} rows of the
                        loaded dataset
                      </caption>
                      <thead>
                        <tr>
                          <th scope="col" className="w-10">
                            #
                          </th>
                          {previewColumns.map((column) => (
                            <th key={column} scope="col">
                              {column}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="font-mono">
                        {filteredPreview.map((row, rowIndex) => (
                          <tr key={rowIndex}>
                            <td className="text-subtle-foreground">{rowIndex + 1}</td>
                            {previewColumns.map((column) => (
                              <td
                                key={column}
                                className="max-w-[180px] truncate text-foreground"
                                title={
                                  row[column] !== null
                                    ? String(row[column])
                                    : undefined
                                }
                              >
                                {row[column] !== null ? String(row[column]) : "—"}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {filteredPreview.length === 0 && (
                      <p className="p-8 text-center text-xs text-subtle-foreground">
                        No preview rows match “{tableSearch}”.
                      </p>
                    )}
                  </div>
                  <p className="mt-2 text-[10px] text-subtle-foreground">
                    Showing {filteredPreview.length} of{" "}
                    {dataset.preview?.length ?? 0} preview rows. The backend
                    returns a bounded sample, not the full table.
                  </p>
                </>
              )}

              {selectedTab === "columns" && (
                <div className="data-scroll max-h-[28rem] overflow-auto rounded-lg border border-border">
                  <table className="data-table text-[11px]">
                    <caption className="sr-only">Column-level profiling</caption>
                    <thead>
                      <tr>
                        <th scope="col">Column</th>
                        <th scope="col">Type</th>
                        <th scope="col">Non-null</th>
                        <th scope="col">Missing</th>
                        <th scope="col">Unique</th>
                        <th scope="col">Kind</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(dataset.columns || []).map((column) => (
                        <tr key={column.name}>
                          <th
                            scope="row"
                            className="font-semibold text-foreground"
                            title={column.name}
                          >
                            {column.name}
                          </th>
                          <td className="font-mono text-muted-foreground">
                            {column.dtype}
                          </td>
                          <td className="tabular text-muted-foreground">
                            {column.non_null_count?.toLocaleString()}
                          </td>
                          <td>
                            <span className="flex items-center gap-2">
                              <span className="w-9 text-right font-bold tabular text-muted-foreground">
                                {column.null_pct?.toFixed(1)}%
                              </span>
                              <span
                                className="h-1 w-12 overflow-hidden rounded-full bg-surface-muted"
                                aria-hidden="true"
                              >
                                <span
                                  className="block h-full rounded-full bg-caution"
                                  style={{
                                    width: `${
                                      (column.null_pct / maxMissingPct) * 100
                                    }%`,
                                  }}
                                />
                              </span>
                            </span>
                          </td>
                          <td className="tabular text-muted-foreground">
                            {column.unique_count?.toLocaleString()}
                          </td>
                          <td>
                            <span
                              className={cn(
                                "rounded border px-1.5 py-0.5 text-[9.5px] font-semibold",
                                column.is_numeric
                                  ? "border-info-line bg-info-soft text-info"
                                  : "border-border bg-surface-muted text-subtle-foreground"
                              )}
                            >
                              {column.is_numeric ? "Numeric" : "Categorical"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

                            {selectedTab === "profile" && (
                <div className="space-y-6">
                  {loadingProfile && (
                    <div className="flex items-center justify-center p-8 gap-2 text-muted-foreground text-xs">
                      <Spinner /> Profiling dataset schema, collinearity, and health score...
                    </div>
                  )}

                  {!loadingProfile && !profile && (
                    <div className="p-8 text-center">
                      <Button onClick={() => { setLoadingProfile(true); void fetchProfile(); }} className="gap-2"><Sparkles size={13} />
                        Generate Full Dataset Intelligence Profile
                      </Button>
                    </div>
                  )}

                  {profile && (
                    <div className="space-y-6">
                      {/* Health score card */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="rounded-xl border border-border bg-surface p-4 flex items-center gap-4">
                          <div className={cn(
                            "flex h-16 w-16 shrink-0 items-center justify-center rounded-full border-4 font-mono text-xl font-extrabold",
                            profile.health.score >= 80 ? "border-success text-success bg-success/5" :
                            profile.health.score >= 60 ? "border-caution text-caution bg-caution/5" :
                            "border-danger text-danger bg-danger/5"
                          )}>
                            {Math.round(profile.health.score)}
                          </div>
                          <div>
                            <div className="text-[10px] uppercase font-bold tracking-wider text-subtle-foreground">
                              Dataset Health Score
                            </div>
                            <div className="text-sm font-semibold text-foreground mt-0.5">
                              {profile.health.interpretation}
                            </div>
                            <div className="text-[11px] text-muted-foreground mt-1">
                              0–100 itemized empirical penalty rubric
                            </div>
                          </div>
                        </div>

                        <div className="rounded-xl border border-border bg-surface p-4">
                          <div className="text-[10px] uppercase font-bold tracking-wider text-subtle-foreground">
                            Collinearity & Condition Number
                          </div>
                          <div className="font-mono text-lg font-bold text-foreground mt-1">
                            κ = {profile.collinearity.condition_number.toFixed(2)}
                          </div>
                          <div className="text-[11px] text-muted-foreground mt-1">
                            {profile.collinearity.severely_collinear ? (
                              <span className="text-danger font-semibold flex items-center gap-1">
                                <AlertTriangle size={12} /> Severely collinear (condition &gt; 30)
                              </span>
                            ) : (
                              <span className="text-success font-semibold flex items-center gap-1">
                                <CheckCircle2 size={12} /> Well-conditioned feature matrix
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="rounded-xl border border-border bg-surface p-4">
                          <div className="text-[10px] uppercase font-bold tracking-wider text-subtle-foreground">
                            Data Completeness & Uniqueness
                          </div>
                          <div className="text-lg font-bold text-foreground mt-1">
                            {((1 - profile.quality.missing_pct) * 100).toFixed(1)}% complete
                          </div>
                          <div className="text-[11px] text-muted-foreground mt-1">
                            {profile.quality.duplicate_rows} duplicate rows ({profile.quality.duplicate_pct.toFixed(1)}%)
                          </div>
                        </div>
                      </div>

                      {/* Deductions if any */}
                      {profile.health.deductions && profile.health.deductions.length > 0 && (
                        <div className="rounded-lg border border-caution-line bg-caution/5 p-3 text-xs text-caution">
                          <div className="font-semibold flex items-center gap-1.5 mb-1">
                            <ShieldAlert size={14} /> Health Score Deductions:
                          </div>
                          <ul className="list-disc pl-5 space-y-0.5 text-[11px]">
                            {profile.health.deductions.map((d, i) => (
                              <li key={i}>{d}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Schema Discovery */}
                      <div className="rounded-xl border border-border bg-surface p-4 space-y-3">
                        <SectionLabel>Inferred Column Schema & Type Classification</SectionLabel>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                          <div className="p-3 rounded-lg border border-border bg-background">
                            <div className="text-xs font-semibold text-foreground flex items-center justify-between">
                              <span>Numerical Features</span>
                              <span className="font-mono text-primary">{profile.schema.numerical_columns.length}</span>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                              {profile.schema.numerical_columns.map(c => (
                                <span key={c} className="rounded bg-surface px-1.5 py-0.5 text-[10px] font-mono text-foreground">
                                  {c}
                                </span>
                              ))}
                            </div>
                          </div>

                          <div className="p-3 rounded-lg border border-border bg-background">
                            <div className="text-xs font-semibold text-foreground flex items-center justify-between">
                              <span>Categorical Features</span>
                              <span className="font-mono text-info">{profile.schema.categorical_columns.length}</span>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                              {profile.schema.categorical_columns.length > 0 ? (
                                profile.schema.categorical_columns.map(c => (
                                  <span key={c} className="rounded bg-surface px-1.5 py-0.5 text-[10px] font-mono text-foreground">
                                    {c}
                                  </span>
                                ))
                              ) : (
                                <span className="text-[10px] text-subtle-foreground">None</span>
                              )}
                            </div>
                          </div>

                          <div className="p-3 rounded-lg border border-border bg-background">
                            <div className="text-xs font-semibold text-foreground flex items-center justify-between">
                              <span>ID / High-Cardinality Candidates</span>
                              <span className="font-mono text-caution">{profile.schema.id_candidates.length}</span>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                              {profile.schema.id_candidates.length > 0 ? (
                                profile.schema.id_candidates.map(c => (
                                  <span key={c} className="rounded bg-caution/10 border border-caution-line px-1.5 py-0.5 text-[10px] font-mono text-caution">
                                    {c} (exclude)
                                  </span>
                                ))
                              ) : (
                                <span className="text-[10px] text-subtle-foreground">None detected</span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Collinear Pairs */}
                      {profile.collinearity.high_correlations && profile.collinearity.high_correlations.length > 0 && (
                        <div className="rounded-xl border border-border bg-surface p-4 space-y-2">
                          <SectionLabel>Highly Correlated Feature Pairs (|r| &gt; 0.85)</SectionLabel>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
                            {profile.collinearity.high_correlations.map((pair, idx) => (
                              <div key={idx} className="flex items-center justify-between rounded border border-border bg-background px-3 py-1.5 text-xs">
                                <span className="font-mono text-foreground">
                                  {pair.feature_a} <span className="text-subtle-foreground">↔</span> {pair.feature_b}
                                </span>
                                <span className="font-mono font-bold text-caution tabular-nums">
                                  r = {pair.pearson_r.toFixed(3)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Recommendations */}
                      {profile.recommendations && profile.recommendations.length > 0 && (
                        <div className="rounded-xl border border-border bg-surface p-4 space-y-2">
                          <div className="flex items-center gap-2">
                            <Sparkles size={14} className="text-primary" />
                            <SectionLabel>Profiler Modeling Recommendations</SectionLabel>
                          </div>
                          <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-1">
                            {profile.recommendations.map((rec, i) => (
                              <li key={i}>{rec}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {selectedTab === "config" && (
                <div className="max-w-xl space-y-5">
                  <div>
                    <label
                      htmlFor="scaler-type"
                      className="mb-1.5 block text-[11px] font-semibold text-foreground"
                    >
                      <SectionLabel className="inline-flex items-center gap-1.5">
                        <Settings2 size={11} aria-hidden="true" />
                        Feature scaling
                      </SectionLabel>
                    </label>
                    <select
                      id="scaler-type"
                      value={scalerType}
                      onChange={(event) => setScalerType(event.target.value)}
                      className="field"
                    >
                      <option value="standard">
                        StandardScaler — zero mean, unit variance
                      </option>
                      <option value="robust">
                        RobustScaler — median and IQR, outlier resistant
                      </option>
                    </select>
                    <p className="mt-1.5 text-[10.5px] leading-relaxed text-muted-foreground">
                      RobustScaler is preferable when extreme values would
                      otherwise dominate the empirical variance.
                    </p>
                  </div>

                  <div>
                    <label
                      htmlFor="label-col"
                      className="mb-1.5 block text-[11px] font-semibold text-foreground"
                    >
                      <SectionLabel className="inline-flex items-center gap-1.5">
                        <Columns3 size={11} aria-hidden="true" />
                        Ground-truth label column (optional)
                      </SectionLabel>
                    </label>
                    <select
                      id="label-col"
                      value={labelCol}
                      onChange={(event) => setLabelCol(event.target.value)}
                      className="field"
                    >
                      <option value="">None — purely unsupervised</option>
                      {dataset.columns?.map((column) => (
                        <option key={column.name} value={column.name}>
                          {column.name}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1.5 text-[10.5px] leading-relaxed text-muted-foreground">
                      Labels are never used for fitting. They enable post-hoc
                      ROC-AUC, F1 and held-out evaluation only.
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-3 border-t border-border pt-4">
                    <Button
                      onClick={handleApplyPreprocessing}
                      disabled={busy !== null}
                    >
                      {busy === "preprocess" ? (
                        <Spinner />
                      ) : (
                        <Sliders size={14} aria-hidden="true" />
                      )}
                      {busy === "preprocess"
                        ? "Rebuilding features…"
                        : "Apply and rebuild feature matrix"}
                    </Button>
                    <p className="text-[10px] text-subtle-foreground">
                      Rebuilding replaces the current feature matrix.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </Card>

          <div className="flex justify-end">
            <Button size="lg" onClick={() => setActiveTab("detection")}>
              <Play size={14} aria-hidden="true" />
              Continue to anomaly detection
            </Button>
          </div>
        </>
      )}
    </div>
  );
};

"use client";

import React, { useState } from "react";
import { useApp } from "@/context/AppContext";
import { StatCard } from "@/components/ui/StatCard";
import { api } from "@/lib/api";
import {
  Upload,
  Database,
  FileSpreadsheet,
  AlertCircle,
  CheckCircle2,
  Sliders,
  Play,
  RotateCcw,
} from "lucide-react";

export const DatasetView: React.FC = () => {
  const { dataset, refreshAll, setActiveTab } = useApp();
  const [isUploading, setIsUploading] = useState(false);
  const [scalerType, setScalerType] = useState("standard");
  const [labelCol, setLabelCol] = useState<string>("");
  const [selectedTab, setSelectedTab] = useState<"preview" | "columns" | "config">("preview");
  const [tableSearch, setTableSearch] = useState("");

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      await api.uploadCsv(file);
      await refreshAll();
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadSample = async () => {
    setIsUploading(true);
    try {
      await api.loadSample();
      await refreshAll();
    } catch (err: any) {
      alert(`Failed to load sample: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleApplyPreprocessing = async () => {
    try {
      await api.preprocess({
        scaler_type: scalerType,
        label_col: labelCol || null,
      });
      await refreshAll();
      alert("Preprocessing reconfigured and feature matrix rebuilt.");
    } catch (err: any) {
      alert(`Preprocessing failed: ${err.message}`);
    }
  };

  const filteredPreview = (dataset?.preview || []).filter((row) => {
    if (!tableSearch) return true;
    const term = tableSearch.toLowerCase();
    return Object.values(row).some((val) =>
      String(val).toLowerCase().includes(term)
    );
  });

  return (
    <div className="space-y-6">
      {/* Upload and Source Card */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upload Dropzone */}
        <div className="saas-card p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Upload size={16} className="text-indigo-500" />
              <span>Upload Custom Dataset</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Upload any tabular CSV file. Preprocessing will automatically handle categorical encoding, imputation, and feature scaling.
            </p>
          </div>

          <div className="mt-5 border-2 border-dashed border-slate-200 dark:border-slate-800 hover:border-indigo-500/50 rounded-xl p-6 text-center transition-colors">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              disabled={isUploading}
              className="hidden"
              id="csv-file-input"
            />
            <label
              htmlFor="csv-file-input"
              className="cursor-pointer flex flex-col items-center justify-center space-y-2"
            >
              <div className="w-10 h-10 rounded-full bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
                <FileSpreadsheet size={20} />
              </div>
              <span className="text-xs font-semibold text-slate-700 dark:text-slate-200">
                {isUploading ? "Uploading & Preprocessing..." : "Click to browse or drop CSV"}
              </span>
              <span className="text-[10px] text-slate-400">
                Supports numeric & categorical tabular datasets
              </span>
            </label>
          </div>
        </div>

        {/* Benchmark Sample Card */}
        <div className="saas-card p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Database size={16} className="text-emerald-500" />
              <span>Benchmark Sample Dataset</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Load our benchmark anomaly dataset with 880 observations (800 normal records + 80 subtle synthetic outliers across 10 numerical/behavioral attributes).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-950/40 border border-slate-200 dark:border-slate-800 space-y-3 mt-4">
            <div className="text-xs font-semibold text-slate-800 dark:text-slate-200">
              sample_anomaly_data.csv
            </div>
            <div className="text-[11px] text-slate-400 space-y-1">
              <div>• 880 instances (800 normal, 80 anomalies)</div>
              <div>• 8 numerical attributes + 2 categorical tags</div>
              <div>• Pre-configured ground-truth label for evaluation</div>
            </div>
            <button
              onClick={handleLoadSample}
              disabled={isUploading}
              className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg shadow-sm transition-all cursor-pointer flex items-center justify-center gap-2"
            >
              <Database size={14} />
              <span>{isUploading ? "Loading..." : "Load Benchmark Dataset"}</span>
            </button>
          </div>
        </div>
      </div>

      {dataset?.loaded && (
        <>
          {/* Dataset Profile KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-6 gap-4">
            <StatCard label="Total Rows" value={dataset.n_rows?.toLocaleString() || 0} />
            <StatCard label="Total Columns" value={dataset.n_cols || 0} />
            <StatCard label="Numerical Cols" value={dataset.numerical_cols?.length || 0} />
            <StatCard label="Categorical Cols" value={dataset.categorical_cols?.length || 0} />
            <StatCard
              label="Missing Values"
              value={dataset.total_missing || 0}
              accentColor={dataset.total_missing && dataset.total_missing > 0 ? "#F59E0B" : undefined}
            />
            <StatCard label="Preprocessed X" value={`${dataset.n_preprocessed_features || 0} feats`} />
          </div>

          {/* Validation Warnings */}
          {dataset.warnings && dataset.warnings.length > 0 && (
            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs text-amber-400 flex items-start gap-3">
              <AlertCircle size={16} className="shrink-0 mt-0.5" />
              <div>
                <span className="font-bold">Dataset Quality Warnings:</span>
                <ul className="list-disc list-inside mt-1 space-y-0.5 text-slate-300">
                  {dataset.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Explorer Tabs: Preview, Column Metadata, Preprocessing Config */}
          <div className="saas-card overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 px-5 pt-3 bg-slate-50 dark:bg-slate-950/20">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setSelectedTab("preview")}
                  className={`pb-3 text-xs font-semibold border-b-2 cursor-pointer transition-colors ${
                    selectedTab === "preview"
                      ? "border-indigo-500 text-indigo-400"
                      : "border-transparent text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Raw Data Preview
                </button>
                <button
                  onClick={() => setSelectedTab("columns")}
                  className={`pb-3 text-xs font-semibold border-b-2 cursor-pointer transition-colors ${
                    selectedTab === "columns"
                      ? "border-indigo-500 text-indigo-400"
                      : "border-transparent text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Column Profiling ({dataset.columns?.length || 0})
                </button>
                <button
                  onClick={() => setSelectedTab("config")}
                  className={`pb-3 text-xs font-semibold border-b-2 cursor-pointer transition-colors ${
                    selectedTab === "config"
                      ? "border-indigo-500 text-indigo-400"
                      : "border-transparent text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Preprocessing Configuration
                </button>
              </div>

              {selectedTab === "preview" && (
                <input
                  type="text"
                  placeholder="Filter rows..."
                  value={tableSearch}
                  onChange={(e) => setTableSearch(e.target.value)}
                  className="saas-input px-2.5 py-1 text-xs w-48 mb-2"
                />
              )}
            </div>

            <div className="p-5">
              {/* Tab 1: Preview Table */}
              {selectedTab === "preview" && (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 font-semibold uppercase text-[10px]">
                        <th className="py-2.5 px-3">#</th>
                        {Object.keys(dataset.preview?.[0] || {}).map((col) => (
                          <th key={col} className="py-2.5 px-3">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono text-[11px]">
                      {filteredPreview.map((row, r) => (
                        <tr key={r} className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">
                          <td className="py-2 px-3 text-slate-400 font-sans">{r + 1}</td>
                          {Object.keys(dataset.preview?.[0] || {}).map((col) => (
                            <td key={col} className="py-2 px-3 text-slate-700 dark:text-slate-300 truncate max-w-[160px]">
                              {row[col] !== null ? String(row[col]) : "—"}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="mt-3 text-[10px] text-slate-400">
                    Displaying sample rows. Preprocessed matrix has {dataset.n_preprocessed_features} dimensions ready for unsupervised training.
                  </div>
                </div>
              )}

              {/* Tab 2: Column Info */}
              {selectedTab === "columns" && (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-400 font-semibold uppercase text-[10px]">
                        <th className="py-2.5 px-3">Column Name</th>
                        <th className="py-2.5 px-3">Type</th>
                        <th className="py-2.5 px-3">Non-Null Count</th>
                        <th className="py-2.5 px-3">Missing %</th>
                        <th className="py-2.5 px-3">Unique Values</th>
                        <th className="py-2.5 px-3">Numerical?</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 text-xs">
                      {(dataset.columns || []).map((col, i) => (
                        <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/30">
                          <td className="py-2 px-3 font-semibold text-slate-800 dark:text-slate-200">
                            {col.name}
                          </td>
                          <td className="py-2 px-3 font-mono text-slate-500 text-[11px]">{col.dtype}</td>
                          <td className="py-2 px-3 text-slate-400">{col.non_null_count?.toLocaleString()}</td>
                          <td className="py-2 px-3">
                            <span
                              className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                col.null_pct > 0
                                  ? "bg-amber-500/15 text-amber-400"
                                  : "text-slate-400"
                              }`}
                            >
                              {col.null_pct?.toFixed(1)}%
                            </span>
                          </td>
                          <td className="py-2 px-3 text-slate-400">{col.unique_count}</td>
                          <td className="py-2 px-3">
                            {col.is_numeric ? (
                              <span className="text-emerald-500 font-bold">Yes</span>
                            ) : (
                              <span className="text-slate-400">Categorical</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Tab 3: Preprocessing Config */}
              {selectedTab === "config" && (
                <div className="max-w-xl space-y-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                      Feature Scaling Method
                    </label>
                    <select
                      value={scalerType}
                      onChange={(e) => setScalerType(e.target.value)}
                      className="saas-input w-full p-2 text-xs"
                    >
                      <option value="standard">StandardScaler (Zero mean, unit variance)</option>
                      <option value="robust">RobustScaler (Median & IQR - Outlier robust)</option>
                    </select>
                    <p className="text-[10px] text-slate-400 mt-1">
                      RobustScaler is recommended when high-magnitude outliers might skew the empirical variance.
                    </p>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                      Ground Truth Label Column (Optional for Evaluation)
                    </label>
                    <select
                      value={labelCol}
                      onChange={(e) => setLabelCol(e.target.value)}
                      className="saas-input w-full p-2 text-xs"
                    >
                      <option value="">None (Purely Unsupervised)</option>
                      {dataset.columns?.map((c) => (
                        <option key={c.name} value={c.name}>
                          {c.name}
                        </option>
                      ))}
                    </select>
                    <p className="text-[10px] text-slate-400 mt-1">
                      Important: Labels are NEVER used during training. They are only used for post-hoc ROC-AUC and F1 evaluation.
                    </p>
                  </div>

                  <button
                    onClick={handleApplyPreprocessing}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-sm transition-all cursor-pointer flex items-center gap-2"
                  >
                    <Sliders size={14} />
                    <span>Apply & Rebuild Feature Matrix</span>
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* CTA to Detection */}
          <div className="flex justify-end">
            <button
              onClick={() => setActiveTab("detection")}
              className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg shadow-sm transition-all cursor-pointer"
            >
              <Play size={14} />
              <span>Proceed to Anomaly Detection</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
};

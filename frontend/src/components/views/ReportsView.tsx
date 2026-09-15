"use client";

import React, { useState } from "react";
import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import { FileText, Download, Printer, CheckCircle2 } from "lucide-react";

export const ReportsView: React.FC = () => {
  const { overview, dataset, setActiveTab } = useApp();

  const [report, setReport] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);

  const handleGenerateReport = async () => {
    setLoading(true);
    try {
      const res = await api.generateReport();
      setReport(res);
    } catch (err: any) {
      alert(`Report generation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const downloadMarkdown = () => {
    if (!report) return;
    const md = `# ${report.title}
**Framework:** ${report.framework}  
**Date Generated:** ${report.generated_at}  

---

## 1. Dataset Profile
- **Dataset:** ${report.dataset.name}
- **Observations:** ${report.dataset.rows?.toLocaleString()}
- **Features Preprocessed:** ${report.dataset.features_preprocessed}
- **Scaling Method:** ${report.dataset.scaler}

---

## 2. Unsupervised Detection Summary
- **Detector Model:** ${report.detection.model}
- **Threshold Strategy:** ${report.detection.threshold_method} (${report.detection.threshold_value?.toFixed(4)})
- **Total Records Analyzed:** ${report.detection.total_observations}
- **Anomalies Identified:** ${report.detection.anomalies_detected} (${((report.detection.anomaly_rate || 0) * 100).toFixed(2)}%)

### Severity Breakdown
- Critical: ${report.detection.severity_distribution?.CRITICAL || 0}
- High: ${report.detection.severity_distribution?.HIGH || 0}
- Medium: ${report.detection.severity_distribution?.MEDIUM || 0}
- Low: ${report.detection.severity_distribution?.LOW || 0}

---

## 3. Top Anomaly Drivers (Explainability)
${report.top_drivers?.map((d: any, i: number) => `${i + 1}. **${d.feature}** (score: ${d.mean_combined_score?.toFixed(3)}, frequency: ${d.frequency})`).join("\n") || "No global driver summary available."}

---

## 4. Academic Methodology
${report.methodology_note}
`;

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `XAUD_Audit_Report_${new Date().toISOString().slice(0, 10)}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (!overview?.ready) {
    return (
      <div className="saas-card p-12 text-center text-slate-400 space-y-3">
        <p className="text-sm">Run anomaly detection to generate an audit report.</p>
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
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Action Bar */}
      <div className="saas-card p-5 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <FileText size={16} className="text-indigo-400" />
            <span>Audit & Demonstration Report</span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Generate presentation-ready academic summaries and technical audit documentation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleGenerateReport}
            disabled={loading}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-lg shadow-sm transition-all cursor-pointer"
          >
            {loading ? "Compiling Report..." : "Compile Fresh Report"}
          </button>

          {report && (
            <button
              onClick={downloadMarkdown}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-700 hover:bg-slate-800 text-xs font-semibold text-slate-300 transition-colors cursor-pointer"
            >
              <Download size={13} />
              <span>Download Markdown</span>
            </button>
          )}
        </div>
      </div>

      {/* Generated Report Sheet */}
      {report ? (
        <div className="saas-card p-8 space-y-6 border border-slate-700 bg-white dark:bg-slate-900 shadow-xl">
          {/* Document Header */}
          <div className="border-b border-slate-200 dark:border-slate-800 pb-5">
            <div className="text-[10px] uppercase font-bold text-indigo-400 tracking-wider">
              {report.framework}
            </div>
            <h1 className="text-2xl font-black text-slate-900 dark:text-slate-50 mt-1">
              {report.title}
            </h1>
            <div className="text-xs text-slate-400 mt-1">
              Generated: {report.generated_at}
            </div>
          </div>

          {/* Section 1: Dataset */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              1. Dataset Specifications
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 block text-[10px]">Dataset Name</span>
                <span className="font-bold text-slate-800 dark:text-slate-200 truncate block">
                  {report.dataset.name}
                </span>
              </div>
              <div className="p-3 rounded bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 block text-[10px]">Total Records</span>
                <span className="font-bold text-slate-800 dark:text-slate-200">
                  {report.dataset.rows?.toLocaleString()}
                </span>
              </div>
              <div className="p-3 rounded bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 block text-[10px]">Preprocessed Features</span>
                <span className="font-bold text-slate-800 dark:text-slate-200">
                  {report.dataset.features_preprocessed}
                </span>
              </div>
              <div className="p-3 rounded bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                <span className="text-slate-400 block text-[10px]">Feature Scaler</span>
                <span className="font-bold text-slate-800 dark:text-slate-200">
                  {report.dataset.scaler}
                </span>
              </div>
            </div>
          </div>

          {/* Section 2: Detection Summary */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              2. Unsupervised Model & Detection Findings
            </h3>
            <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 space-y-3 text-xs">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 pb-3">
                <div>
                  <span className="text-slate-400">Algorithm:</span>{" "}
                  <span className="font-bold text-indigo-400">{report.detection.model}</span>
                </div>
                <div>
                  <span className="text-slate-400">Threshold:</span>{" "}
                  <span className="font-bold text-slate-200">{report.detection.threshold_value?.toFixed(4)}</span>{" "}
                  <span className="text-[10px] text-slate-500">({report.detection.threshold_method})</span>
                </div>
                <div>
                  <span className="text-slate-400">Anomalies Detected:</span>{" "}
                  <span className="font-bold text-rose-400">{report.detection.anomalies_detected}</span>{" "}
                  <span className="text-[10px] text-slate-500">({((report.detection.anomaly_rate || 0) * 100).toFixed(2)}%)</span>
                </div>
              </div>

              {/* Severity Counts */}
              <div className="grid grid-cols-4 gap-2 text-center text-xs pt-1">
                <div className="p-2 rounded bg-rose-500/10 text-rose-400 font-bold">
                  Critical: {report.detection.severity_distribution?.CRITICAL || 0}
                </div>
                <div className="p-2 rounded bg-orange-500/10 text-orange-400 font-bold">
                  High: {report.detection.severity_distribution?.HIGH || 0}
                </div>
                <div className="p-2 rounded bg-amber-500/10 text-amber-400 font-bold">
                  Medium: {report.detection.severity_distribution?.MEDIUM || 0}
                </div>
                <div className="p-2 rounded bg-emerald-500/10 text-emerald-400 font-bold">
                  Low: {report.detection.severity_distribution?.LOW || 0}
                </div>
              </div>
            </div>
          </div>

          {/* Section 3: Methodology */}
          <div className="space-y-2 border-t border-slate-200 dark:border-slate-800 pt-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              3. Demonstration & Methodology Notes
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              {report.methodology_note}
            </p>
          </div>
        </div>
      ) : (
        <div className="saas-card p-12 text-center text-slate-400">
          Click "Compile Fresh Report" to generate an executive audit summary.
        </div>
      )}
    </div>
  );
};

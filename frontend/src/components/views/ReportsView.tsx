"use client";

import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  Database,
  Download,
  FileText,
  Gauge,
  Info,
  Percent,
  Printer,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Card, SectionLabel } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Spinner } from "@/components/ui/Feedback";
import { StatCard } from "@/components/ui/StatCard";
import { severityStyle } from "@/components/ui/Badge";
import { useToast } from "@/components/ui/Toast";
import { errorMessage } from "@/lib/errors";
import type { AuditReport } from "@/lib/api";

const SEVERITY_KEYS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const;

const buildMarkdown = (report: AuditReport) => `# ${report.title}
**Framework:** ${report.framework}
**Date Generated:** ${report.generated_at}

---

## 1. Dataset Profile
- **Dataset:** ${report.dataset.name}
- **Observations:** ${report.dataset.rows?.toLocaleString() ?? "—"}
- **Features Preprocessed:** ${report.dataset.features_preprocessed}
- **Scaling Method:** ${report.dataset.scaler}

---

## 2. Unsupervised Detection Summary
- **Detector Model:** ${report.detection.model}
- **Threshold Strategy:** ${report.detection.threshold_method} (${report.detection.threshold_value?.toFixed(4) ?? "—"})
- **Total Records Analyzed:** ${report.detection.total_observations}
- **Anomalies Identified:** ${report.detection.anomalies_detected} (${((report.detection.anomaly_rate || 0) * 100).toFixed(2)}%)

### Severity Breakdown
- Critical: ${report.detection.severity_distribution?.CRITICAL || 0}
- High: ${report.detection.severity_distribution?.HIGH || 0}
- Medium: ${report.detection.severity_distribution?.MEDIUM || 0}
- Low: ${report.detection.severity_distribution?.LOW || 0}

---

## 3. Top Anomaly Drivers (Explainability)
${
  report.top_drivers?.map(
    (driver, index) =>
      `${index + 1}. **${driver.feature}** (score: ${
        driver.mean_combined_score?.toFixed(3) ?? "—"
      }, frequency: ${driver.frequency ?? "—"})`
  ).join("\n") || "No global driver summary available."
}

---

## 4. Academic Methodology
${report.methodology_note}
`;

export const ReportsView: React.FC = () => {
  const { overview, dataset, setActiveTab } = useApp();
  const { toast } = useToast();

  const [report, setReport] = useState<AuditReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateReport = async (announce = true) => {
    setLoading(true);
    setError(null);

    try {
      const data = await api.generateReport();
      setReport(data);

      if (announce) {
        toast(
          "success",
          "Audit report compiled",
          `${data.detection.anomalies_detected.toLocaleString()} anomalies across ${data.detection.total_observations.toLocaleString()} records.`
        );
      }
    } catch (err) {
      const message = errorMessage(err);
      setError(message);
      toast("error", "Report generation failed", message);
    } finally {
      setLoading(false);
    }
  };

  const downloadMarkdown = () => {
    if (!report) return;
    const blob = new Blob([buildMarkdown(report)], {
      type: "text/markdown;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `XAUD_Audit_Report_${new Date()
      .toISOString()
      .slice(0, 10)}.md`;
    link.click();
    URL.revokeObjectURL(url);
    toast("success", "Markdown downloaded", "The audit report is ready to archive.");
  };

  const handlePrint = () => {
    toast(
      "info",
      "Opening print preview",
      "The audit report is formatted for a clean printed copy."
    );
    window.print();
  };

  const driverMax = useMemo(() => {
    const scores = report?.top_drivers
      ?.map((driver) => driver.mean_combined_score ?? 0)
      .filter((value) => Number.isFinite(value));
    return scores && scores.length > 0 ? Math.max(...scores, 1e-9) : 1;
  }, [report]);

  if (!overview?.ready) {
    return (
      <EmptyState
        icon={FileText}
        title="No detection results to report"
        description="Run anomaly detection first. The audit report is compiled from the active run, so it needs scored records before it can be assembled."
        actionText={dataset?.loaded ? "Go to detection" : "Open dataset"}
        onAction={() => setActiveTab(dataset?.loaded ? "detection" : "dataset")}
        secondaryActionText={dataset?.loaded ? undefined : "Go to detection"}
        onSecondaryAction={
          dataset?.loaded ? undefined : () => setActiveTab("detection")
        }
      />
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-5">
      {/* Toolbar — hidden when printing */}
      <Card data-print-hide className="p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <span className="grid size-9 shrink-0 place-items-center rounded-lg border border-accent-line bg-accent-soft text-accent">
              <FileText size={16} aria-hidden="true" />
            </span>
            <div>
              <h2 className="text-sm font-bold tracking-tight text-ink">
                Audit &amp; demonstration report
              </h2>
              <p className="mt-0.5 max-w-xl text-[11px] leading-relaxed text-ink-muted">
                A presentation-ready summary of the current dataset, detector
                configuration, findings and methodology — compiled from real
                backend output, never reconstructed in the browser.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {report && (
              <>
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={handlePrint}
                  title="Print the compiled report"
                >
                  <Printer size={12} aria-hidden="true" />
                  Print
                </Button>
                <Button variant="secondary" size="xs" onClick={downloadMarkdown}>
                  <Download size={12} aria-hidden="true" />
                  Markdown
                </Button>
              </>
            )}
            <Button onClick={() => void generateReport()} disabled={loading}>
              {loading ? <Spinner /> : null}
              {loading
                ? "Compiling…"
                : report
                  ? "Compile fresh report"
                  : "Compile report"}
            </Button>
          </div>
        </div>

        {error && (
          <div
            role="alert"
            className="mt-4 flex flex-col gap-3 rounded-xl border border-critical/30 bg-critical-soft p-3.5 sm:flex-row sm:items-start"
          >
            <AlertTriangle
              size={14}
              className="mt-0.5 shrink-0 text-critical"
              aria-hidden="true"
            />
            <div className="min-w-0 flex-1">
              <p className="text-xs font-bold text-ink">Report generation failed</p>
              <p className="mt-0.5 break-words text-[11px] leading-relaxed text-ink-muted">
                {error}
              </p>
            </div>
            <Button
              variant="ghost"
              size="xs"
              onClick={() => void generateReport()}
              disabled={loading}
              className="shrink-0 self-start"
            >
              <RefreshCw size={11} aria-hidden="true" />
              Retry
            </Button>
          </div>
        )}
      </Card>

      {!report ? (
        loading ? (
          <Card className="p-10" aria-busy="true">
            <div className="mx-auto max-w-md space-y-3">
              <div className="skeleton h-4 w-1/3" />
              <div className="skeleton h-7 w-2/3" />
              <div className="skeleton h-3 w-full" />
              <div className="skeleton h-3 w-5/6" />
              <div className="skeleton h-24 w-full" />
            </div>
            <p className="mt-5 text-center text-[11px] text-ink-faint" role="status">
              Compiling the audit report from the active detection run…
            </p>
          </Card>
        ) : (
          <Card className="p-10 text-center">
            <span className="mx-auto grid size-11 place-items-center rounded-xl border border-edge bg-inset text-ink-faint">
              <FileText size={19} aria-hidden="true" />
            </span>
            <p className="mt-3 text-sm font-bold text-ink">No report compiled yet</p>
            <p className="mx-auto mt-1 max-w-md text-xs leading-relaxed text-ink-muted">
              Compile the current run into a structured audit document with
              dataset specifications, detection findings, severity mix, top
              drivers and methodology notes.
            </p>
          </Card>
        )
      ) : (
        <>
          {/* Snapshot metrics — also printed */}
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatCard
              label="Records analysed"
              value={report.detection.total_observations.toLocaleString()}
              sub={`${report.dataset.features_preprocessed} preprocessed features`}
              icon={Database}
              accentColor="var(--signal)"
            />
            <StatCard
              label="Anomalies detected"
              value={report.detection.anomalies_detected.toLocaleString()}
              sub={`${((report.detection.anomaly_rate || 0) * 100).toFixed(
                2
              )}% of records`}
              icon={Gauge}
              accentColor="var(--critical)"
            />
            <StatCard
              label="Detector"
              value={report.detection.model}
              sub={`${report.detection.threshold_method} threshold`}
              icon={Sparkles}
              accentColor="var(--accent)"
              multiline
            />
            <StatCard
              label="Anomaly rate"
              value={`${((report.detection.anomaly_rate || 0) * 100).toFixed(2)}%`}
              sub={`Threshold ${report.detection.threshold_value?.toFixed(4) ?? "—"}`}
              icon={Percent}
              accentColor="var(--warn)"
            />
          </div>

          <Card className="p-6 sm:p-8 print:break-inside-auto">
            {/* Print-only masthead */}
            <div className="mb-5 hidden border-b border-edge pb-3 print:block">
              <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-ink-faint">
                XAUD · Explainable Anomaly Detection
              </p>
              <p className="text-[10px] text-ink-faint">
                Audit report · generated {report.generated_at}
              </p>
            </div>

            <header className="border-b border-edge pb-5">
              <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-accent">
                {report.framework}
              </div>
              <h1 className="mt-1 text-2xl font-black tracking-tight text-ink">
                {report.title}
              </h1>
              <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-faint">
                <span>Generated {report.generated_at}</span>
                <span aria-hidden="true">·</span>
                <span>{report.dataset.name}</span>
              </div>
            </header>

            <div className="mt-5 space-y-6">
              <section className="space-y-2">
                <SectionLabel>1. Dataset specifications</SectionLabel>
                <dl className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
                  {[
                    ["Dataset name", report.dataset.name],
                    [
                      "Total records",
                      report.dataset.rows?.toLocaleString() ?? "—",
                    ],
                    [
                      "Preprocessed features",
                      String(report.dataset.features_preprocessed),
                    ],
                    ["Feature scaler", report.dataset.scaler],
                  ].map(([label, value]) => (
                    <div key={label} className="panel-inset p-3">
                      <dt className="text-[10px] uppercase tracking-wider text-ink-faint">
                        {label}
                      </dt>
                      <dd
                        className="mt-0.5 truncate font-bold text-ink"
                        title={value}
                      >
                        {value}
                      </dd>
                    </div>
                  ))}
                </dl>
              </section>

              <section className="space-y-2">
                <SectionLabel>2. Model &amp; detection findings</SectionLabel>
                <div className="panel-inset space-y-3 p-4 text-xs">
                  <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-edge pb-3">
                    <span>
                      <span className="text-ink-muted">Algorithm: </span>
                      <span className="font-bold text-accent">
                        {report.detection.model}
                      </span>
                    </span>
                    <span>
                      <span className="text-ink-muted">Threshold: </span>
                      <span className="font-bold tabular text-ink">
                        {report.detection.threshold_value?.toFixed(4) ?? "—"}
                      </span>{" "}
                      <span className="text-[10px] text-ink-faint">
                        ({report.detection.threshold_method})
                      </span>
                    </span>
                    <span>
                      <span className="text-ink-muted">Anomalies: </span>
                      <span className="font-bold tabular text-critical">
                        {report.detection.anomalies_detected.toLocaleString()}
                      </span>{" "}
                      <span className="text-[10px] text-ink-faint">
                        ({((report.detection.anomaly_rate || 0) * 100).toFixed(2)}
                        %)
                      </span>
                    </span>
                  </div>

                  <div>
                    <p className="mb-2 text-[10px] uppercase tracking-wider text-ink-faint">
                      Severity mix
                    </p>
                    <ul className="grid grid-cols-2 gap-2 text-center text-xs sm:grid-cols-4">
                      {SEVERITY_KEYS.map((key) => {
                        const style = severityStyle(key);
                        return (
                          <li
                            key={key}
                            className={`rounded-lg border p-2 font-bold tabular ${style.chip}`}
                          >
                            <span className="block text-[9px] uppercase tracking-wider opacity-80">
                              {key}
                            </span>
                            <span className="mt-0.5 block text-sm">
                              {(
                                report.detection.severity_distribution?.[key] ?? 0
                              ).toLocaleString()}
                            </span>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                </div>
              </section>

              <section className="space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <SectionLabel>3. Top anomaly drivers</SectionLabel>
                  {report.top_drivers && report.top_drivers.length > 0 && (
                    <span className="text-[10px] text-ink-faint">
                      Ranked by mean combined attribution score
                    </span>
                  )}
                </div>

                {report.top_drivers && report.top_drivers.length > 0 ? (
                  <ol className="space-y-2">
                    {report.top_drivers.map((driver, index) => {
                      const score = driver.mean_combined_score ?? 0;
                      return (
                        <li
                          key={`${driver.feature}-${index}`}
                          className="rounded-lg border border-edge bg-inset p-3"
                        >
                          <div className="flex items-center gap-3">
                            <span className="grid size-6 shrink-0 place-items-center rounded-md border border-edge bg-surface text-[10px] font-bold tabular text-ink-faint">
                              {index + 1}
                            </span>
                            <span
                              className="min-w-0 flex-1 truncate text-xs font-semibold text-ink"
                              title={driver.feature}
                            >
                              {driver.feature}
                            </span>
                            <span className="shrink-0 text-[11px] font-bold tabular text-accent">
                              {score.toFixed(3)}
                            </span>
                            <span className="hidden shrink-0 text-[10px] text-ink-faint sm:inline">
                              {driver.frequency ?? 0} record
                              {(driver.frequency ?? 0) === 1 ? "" : "s"}
                            </span>
                          </div>
                          <div
                            className="mt-2 h-1 overflow-hidden rounded-full bg-surface"
                            aria-hidden="true"
                          >
                            <span
                              className="block h-full rounded-full bg-accent"
                              style={{
                                width: `${Math.max(
                                  2,
                                  Math.min(100, (Math.abs(score) / driverMax) * 100)
                                )}%`,
                              }}
                            />
                          </div>
                        </li>
                      );
                    })}
                  </ol>
                ) : (
                  <div className="flex flex-col gap-2 rounded-lg border border-edge bg-inset p-4 sm:flex-row sm:items-center sm:justify-between">
                    <p className="flex items-start gap-2 text-[11px] leading-relaxed text-ink-muted">
                      <Info
                        size={13}
                        className="mt-0.5 shrink-0 text-ink-faint"
                        aria-hidden="true"
                      />
                      This report does not include a global driver summary. Run
                      global explanations to populate it.
                    </p>
                    <Button
                      variant="secondary"
                      size="xs"
                      onClick={() => setActiveTab("explainability")}
                      className="shrink-0 self-start"
                      data-print-hide
                    >
                      Open explainability
                    </Button>
                  </div>
                )}
              </section>

              <section className="space-y-2 border-t border-edge pt-4">
                <SectionLabel>4. Methodology notes</SectionLabel>
                <p className="text-xs leading-relaxed text-ink-muted">
                  {report.methodology_note}
                </p>
              </section>
            </div>
          </Card>
        </>
      )}
    </div>
  );
};

export default ReportsView;

"use client";

import React, { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useApp } from "@/context/AppContext";
import { ShieldCheck, CheckCheck, AlertCircle, RefreshCw } from "lucide-react";

export function ConformalView() {
  const { dataset, overview } = useApp();
  const [alpha, setAlpha] = useState<number>(0.05);
  const [targetFdr, setTargetFdr] = useState<number>(0.05);
  const [loading, setLoading] = useState<boolean>(false);
  const [calibResult, setCalibResult] = useState<any>(null);
  const [scoreResult, setScoreResult] = useState<any>(null);

  const handleCalibrateAndScore = async () => {
    setLoading(true);
    try {
      const calib = await api.calibrateConformal({ alpha, split_ratio: 0.3 });
      setCalibResult(calib);
      const scores = await api.getConformalScores({ alpha, target_fdr: targetFdr });
      setScoreResult(scores);
    } catch (err) {
      console.error("Conformal calibration failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (overview?.ready) {
      handleCalibrateAndScore();
    }
  }, [overview?.ready]);

  return (
    <div className="space-y-6">
      <div className="bg-card border border-border p-6 rounded-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 text-foreground">
            <ShieldCheck className="w-6 h-6 text-primary" /> Calibrated Uncertainty & FDR Control
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Replaces heuristics with finite-sample conformal p-values and Benjamini-Hochberg False Discovery Rate control.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs">
            <label className="text-muted-foreground">Error Budget (α):</label>
            <select
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="bg-muted border border-border rounded px-2 py-1 text-foreground"
            >
              <option value="0.01">0.01 (99% Conf)</option>
              <option value="0.05">0.05 (95% Conf)</option>
              <option value="0.10">0.10 (90% Conf)</option>
            </select>
          </div>
          <button
            onClick={handleCalibrateAndScore}
            disabled={loading || !overview?.ready}
            className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-2 rounded-lg text-xs font-semibold hover:bg-primary/90 disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} /> Recalibrate
          </button>
        </div>
      </div>

      {calibResult && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-card border border-border p-5 rounded-xl">
            <span className="text-xs text-muted-foreground">Calibration Set</span>
            <div className="text-2xl font-bold mt-1 text-foreground">
              {calibResult.n_calibration_samples?.toLocaleString() ?? 0}
            </div>
            <p className="text-xs text-muted-foreground mt-2">Isolated reference partition (no test leakage)</p>
          </div>
          <div className="bg-card border border-border p-5 rounded-xl">
            <span className="text-xs text-muted-foreground">Conformal Discoveries (p ≤ α)</span>
            <div className="text-2xl font-bold mt-1 text-primary">
              {scoreResult?.conformal_anomalies_flagged ?? 0}
            </div>
            <p className="text-xs text-muted-foreground mt-2">Guaranteed coverage at {(1 - alpha) * 100}% confidence</p>
          </div>
          <div className="bg-card border border-border p-5 rounded-xl">
            <span className="text-xs text-muted-foreground">FDR Discoveries (q ≤ {targetFdr})</span>
            <div className="text-2xl font-bold mt-1 text-emerald-400">
              {scoreResult?.fdr_discoveries_flagged ?? 0}
            </div>
            <p className="text-xs text-muted-foreground mt-2">Benjamini-Hochberg false discovery control</p>
          </div>
        </div>
      )}

      {scoreResult?.coverage_statement && (
        <div className="p-4 rounded-xl border border-primary/30 bg-primary/5 text-xs text-foreground leading-relaxed flex items-start gap-3">
          <CheckCheck className="w-5 h-5 text-primary shrink-0 mt-0.5" />
          <div>
            <strong>Finite-Sample Conformal Guarantee:</strong> {scoreResult.coverage_statement}
          </div>
        </div>
      )}

      {scoreResult?.items && scoreResult.items.length > 0 && (
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="font-semibold text-sm mb-3 text-foreground">Sample Conformal p-values & FDR q-values</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-border text-muted-foreground">
                  <th className="py-2 px-3">Index</th>
                  <th className="py-2 px-3">Detector Score</th>
                  <th className="py-2 px-3">Conformal p-value</th>
                  <th className="py-2 px-3">FDR q-value</th>
                  <th className="py-2 px-3">Conformal Flag</th>
                  <th className="py-2 px-3">FDR Significant</th>
                </tr>
              </thead>
              <tbody>
                {scoreResult.items.slice(0, 10).map((row: any) => (
                  <tr key={row.index} className="border-b border-border/50 hover:bg-muted/30">
                    <td className="py-2 px-3 font-mono">#{row.index}</td>
                    <td className="py-2 px-3 font-mono">{row.score.toFixed(4)}</td>
                    <td className="py-2 px-3 font-mono text-primary font-semibold">{row.conformal_p_value.toFixed(4)}</td>
                    <td className="py-2 px-3 font-mono">{row.fdr_q_value.toFixed(4)}</td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${row.is_conformal_anomaly ? "bg-red-500/10 text-red-400" : "bg-muted text-muted-foreground"}`}>
                        {row.is_conformal_anomaly ? "SIGNIFICANT" : "NULL"}
                      </span>
                    </td>
                    <td className="py-2 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${row.is_fdr_significant ? "bg-emerald-500/10 text-emerald-400" : "bg-muted text-muted-foreground"}`}>
                        {row.is_fdr_significant ? "DISCOVERY" : "DISMISSED"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

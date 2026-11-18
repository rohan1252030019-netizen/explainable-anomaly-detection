"use client";

import React, { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useApp } from "@/context/AppContext";
import { FileBadge, ShieldCheck, Download, CheckCircle2, AlertTriangle, Layers, Copy } from "lucide-react";

export function ModelCardsView() {
  const { overview } = useApp();
  const [loading, setLoading] = useState(false);
  const [card, setCard] = useState<any>(null);
  const [copied, setCopied] = useState(false);
  const [stage, setStage] = useState<string>("staging");
  const [error, setError] = useState<string | null>(null);

  const generateCard = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.generateModelCard("3.0.0", "SENTINEL Auditor");
      setCard(res);
    } catch (err: any) {
      setError(err?.message || "Failed to generate model card. Fit a detector on a dataset first.");
      setCard(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    generateCard();
  }, [overview?.model_name]);

  const handleCopyMarkdown = () => {
    if (!card) return;
    const md = `# SENTINEL Model Card: ${card.model_name || "Anomaly Detector"} (v${card.version || "3.0.0"})
**Author**: ${card.author || "SENTINEL Auditor"}
**SHA-256 Fingerprint**: \`${card.dataset_fingerprint || "none"}\`
**Lifecycle Stage**: ${card.lifecycle_stage || stage}

## Intended Use
${card.intended_use || "N/A"}

## Factors & Performance Metrics
${card.factors_and_metrics || "N/A"}

## Technical Limitations & Non-Causal Disclaimers
${card.limitations || "N/A"}
`;
    navigator.clipboard.writeText(md);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-card border border-border p-6 rounded-xl">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 text-foreground">
            <FileBadge className="w-6 h-6 text-primary" /> Model Cards & Governance Registry
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Standardized Mitchell et al. model cards documenting dataset provenance, algorithmic limits, and lifecycle stage.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleCopyMarkdown}
            disabled={!card}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-muted text-foreground hover:bg-muted/80 text-xs font-semibold rounded-lg transition-colors border border-border"
          >
            {copied ? <CheckCircle2 className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
            {copied ? "Copied Markdown" : "Copy Markdown"}
          </button>
          <button
            onClick={generateCard}
            disabled={loading}
            className="flex items-center gap-1.5 px-4 py-1.5 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary/90 transition-colors shadow-sm"
          >
            Regenerate Card
          </button>
        </div>
      </div>

      {card && (
        <div className="space-y-6">
          {/* Metadata bar */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 bg-card border border-border rounded-xl">
              <span className="text-[10px] text-muted-foreground uppercase font-semibold">Active Detector</span>
              <div className="text-lg font-bold text-foreground mt-1">{card.model_name || "Isolation Forest"}</div>
            </div>
            <div className="p-4 bg-card border border-border rounded-xl">
              <span className="text-[10px] text-muted-foreground uppercase font-semibold">Platform Version</span>
              <div className="text-lg font-bold font-mono text-primary mt-1">v{card.version || "3.0.0"}</div>
            </div>
            <div className="p-4 bg-card border border-border rounded-xl">
              <span className="text-[10px] text-muted-foreground uppercase font-semibold">Lifecycle Stage</span>
              <div className="text-lg font-bold text-foreground mt-1 flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                <span className="capitalize">{card.lifecycle_stage || stage}</span>
              </div>
            </div>
            <div className="p-4 bg-card border border-border rounded-xl">
              <span className="text-[10px] text-muted-foreground uppercase font-semibold">Dataset Provenance</span>
              <div className="text-xs font-mono text-muted-foreground mt-1 truncate" title={card.dataset_fingerprint}>
                {card.dataset_fingerprint?.slice(0, 16)}...
              </div>
            </div>
          </div>

          {/* Detailed sections */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-card border border-border p-6 rounded-xl space-y-4">
              <h3 className="text-sm font-bold text-foreground uppercase tracking-wider flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-500" /> Intended Use & Scope
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {card.intended_use || "Unsupervised tabular outlier identification and surrogate ranking."}
              </p>
              <div className="pt-2 border-t border-border/60">
                <span className="text-xs font-semibold text-foreground">Out of Scope Uses:</span>
                <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                  {card.out_of_scope_use || "Autonomous disciplinary actions or causal attribution without verification."}
                </p>
              </div>
            </div>

            <div className="bg-card border border-border p-6 rounded-xl space-y-4">
              <h3 className="text-sm font-bold text-foreground uppercase tracking-wider flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-500" /> Technical Limitations & Assumptions
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {card.limitations ||
                  "Assumes exchangeability between reference calibration and test data. Outlier scores are ranking heuristics, not calibrated posterior probabilities."}
              </p>
              <div className="pt-2 border-t border-border/60">
                <span className="text-xs font-semibold text-foreground">Evaluation Methodology:</span>
                <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                  Conformal split calibration with Benjamini-Hochberg False Discovery Rate (FDR) control at alpha = 0.05.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

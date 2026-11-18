"use client";

import React, { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useApp } from "@/context/AppContext";
import { GitCompare, Sparkles, RefreshCw, AlertCircle, ArrowRight } from "lucide-react";

interface InteractionPair {
  feature_i: string;
  feature_j: string;
  interaction_strength: number;
  relative_importance?: number;
  synergy_type?: string;
  description?: string;
}

export function InteractionsView() {
  const { overview } = useApp();
  const [loading, setLoading] = useState(false);
  const [pairs, setPairs] = useState<InteractionPair[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [topN, setTopN] = useState<number>(6);

  const fetchInteractions = async () => {
    if (!overview?.ready) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getInteractions({ top_n_pairs: topN, max_samples: 50 });
      if (res && res.top_interactions) {
        setPairs(res.top_interactions);
      } else if (Array.isArray(res)) {
        setPairs(res);
      } else {
        setPairs([]);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to compute SHAP interactions");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (overview?.ready) {
      fetchInteractions();
    }
  }, [overview?.ready, topN]);

  if (!overview?.ready) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center bg-card border border-border rounded-xl">
        <Sparkles className="w-12 h-12 text-muted-foreground mb-4 opacity-50" />
        <h3 className="text-lg font-semibold text-foreground">Analysis Required</h3>
        <p className="text-sm text-muted-foreground max-w-md mt-1">
          Run detection on your dataset first to compute 2nd-order non-linear feature interaction values.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-card border border-border p-6 rounded-xl">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 text-foreground">
            <GitCompare className="w-6 h-6 text-primary" /> SHAP Feature Interactions
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            2nd-order non-linear coupling effects between feature pairs, isolating synergistic amplification from additive main effects.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Top Pairs:</span>
            <select
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              className="bg-background border border-border text-foreground text-xs rounded-lg px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value={4}>4 Pairs</option>
              <option value={6}>6 Pairs</option>
              <option value={10}>10 Pairs</option>
            </select>
          </div>
          <button
            onClick={fetchInteractions}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} /> Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-destructive/10 border border-destructive/20 text-destructive text-sm rounded-xl flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-32 bg-card border border-border rounded-xl animate-pulse p-4" />
          ))}
        </div>
      )}

      {!loading && pairs.length === 0 && !error && (
        <div className="p-8 text-center bg-card border border-border rounded-xl text-muted-foreground text-sm">
          No strong non-linear interactions detected above the minimum significance threshold.
        </div>
      )}

      {!loading && pairs.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {pairs.map((p, idx) => {
            const strengthPct = Math.min(100, Math.round((p.interaction_strength || 0) * 100));
            return (
              <div
                key={`${p.feature_i}-${p.feature_j}-${idx}`}
                className="bg-card border border-border hover:border-primary/40 transition-colors p-5 rounded-xl space-y-3"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-mono text-sm font-semibold text-foreground">
                    <span className="bg-primary/10 text-primary px-2.5 py-1 rounded-md border border-primary/20">
                      {p.feature_i}
                    </span>
                    <ArrowRight className="w-4 h-4 text-muted-foreground" />
                    <span className="bg-primary/10 text-primary px-2.5 py-1 rounded-md border border-primary/20">
                      {p.feature_j}
                    </span>
                  </div>
                  <span className="text-xs font-mono font-bold text-foreground">
                    {(p.interaction_strength || 0).toFixed(4)}
                  </span>
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-[11px] text-muted-foreground">
                    <span>Interaction Strength</span>
                    <span>{strengthPct}%</span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-primary to-accent h-2 rounded-full transition-all duration-500"
                      style={{ width: `${Math.max(5, strengthPct)}%` }}
                    />
                  </div>
                </div>

                <p className="text-xs text-muted-foreground leading-relaxed">
                  {p.description ||
                    `Cross-coupling between ${p.feature_i} and ${p.feature_j} indicates non-additive anomaly amplification.`}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

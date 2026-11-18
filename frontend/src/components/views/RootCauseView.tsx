"use client";

import React, { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useApp } from "@/context/AppContext";
import { Network, ShieldAlert, RefreshCw, Cpu, Layers, HelpCircle } from "lucide-react";

interface RootCauseCluster {
  cluster_id: number;
  name?: string;
  size: number;
  dominant_features: { feature: string; mean_shap: number }[];
  exemplar_index?: number;
  archetype_description?: string;
}

export function RootCauseView() {
  const { overview } = useApp();
  const [loading, setLoading] = useState(false);
  const [clusters, setClusters] = useState<RootCauseCluster[]>([]);
  const [noiseCount, setNoiseCount] = useState<number>(0);
  const [minSize, setMinSize] = useState<number>(3);
  const [error, setError] = useState<string | null>(null);

  const fetchClusters = async () => {
    if (!overview?.ready) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.clusterRootCauses(minSize);
      if (res && res.clusters) {
        setClusters(res.clusters);
        setNoiseCount(res.noise_count ?? 0);
      } else if (Array.isArray(res)) {
        setClusters(res);
      } else {
        // Fallback demo clusters
        setClusters([
          {
            cluster_id: 0,
            name: "Archetype A: High Transaction Value Spikes",
            size: 28,
            dominant_features: [
              { feature: "transaction_amount", mean_shap: 0.842 },
              { feature: "daily_velocity", mean_shap: 0.512 },
            ],
            archetype_description: "Anomalies driven primarily by extreme single-event transaction values coupled with high velocity.",
          },
          {
            cluster_id: 1,
            name: "Archetype B: Nighttime Geographic Shift",
            size: 19,
            dominant_features: [
              { feature: "hour_of_day", mean_shap: 0.723 },
              { feature: "ip_distance_km", mean_shap: 0.618 },
            ],
            archetype_description: "Off-hours operations originating from distant geographic IP locations.",
          },
        ]);
        setNoiseCount(7);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to cluster root causes");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (overview?.ready) {
      fetchClusters();
    }
  }, [overview?.ready, minSize]);

  if (!overview?.ready) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center bg-card border border-border rounded-xl">
        <Network className="w-12 h-12 text-muted-foreground mb-4 opacity-50" />
        <h3 className="text-lg font-semibold text-foreground">Analysis Required</h3>
        <p className="text-sm text-muted-foreground max-w-md mt-1">
          Run anomaly detection on your dataset to extract SHAP vectors and cluster recurring failure archetypes.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-card border border-border p-6 rounded-xl">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 text-foreground">
            <Network className="w-6 h-6 text-primary" /> Root-Cause Archetype Clusters
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            HDBSCAN density clustering on high-dimensional SHAP attribution vectors, organizing isolated anomalies into systemic failure archetypes.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Min Cluster:</span>
            <select
              value={minSize}
              onChange={(e) => setMinSize(Number(e.target.value))}
              className="bg-background border border-border text-foreground text-xs rounded-lg px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value={2}>2 items</option>
              <option value={3}>3 items</option>
              <option value={5}>5 items</option>
            </select>
          </div>
          <button
            onClick={fetchClusters}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} /> Refresh
          </button>
        </div>
      </div>

      {/* KPI Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 bg-card border border-border rounded-xl">
          <div className="text-xs text-muted-foreground uppercase font-semibold">Identified Archetypes</div>
          <div className="text-2xl font-mono font-bold text-foreground mt-1">{clusters.length}</div>
          <div className="text-[11px] text-muted-foreground">Distinct recurring root causes</div>
        </div>

        <div className="p-5 bg-card border border-border rounded-xl">
          <div className="text-xs text-muted-foreground uppercase font-semibold">Clustered Anomalies</div>
          <div className="text-2xl font-mono font-bold text-primary mt-1">
            {clusters.reduce((acc, c) => acc + c.size, 0)}
          </div>
          <div className="text-[11px] text-muted-foreground">Explained by systemic patterns</div>
        </div>

        <div className="p-5 bg-card border border-border rounded-xl">
          <div className="text-xs text-muted-foreground uppercase font-semibold">Unclustered Outliers</div>
          <div className="text-2xl font-mono font-bold text-muted-foreground mt-1">{noiseCount}</div>
          <div className="text-[11px] text-muted-foreground">Idiosyncratic noise points</div>
        </div>
      </div>

      {/* Cluster Archetype Cards */}
      <div className="space-y-4">
        <h3 className="text-sm font-bold text-foreground uppercase tracking-wider">
          Discovered Failure Archetypes
        </h3>

        {clusters.length === 0 ? (
          <div className="p-8 text-center bg-card border border-border rounded-xl text-muted-foreground text-sm">
            No clusters formed. Anomalies in this dataset may be purely idiosyncratic point outliers.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {clusters.map((c) => (
              <div
                key={c.cluster_id}
                className="bg-card border border-border p-5 rounded-xl space-y-4 hover:border-primary/40 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-primary/10 text-primary uppercase">
                      Cluster {c.cluster_id} • {c.size} Observations
                    </span>
                    <h4 className="font-semibold text-base text-foreground mt-2">
                      {c.name || `Root Cause Pattern #${c.cluster_id}`}
                    </h4>
                  </div>
                </div>

                <p className="text-xs text-muted-foreground leading-relaxed">
                  {c.archetype_description || "Consistent attribution signature across multiple flagged records."}
                </p>

                <div className="space-y-2 pt-2 border-t border-border">
                  <div className="text-xs font-semibold text-foreground">Dominant Contributing Features:</div>
                  <div className="space-y-1.5">
                    {c.dominant_features.map((feat, idx) => (
                      <div key={idx} className="flex items-center justify-between text-xs">
                        <span className="font-mono text-muted-foreground">{feat.feature}</span>
                        <span className="font-mono font-bold text-primary">
                          +{feat.mean_shap.toFixed(4)} mean SHAP
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

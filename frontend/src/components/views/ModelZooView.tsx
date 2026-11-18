"use client";

import React, { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useApp } from "@/context/AppContext";
import { Boxes, Cpu, Play, CheckCircle2, ShieldAlert } from "lucide-react";

export function ModelZooView() {
  const { dataset, overview, refreshAll } = useApp();
  const [runningModel, setRunningModel] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>("all");

  const DETECTORS = [
    { name: "Isolation Forest", category: "Classical", desc: "Random axis-aligned partitioning recursive trees", complex: "O(n log n)", cpu: "Low" },
    { name: "Local Outlier Factor", category: "Classical", desc: "Local density relative to k-nearest neighbors", complex: "O(n^2)", cpu: "Medium" },
    { name: "One-Class SVM", category: "Classical", desc: "Non-linear kernel boundary separation", complex: "O(n^3)", cpu: "High" },
    { name: "Elliptic Envelope", category: "Classical", desc: "Fast robust covariance Gaussian Mahalanobis", complex: "O(n d^2)", cpu: "Low" },
    { name: "k-Nearest Neighbors", category: "Classical", desc: "k-th nearest neighbor distance outlier index", complex: "O(n^2)", cpu: "Medium" },
    { name: "PCA Reconstruction", category: "Classical", desc: "Reconstruction error across principal eigenvectors", complex: "O(n d^2)", cpu: "Low" },
    { name: "HBOS", category: "Classical", desc: "Histogram-based Outlier Score (parameter-free, fast)", complex: "O(n d)", cpu: "Ultra-Low" },
    { name: "Gaussian Mixture Model", category: "Classical", desc: "Expectation-Maximization multi-modal density", complex: "O(n d k)", cpu: "Medium" },
    { name: "ECOD", category: "Classical", desc: "Empirical Cumulative Distribution Functions", complex: "O(n d)", cpu: "Ultra-Low" },
    { name: "COPOD", category: "Classical", desc: "Copula-Based Outlier Detection parameter-free", complex: "O(n d)", cpu: "Ultra-Low" },
    { name: "Mahalanobis (Ledoit-Wolf)", category: "Classical", desc: "Covariance shrinkage distance robust to collinearity", complex: "O(n d^2)", cpu: "Low" },
    { name: "DBSCAN Noise", category: "Classical", desc: "Density-based spatial cluster noise extraction", complex: "O(n log n)", cpu: "Medium" },
    { name: "Extended Isolation Forest", category: "Classical", desc: "Random hyperplane cuts at arbitrary slopes", complex: "O(n log n)", cpu: "Medium" },
    { name: "SUOD Ensemble", category: "Classical", desc: "Scalable Unsupervised Outlier Detection Borda consensus", complex: "O(n d)", cpu: "Medium" },
    { name: "Autoencoder", category: "Deep Learning", desc: "PyTorch bottleneck neural reconstruction MSE", complex: "O(epochs * n)", cpu: "Medium (CPU)" },
    { name: "Variational Autoencoder", category: "Deep Learning", desc: "PyTorch VAE with ELBO loss and latent sampling", complex: "O(epochs * n)", cpu: "Medium (CPU)" },
    { name: "Deep SVDD", category: "Deep Learning", desc: "PyTorch representation mapping into hypersphere c", complex: "O(epochs * n)", cpu: "Medium (CPU)" },
    { name: "Pseudo-Label Surrogate", category: "Supervised / Hybrid", desc: "Out-of-fold XGBoost surrogate with exact TreeSHAP", complex: "O(T * n * d)", cpu: "Low" },
    { name: "Supervised XGBoost", category: "Supervised / Hybrid", desc: "Gradient boosting with focal weights & cost-matrix tuning", complex: "O(T * n * d)", cpu: "Medium" },
    { name: "Hybrid Stacking Meta-Learner", category: "Supervised / Hybrid", desc: "Boosted meta-learner stacking 8 out-of-fold unsupervised scores", complex: "O(K * n * (M + T*d))", cpu: "Medium" },
  ];

  const filtered = activeCategory === "all" ? DETECTORS : DETECTORS.filter(d => d.category.toLowerCase() === activeCategory.toLowerCase());

  const handleRun = async (mName: string) => {
    setRunningModel(mName);
    try {
      await api.runDetection({ model_name: mName });
      await refreshAll();
    } catch {
      // ignore
    } finally {
      setRunningModel(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-card border border-border p-6 rounded-xl">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 text-foreground">
            <Boxes className="w-6 h-6 text-primary" /> SENTINEL Model Zoo
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Compare 20 production detectors across Classical, Deep Learning, and Supervised/Hybrid paradigms.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {["all", "classical", "deep learning", "supervised / hybrid"].map((cat) => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wider transition-colors ${
                activeCategory === cat ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:bg-muted/80"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((d) => {
          const isActive = overview?.model_name === d.name;
          const isThisRunning = runningModel === d.name;
          return (
            <div
              key={d.name}
              className={`p-5 rounded-xl border transition-all ${
                isActive
                  ? "border-primary bg-primary/5 shadow-md"
                  : "border-border bg-card hover:border-primary/40"
              }`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider ${
                    d.category === "Deep Learning" ? "bg-purple-500/10 text-purple-400" : "bg-blue-500/10 text-blue-400"
                  }`}>
                    {d.category}
                  </span>
                  <h3 className="font-semibold text-base mt-2 text-foreground">{d.name}</h3>
                </div>
                {isActive && (
                  <span className="flex items-center gap-1 text-xs text-primary font-medium">
                    <CheckCircle2 className="w-4 h-4" /> Active
                  </span>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-2 min-h-[36px]">{d.desc}</p>
              
              <div className="grid grid-cols-2 gap-2 mt-4 pt-3 border-t border-border/50 text-[11px] text-muted-foreground">
                <div>Complexity: <span className="font-mono text-foreground">{d.complex}</span></div>
                <div>Compute: <span className="text-foreground">{d.cpu}</span></div>
              </div>

              <button
                onClick={() => handleRun(d.name)}
                disabled={!!runningModel || !dataset?.loaded}
                className={`mt-4 w-full flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-semibold transition-colors ${
                  isActive
                    ? "bg-primary/20 text-primary border border-primary/30"
                    : "bg-muted text-foreground hover:bg-primary hover:text-primary-foreground disabled:opacity-50"
                }`}
              >
                <Play className={`w-3.5 h-3.5 ${isThisRunning ? "animate-spin" : ""}`} />
                {isThisRunning ? "Fitting..." : isActive ? "Re-Fit Detector" : "Fit & Score"}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

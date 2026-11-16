# TECHNICAL AUDIT — XAUD SYSTEM (V1 to V2 Transition)

**Project:** Explainable Anomaly Detection Framework (XAUD)  
**Audit Date:** November 2026  
**Auditor:** Principal Machine Learning Engineer & Systems Architect  
**Scope:** Complete Codebase (`src/`, `backend/`, `frontend/`, `tests/`, `configs/`)

---

## 1. Executive Summary

A comprehensive, line-by-line engineering and scientific audit of the existing XAUD codebase was conducted. The system demonstrates a high standard of statistical rigor compared to typical open-source anomaly detection projects—specifically in its rejection of naive min-max scaling, its verification of SHAP additivity against normalized score space, its honest out-of-sample holdout reporting with optimism gap tracking, and its prototype-based counterfactual recommendations.

However, to transition from a single-session demonstration tool to a **research-grade autonomous anomaly intelligence platform (XAUD 2.0)**, significant architectural, algorithmic, and operational gaps must be addressed:

1. **Model Diversity & Registry:** Only 4 detectors are supported (Isolation Forest, LOF, One-Class SVM, Elliptic Envelope). Fast histogram-based (HBOS), distance-based (k-NN), subspace/projection-based (PCA reconstruction error), and density/mixture-based (GMM, Robust Mahalanobis) detectors are missing.
2. **Ensemble & Consensus Engine:** Current ensembling merely averages normalized scores. It lacks model agreement counting, rank-based (Borda count) consensus aggregation, pairwise disagreement metrics (Jaccard / Cohen's Kappa), and adaptive model filtering.
3. **Data Intelligence & Profiling:** Preprocessing handles scaling and basic one-hot encoding, but lacks automatic schema type inference with user overrides, multicollinearity diagnostics (VIF / correlation conditioning), duplicate/near-constant profiling, and an evidence-based Dataset Health Score.
4. **Synthetic Benchmark Laboratory:** While `generate_sample_dataset` creates synthetic normal/anomalous mixtures, there is no controlled, multi-type anomaly injection engine (point/extreme, contextual, multivariate correlation shift, cluster anomalies) to benchmark detectors with known ground-truth difficulty curves.
5. **Drift & Monitoring Engine:** No facility exists to compare a baseline reference dataset against a current or incoming production batch (Population Stability Index, Kolmogorov-Smirnov distribution drift, Jensen-Shannon divergence, score drift).
6. **State & Reproducibility Management:** `backend/state.py` relies on an in-memory singleton. While thread-safe via locks, it lacks isolated experiment workspaces, persistence to disk/database, and execution lineage metadata.
7. **Explanation Fidelity & Multi-Method Explanations:** Explanations currently lean primarily on SHAP with fallback to perturbation. Non-tree models (e.g., PCA, LOF, k-NN) benefit from native reconstruction error contributions, distance decomposition, and systematic insertion-AUC curves.

---

## 2. Pipeline Trace & Component Analysis

```
[Raw Data / CSV Upload]
          │
          ▼
┌───────────────────────────────────────┐
│ 1. Data Loader & Schema Validator     │ --> Checks: min rows, types, nulls, constants
└─────────────────┬─────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────┐
│ 2. Preprocessing & Leakage Control    │ --> Scaler, Imputer, Encoder, Inverse-transform
└─────────────────┬─────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────┐
│ 3. Unsupervised Detector Training     │ --> BaseDetector subclasses, fit(X) without y
└─────────────────┬─────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────┐
│ 4. Score Normalization & Severity     │ --> Quantile / Rank normalization, Severity bands
└─────────────────┬─────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────┐
│ 5. Explainability & Counterfactuals   │ --> SHAP additivity check, fidelity, nearest normal
└─────────────────┬─────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────┐
│ 6. Out-of-Sample Holdout & Rigor      │ --> In-sample vs. held-out ROC/PR, optimism gap
└─────────────────┬─────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────┐
│ 7. API Routers & Next.js Presentation │ --> REST endpoints, Recharts UI, export
└───────────────────────────────────────┘
```

---

## 3. Mathematical, Statistical & Scientific Correctness Audit

### 3.1 Anomaly Scores are NOT Probabilities
- **Current State:** The system correctly defines normalized scores $s \in [0, 1]$ where $0$ is nominal and $1$ is extreme outlier.
- **Audit Finding:** In some UI views and schemas, scores could be colloquially misinterpreted as "probability of anomaly". Unsupervised models do NOT output posterior probabilities $P(\text{anomaly} \mid x)$ without explicit calibration against a reference null distribution or conformal prediction bounds.
- **Requirement for V2:**
  - Standardize API and UI terminology to "Anomaly Severity Index" or "Calibrated Anomaly Score".
  - Implement empirical tail-quantile / conformal p-values: $p(x) = \frac{1}{N+1} \left(1 + \sum_{i=1}^N \mathbb{I}(s_i \ge s(x))\right)$, measuring empirical rarity rather than Bayesian probability.

### 3.2 Preprocessing & Data Leakage Audit
- **Current State:** `src/data/preprocessing.py` implements `fit` and `transform` methods cleanly. In `holdout.py`, the train/test split occurs before model training.
- **Audit Finding:** In full pipeline runs, `fit_transform` was occasionally called on the complete uploaded batch before splitting into holdout subsets. This constitutes subtle **transformation leakage** (scaler means/variances and quantile thresholds computed over test rows).
- **Requirement for V2:**
  - Guarantee strict leakage isolation: fit preprocessing parameters strictly on the training partition ($X_{\text{train}}$), then apply `.transform()` to $X_{\text{test}}$.
  - Ensure threshold calculation (e.g., contamination quantile) is estimated strictly on $X_{\text{train}}$ and evaluated out-of-sample.

### 3.3 Score Normalization & Outlier Suppression
- **Current State:** Quantile-anchored normalization (`ScoreNormalizer`) uses the 1st and 99th percentiles to prevent single extreme outliers from squashing the entire distribution.
- **Audit Finding:** High mathematical validity. However, when ensembling multiple detectors, averaging raw normalized scores can still be skewed if detector distributions have differing kurtosis.
- **Requirement for V2:**
  - Introduce **Rank-based Borda consensus** and **Robust Rank Average** as primary ensemble methods alongside weighted score averaging.

### 3.4 SHAP Additivity & Unit Verification
- **Current State:** `src/explainability/shap_explainer.py` checks that $\sum \phi_i + \text{base} \approx \text{score}$.
- **Audit Finding:** Exceptional mathematical integrity. When tree models output in margin space or raw log-loss, XAUD rescales or flags unit mismatches rather than misleading users.
- **Requirement for V2:**
  - Extend verification to include **Insertion-AUC** (in addition to Deletion-AUC).
  - Define explicit marginal median/mode baseline replacement rather than arbitrary zero vectors when computing feature ablation.

---

## 4. Architectural & System Strengths

1. **Clean Abstract Base Class (`BaseDetector`):** Enforces `fit(X)`, `score_samples(X)`, `predict(X)`, and `get_params()` without label contamination.
2. **Typed Data Validation:** `DataValidator` checks row count, missing rates, near-zero variance columns, and high cardinality.
3. **Comprehensive Test Suite:** 197 automated pytest unit and integration tests passing with 0 errors.
4. **Production Build Cleanliness:** Next.js frontend builds with zero TypeScript errors or Turbopack warnings.
5. **No Blind Dependency Bloat:** Core ML code relies on standard scientific Python (`numpy`, `pandas`, `scipy`, `scikit-learn`, `shap`).

---

## 5. Prioritized Engineering Gaps & Remediation Plan

| ID | Priority | Module | Identified Gap | Remediation in XAUD 2.0 |
| :--- | :--- | :--- | :--- | :--- |
| **GAP-01** | **P0** | `src/models/` | Limited detector suite (4 models). | Implement **HBOS**, **k-NN Anomaly Detection**, **PCA Reconstruction Error**, **Robust Mahalanobis Distance**, and **Gaussian Mixture Model (GMM)** detectors with a unified capability registry. |
| **GAP-02** | **P0** | `src/detection/` | Ensemble lacks consensus and disagreement metrics. | Build `IntelligentEnsembleEngine` supporting rank consensus, pairwise Jaccard/Kappa disagreement, and model agreement counts. |
| **GAP-03** | **P1** | `src/data/` | Schema discovery is basic; no dataset health score. | Build `DataIntelligenceEngine` with automated type discovery (ID, datetime, categorical, binary, numeric), multicollinearity profiling, and an evidence-based Dataset Health Score. |
| **GAP-04** | **P1** | `src/evaluation/` | No synthetic anomaly injection testbed. | Build `SyntheticBenchmarkLaboratory` for controlled extreme, multivariate, contextual, and cluster anomaly injection with ROC/PR benchmarking. |
| **GAP-05** | **P1** | `src/evaluation/` | No dataset drift or monitoring. | Implement `DriftMonitor` computing PSI, Kolmogorov-Smirnov two-sample tests, JS divergence, and score drift between reference and production batches. |
| **GAP-06** | **P1** | `src/explainability/` | Counterfactuals lack immutable constraint enforcement. | Upgrade `CounterfactualEngine` with user-defined immutable features (e.g. IDs, timestamps), bounds constraints, and L1/L2 normalized intervention cost. |
| **GAP-07** | **P2** | `backend/state.py` | Global singleton state lacks experiment tracking. | Implement `ExperimentTracker` with unique run IDs, reproducibility fingerprints, parameter logging, and persistent artifact storage. |
| **GAP-08** | **P2** | `frontend/` | UI lacks views for Drift, Benchmark Lab, and Experiments. | Add dedicated navigation tabs and rich interactive visualization views for Drift Monitoring, Synthetic Benchmark Lab, and Experiment History. |

---

## 6. Audit Conclusion
The XAUD foundation is sound, robust, and mathematically conscientious. Implementing the prioritized remediation items in Phases 1 through 12 will elevate XAUD into a true research-grade anomaly intelligence platform.

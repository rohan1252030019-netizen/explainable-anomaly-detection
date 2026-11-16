# ARCHITECTURE SPECIFICATION — XAUD 2.0
## Autonomous Anomaly Intelligence Platform

**Version:** 2.0.0  
**Status:** Approved Architecture  
**Target Environment:** Local / Workstation / Server (Windows / Linux / macOS)  
**Primary Tech Stack:** Python 3.10+ (FastAPI, Scikit-learn, SciPy, NumPy, SHAP) + Next.js 15 (TypeScript, Tailwind CSS, Recharts)

---

## 1. System Vision & Architecture Principles

XAUD 2.0 shifts anomaly detection from a passive visualization script into an **autonomous, evidence-based anomaly intelligence platform**. Every alert produced by XAUD 2.0 must satisfy four foundational principles:

1. **Falsifiability:** Every flagged anomaly and associated explanation can be challenged through deletion/insertion ablation and out-of-sample holdout tests.
2. **Attribution Correctness:** Explanations must verify that $\sum \phi_i + \text{base} \approx \text{score}$ in the explained model's output space.
3. **Leakage Immunity:** Preprocessing parameters, score normalizer reference distributions, and threshold cutoffs are strictly computed on training partitions and evaluated out-of-sample.
4. **Actionable Remediation:** Anomalies are accompanied by constrained, feasible counterfactual recommendations in the original data units with estimated change costs.

---

## 2. High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               NEXT.JS 15 FRONTEND (PORT 3000)                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Navigation: Overview │ Data Intelligence │ Detection Lab │ Model Comparison │         │
│              Anomaly Explorer │ Investigation │ Counterfactuals │ Rigor & Holdout │    │
│              Drift Monitor │ Benchmark Lab │ Experiments │ Settings                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Primitives: ScoreGauge, StatCard, Badge, Toast, ChartFrame, ConfirmDialog, Tabs       │
│  Charts: PCA/Scatter, Score Histogram, Severity Donut, Waterfall, Deletion/Insertion   │
└───────────────────────────────────────────▲────────────────────────────────────────────┘
                                            │ REST API (JSON / Multipart)
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                               FASTAPI BACKEND (PORT 8000)                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Routers:                                                                              │
│    • /api/dataset        : Upload, schema profiling, health score, correlation        │
│    • /api/models         : Capability registry, hyperparameters, model info            │
│    • /api/detection      : Single & ensemble training, thresholding, severity          │
│    • /api/anomalies      : Filtered anomalies, ranking, investigation detail           │
│    • /api/explainability : SHAP, LOFO, Reconstruction error, Deletion/Insertion curves │
│    • /api/counterfactuals: Constrained optimization, nearest-normal in original units  │
│    • /api/evaluation     : In-sample vs. Held-out metrics, optimism gap                │
│    • /api/drift          : PSI, KS-test, Jensen-Shannon, score distribution drift      │
│    • /api/benchmarks     : Synthetic anomaly injection & sensitivity laboratory        │
│    • /api/experiments    : Lineage tracking, reproducibility fingerprints, artifact IO │
│    • /api/reports        : PDF / JSON / CSV summary generation                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  State & Service Orchestration Layer:                                                  │
│    • AppState (thread-safe session state)                                              │
│    • ExperimentTracker (run metadata, dataset hash, seed, parameters)                  │
│    • JobRunner (asynchronous task execution & progress tracking)                      │
└───────────────────────────────────────────▲────────────────────────────────────────────┘
                                            │ Python ML Engine
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                                  ML CORE ENGINE (`src/`)                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  1. Data Intelligence (`src/data/`):                                                   │
│     • DataLoader & Safe CSV/Parquet Parser                                             │
│     • SchemaDiscovery (Type inference: Numeric, Categorical, ID, Binary, Datetime)     │
│     • DataQualityProfiler (Missingness, Cardinality, Multicollinearity, Health Score)  │
│     • PreprocessingPipeline (Robust/Standard Scalers, Median/Mode Imputer, OHE)        │
│                                                                                        │
│  2. Anomaly Detection Models (`src/models/`):                                          │
│     • Tree-Based     : Isolation Forest Detector                                       │
│     • Proximity/Dist : k-Nearest Neighbors Outlier Detector (KNN)                      │
│     • Density-Based  : Local Outlier Factor (LOF) Detector                             │
│     • Linear/Subspace: PCA Reconstruction Error Detector, One-Class SVM (RBF/Linear)   │
│     • Statistical    : Histogram-Based Outlier Score (HBOS), Robust Mahalanobis (EE)  │
│     • Probabilistic  : Gaussian Mixture Model (GMM) Log-Likelihood Detector            │
│     • ModelManager   : Capability registry, serialization, hyperparameter management   │
│                                                                                        │
│  3. Scoring & Ensembling (`src/detection/`):                                           │
│     • ScoreNormalizer (Quantile-anchored, Rank/Borda, MinMax)                          │
│     • IntelligentEnsembleEngine (Consensus count, Model disagreement, Weighted avg)    │
│     • Thresholding (Contamination, Statistical sigma, IQR, Extreme Value Theory)       │
│     • SeverityTiers (CRITICAL, HIGH, MEDIUM, LOW) & Ranking                            │
│                                                                                        │
│  4. Explainability & Counterfactuals (`src/explainability/`):                          │
│     • SHAPExplainer (TreeExplainer, PermutationExplainer with Additivity Verification) │
│     • ReconstructionExplainer (PCA subspace feature error decomposition)               │
│     • ValidationEngine (Deletion-AUC & Insertion-AUC against random baseline)          │
│     • CounterfactualEngine (Nearest-normal search with immutable feature constraints)  │
│                                                                                        │
│  5. Scientific Validation & Monitoring (`src/evaluation/`):                           │
│     • HoldoutEvaluator (Leakage-free train/test split & optimism gap quantification)   │
│     • SyntheticInjectionLab (Extreme, multivariate, contextual anomaly benchmarks)     │
│     • DriftMonitor (PSI, Kolmogorov-Smirnov distribution drift, Score shift)           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Model Capability Registry
Every detector in XAUD 2.0 registers its computational and statistical capabilities:

```python
@dataclass(frozen=True)
class ModelCapability:
    name: str
    family: str  # Tree, Proximity, Density, Linear/Subspace, Statistical, Probabilistic
    time_complexity: str
    space_complexity: str
    supports_novelty: bool
    requires_scaling: bool
    sensitive_to_dim: bool
    explanation_methods: list[str]  # e.g., ["shap", "lofo", "reconstruction"]
    score_direction: str  # "lower_is_anomalous" (raw)
```

### 3.2 Intelligent Ensemble Consensus Engine
- **Consensus Count:** For a row $x$, $C(x) = \sum_{m \in M} \mathbb{I}(\text{prediction}_m(x) == -1)$.
- **Consensus Strength:** $\frac{C(x)}{|M|}$.
- **Pairwise Model Disagreement:**
  - Jaccard Distance between flagged binary anomaly sets: $1 - \frac{|A_i \cap A_j|}{|A_i \cup A_j|}$.
  - Rank correlation of anomaly scores: Spearman's $\rho(s_i, s_j)$.
- **Aggregation Strategies:**
  - `weighted_score`: $\sum_m w_m s_m(x)$.
  - `rank_borda`: $\frac{1}{|M|} \sum_m \text{rank}_m(x) / N$.
  - `consensus_filter`: Flagged only when $C(x) \ge k$.

### 3.3 Data Intelligence & Profiling Engine
- **Schema Discovery:** Automatically flags candidate primary keys (uniqueness $= 1.0$), categorical columns with low cardinality ($< 50$ unique values), datetime columns, and constant features.
- **Multicollinearity Diagnostic:** Computes pairwise correlation matrix $|R_{ij}| \ge 0.90$ and the condition number $\kappa(X^T X)$ of standardized numerical features.
- **Dataset Health Score (0 - 100):**
  $$\text{Score} = 100 - (w_{\text{null}} \times \text{missing\_ratio}) - (w_{\text{dup}} \times \text{dup\_ratio}) - (w_{\text{collin}} \times \text{collin\_ratio}) - (w_{\text{const}} \times \text{const\_features})$$

### 3.4 Scientific Validation & Synthetic Injection Lab
Controlled synthetic anomalies injected into unlabelled datasets with tracked boolean labels:
1. **Extreme Point Outliers:** Shifting selected numerical values by $k \cdot \sigma$ (e.g. $k \in [3, 6]$).
2. **Multivariate Correlation Anomalies:** Permuting correlated feature values to break joint covariance while keeping marginal distributions intact.
3. **Contextual Anomalies:** Setting values that are normal globally but anomalous given a categorical group.
4. **Cluster Anomalies:** Injecting dense micro-clusters in low-density feature space.
Provides exact benchmark metrics (ROC-AUC, Precision@k, Recall, PR-AUC) for controlled calibration.

### 3.5 Drift Monitoring Engine
- **Population Stability Index (PSI):**
  $$\text{PSI} = \sum_{b=1}^B (P_b - Q_b) \times \ln\left(\frac{P_b + \epsilon}{Q_b + \epsilon}\right)$$
  - $\text{PSI} < 0.1$: No significant drift.
  - $0.1 \le \text{PSI} < 0.25$: Moderate drift (investigate).
  - $\text{PSI} \ge 0.25$: Significant distribution shift.
- **Kolmogorov-Smirnov (KS) Test:** Non-parametric two-sample test comparing continuous feature distributions; reports statistic $D$ and $p$-value.
- **Score Drift:** Evaluates whether model anomaly scores on the production batch have shifted relative to the baseline training batch.

---

## 4. API Contract & Endpoint Definitions

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/dataset/profile` | `GET` | Return comprehensive schema discovery, quality metrics, and Health Score |
| `/api/models/registry` | `GET` | Return all detector capabilities and recommended configurations |
| `/api/detection/ensemble` | `POST` | Train multi-model ensemble with consensus & disagreement analysis |
| `/api/benchmarks/inject` | `POST` | Run synthetic anomaly injection experiment with ROC/PR benchmarking |
| `/api/drift/evaluate` | `POST` | Compare current dataset against baseline reference for PSI & KS drift |
| `/api/explainability/fidelity/{idx}` | `GET` | Compute Deletion-AUC & Insertion-AUC curve against random baseline |
| `/api/counterfactuals/constrained/{idx}` | `POST` | Compute nearest normal counterfactual under immutable feature constraints |
| `/api/experiments/history` | `GET` | Return list of reproducible experiment runs with parameters and metrics |

---

## 5. Security & Resource Boundaries
- Strict file validation: Content-type, MIME type, max file size (25 MB default), max row count (200k default).
- Zero eval/exec code execution on user input.
- Thread-safe state access via read/write locking semantics.
- Path traversal protection on all report downloads and model artifact paths.

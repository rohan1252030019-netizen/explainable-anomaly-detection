<p align="center">
  <img src="docs/sentinel-logo.svg" alt="SENTINEL Wordmark Logo" width="480"/>
</p>

# SENTINEL — Autonomous Anomaly Intelligence Platform

> **Tagline:** Detect. Explain. Verify. Act.  
> **SENTINEL:** *Scientific, Explainable, Next-gen Threat & Irregularity Neutralization and Evaluation Layer*  
> **Philosophy:** Mathematical correctness over flattering metrics. Explanation faithfulness over decorative charts. Evidence-backed conclusions over unverified claims.

---

## Executive Overview

**SENTINEL** transforms conventional anomaly detection dashboards into an enterprise research-grade, full-stack anomaly intelligence system. Designed specifically for complex tabular data operating **without pre-existing ground-truth labels**, SENTINEL independently detects, investigates, validates, explains, challenges, and recommends actionable remediation for anomalous records.

Unlike standard tools that report inflated in-sample scores or present arbitrary distance metrics as "posterior probabilities", SENTINEL maintains strict scientific integrity:
- **Zero Mock / Fake Metrics:** Unsupervised anomaly scores are relative ranking indices, never misrepresented as probabilities without calibrated conformal justification.
- **Strict Data Leakage Prevention:** Preprocessing transforms, scalers, imputers, and threshold quantiles are fitted strictly on baseline training splits and applied deterministically to holdout partitions.
- **Explanation Faithfulness:** Attribution importance is systematically audited using **Deletion-AUC** and **Insertion-AUC** perturbation curves.
- **Constrained Counterfactuals:** Actionable recommendations are computed in the dataset's **original physical units**, enforcing immutable column constraints and calculating explicit intervention costs.
- **Reproducible Experiment Tracking:** Every execution is cryptographically fingerprinted with **SHA-256 dataset hashes**, logged with full hyperparameter recipes in persistent structured JSON registries.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                SENTINEL PLATFORM                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  DATA INTELLIGENCE                                                              │
│  ├── Schema Discovery (Numerical, Categorical, Binary, Datetime, ID Candidates) │
│  ├── Quality Rubric (Missingness, Duplicates, Zero-Variance Penalties)          │
│  ├── Multicollinearity Matrix (Condition Number κ, High Pearson Pairs)          │
│  └── Dataset Health Score (0–100 Itemized Quality Score)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  DETECTION ENGINE (8 Research-Grade Models)                                     │
│  ├── Isolation Forest (Partition Depth O(N log N))                              │
│  ├── Local Outlier Factor (Local Reachability Density O(N²))                    │
│  ├── One-Class SVM (Kernel Hypersphere O(N²–N³))                                │
│  ├── Elliptic Envelope (Mahalanobis Robust Covariance O(N · P²))                 │
│  ├── k-Nearest Neighbors (k-th Neighbour Distance O(N log N))                   │
│  ├── PCA Reconstruction Error (Orthogonal Subspace Residual O(N · P²))          │
│  ├── HBOS (Linear-Time Inverse Log Histogram Density O(N · P))                  │
│  └── Gaussian Mixture Model (Multimodal EM Log-Likelihood O(N · K · P))         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  INTELLIGENT ENSEMBLE ENGINE                                                    │
│  ├── Score Normalization via Reference Quantiles                                │
│  ├── Rank Borda Count Aggregation (Distortion-Free Percentile Ranks)            │
│  ├── Consensus Strength & Voting Agreement                                      │
│  └── Pairwise Disagreement Matrix (Jaccard Distance & Spearman Rank Correlation)│
├─────────────────────────────────────────────────────────────────────────────────┤
│  SCIENTIFIC VALIDATION & DRIFT LAB                                              │
│  ├── Synthetic Injection Lab (Point, Multivariate Subspace, Micro-Clusters)     │
│  ├── Controlled Ground-Truth Benchmarks (ROC-AUC, PR-AUC, Precision@K)          │
│  ├── Covariate Drift Monitor (Population Stability Index - PSI)                 │
│  └── 2-Sample Kolmogorov-Smirnov Distribution Divergence Tests (D_stat, p-val)  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  EXPLAINABLE AI 2.0 & INVESTIGATION                                             │
│  ├── Multi-Method Explanations (SHAP, Permutation, PCA Feature Residuals)       │
│  ├── Deletion-AUC & Insertion-AUC Attribution Verification                      │
│  ├── Constrained Nearest-Normal Counterfactuals (Original Physical Units)       │
│  └── Immutable Feature Locking & Quantitative Intervention Cost                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  REPRODUCIBILITY & INTERFACES                                                   │
│  ├── Cryptographic SHA-256 Dataset Fingerprinting & Experiment Logs             │
│  ├── FastAPI Asynchronous REST Backend                                          │
│  └── Next.js 16 (Turbopack, TypeScript, Tailwind CSS, Recharts) Frontend        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Comparison

| Capability | Standard Anomaly Tools | SENTINEL Platform |
|---|---|---|
| **Anomaly Score Interpretation** | Often presented as "Probability of Anomaly" | Explicitly treated as relative ranking indices; normalized via reference quantiles |
| **Leakage Prevention** | Scaler fit across entire dataset before splitting | Strict pipeline isolation; scalers & thresholds fit strictly on reference splits |
| **Model Diversity** | Usually 1 or 2 basic algorithms | 8 distinct algorithmic perspectives (tree, density, kernel, covariance, distance, subspace, histogram, mixture) |
| **Ensemble Aggregation** | Flawed arithmetic average of uncalibrated scores | Non-parametric **Rank Borda Count** preventing kurtosis distortion |
| **Consensus Evidence** | Binary flag only | Consensus strength, agreeing model counts, and pairwise Jaccard disagreement matrix |
| **Benchmarking Without Labels** | Fabricates false ROC-AUC on unlabeled data | Controlled **Synthetic Anomaly Injection Lab** (multivariate subspace, point, cluster) |
| **Drift Monitoring** | None | Real-time **PSI** & **2-sample Kolmogorov-Smirnov** statistical distribution shifts |
| **Counterfactual Remediation** | None or abstract z-scores | Actionable deltas in **original units**, enforcing immutable column locks & intervention cost |
| **Attribution Verification** | Unchecked SHAP plots | Behavioural perturbation tests (**Deletion-AUC** & **Insertion-AUC**) |
| **Audit & Reproducibility** | Transient session state | Cryptographic **SHA-256 dataset fingerprinting** with persistent JSON experiment history |

---

## Detection Algorithms

1. **Isolation Forest (`Isolation Forest`)**: Recursive random partitioning identifying anomalies via shorter path lengths ($O(N \log N)$). Best for global outliers and high-dimensional spaces.
2. **Local Outlier Factor (`Local Outlier Factor`)**: Computes local reachability density against $k$-nearest neighbours ($O(N^2)$). Excels at detecting anomalies embedded in clusters of varying densities.
3. **One-Class SVM (`One-Class SVM`)**: Fits a maximum-margin hypersphere in kernel feature space ($O(N^2 - N^3)$). Ideal for complex non-linear boundary identification.
4. **Elliptic Envelope (`Elliptic Envelope`)**: Estimates robust covariance using the FastMCD estimator to compute Mahalanobis distances ($O(N \cdot P^2)$). Suitable for unimodal, approximately Gaussian distributions.
5. **k-Nearest Neighbors (`k-Nearest Neighbors`)**: Measures Euclidean distance to the $k$-th nearest neighbour in standardized space ($O(N \log N)$). Detects distance outliers without assuming parametric forms.
6. **PCA Reconstruction Error (`PCA Reconstruction`)**: Projects observations onto leading principal components and measures residual reconstruction error ($O(N \cdot P^2)$). Identifies observations that violate multivariate correlation constraints.
7. **Histogram-Based Outlier Score (`Histogram-Based Outlier Score`)**: Fast, linear-time ($O(N \cdot P)$) estimator assuming feature independence that inverts marginal log densities. Ideal for high-throughput streaming and massive datasets.
8. **Gaussian Mixture Model (`Gaussian Mixture Model`)**: Fits an Expectation-Maximization mixture of Gaussians ($O(N \cdot K \cdot P)$) and scores observations by negative log-likelihood. Models multi-modal cluster distributions cleanly.

---

## Intelligent Ensemble Engine

Different detectors embody fundamentally different statistical assumptions (e.g. density vs subspace vs isolation). A simple average of raw detector scores is mathematically invalid because raw scores possess vastly different kurtosis, scale, and dispersion.

XAUD 2.0 resolves this with the **Intelligent Ensemble Engine**:
- **Empirical Rank Normalization:** Scores from each detector are transformed into empirical percentile ranks ($[0, 1]$).
- **Rank Borda Aggregation:** The consensus score is computed as the Borda rank sum across detectors, preserving relative ordering without scale distortion:
  $$\text{Score}_{\text{Borda}}(x) = \sum_{m=1}^{M} w_m \cdot \text{Rank}_m(x)$$
- **Model Disagreement & Consensus Strength:** Reports the proportion of models agreeing on anomaly classifications and computes pairwise Jaccard distance:
  $$J(A_i, A_j) = 1 - \frac{|A_i \cap A_j|}{|A_i \cup A_j|}$$

---

## Controlled Synthetic Benchmark Lab

Unsupervised learning lacks ground truth. Fabricating accuracy or ROC-AUC scores on unlabeled data is statistically fraudulent.

XAUD's **Synthetic Anomaly Injection Laboratory** provides reproducible benchmarking by injecting controlled, traceable perturbations:
- **Multivariate Subspace Violations:** Perturbs joint feature covariance while preserving marginal 1D distributions.
- **Extreme Point Outliers:** High-magnitude displacements ($> 4 \times \text{IQR}$) in isolated dimensions.
- **Micro-Cluster Outliers:** Tightly clustered synthetic anomalies in low-density subspaces.

Detectors are evaluated on genuine binary labels using **ROC-AUC**, **PR-AUC** (vital for severe class imbalance), and **Precision@K**.

---

## Covariate & Score Drift Monitoring

In production, data distributions shift over time (covariate shift), degrading detector reliability.

XAUD monitors distribution stability using:
- **Population Stability Index (PSI):**
  $$\text{PSI} = \sum_{b=1}^{B} (P_b - Q_b) \cdot \ln\left(\frac{P_b}{Q_b}\right)$$
  - $\text{PSI} < 0.10$: Distribution is stable.
  - $0.10 \le \text{PSI} < 0.25$: Moderate drift; monitor ongoing batches.
  - $\text{PSI} \ge 0.25$: Significant covariate shift; triggers model recalibration alerts.
- **Two-Sample Kolmogorov-Smirnov Test:** Non-parametric evaluation of empirical cumulative distribution divergence ($D_{KS}$ and $p$-value).

---

## Installation & Quickstart

### Real fraud dataset

The repository ships with a synthetic demo set and a script that stages the real
**ULB credit-card fraud** dataset (284,807 real transactions, 492 frauds) plus a
20,000-row app sample that keeps every fraud:

```bash
python scripts/prepare_ulb_dataset.py          # download + verify + stage
python scripts/run_ulb_benchmark.py            # all 8 detectors, time split
python scripts/smoke_ulb_api.py                # full HTTP surface, in-process
```

Provenance, licences, measured PR-AUC/ROC-AUC per detector and the caveats that
must be quoted alongside them are in [`docs/DATASETS.md`](docs/DATASETS.md).

### Prerequisites
- Python 3.11+ (the floor declared by `requires-python` in `pyproject.toml`)
- Node.js 20+ & npm

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/rohan1252030019-netizen/explainable-anomaly-detection.git
cd explainable-anomaly-detection

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at `http://localhost:8000/docs`.

### 2. Frontend Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Build & Run Next.js with Turbopack
npm run dev
```
Dashboard will be live at `http://localhost:3000`.

### 3. Run Test Suite
```bash
# Backend: unit, API, scientific-validity and packaging tests
pytest tests/

# Frontend: lint, type check, unit/component tests and E2E
cd frontend
npm run lint
npm run typecheck
npm test
npm run test:e2e
```

The wheel can be built and verified in isolation with:

```bash
python -m pip wheel . --no-deps -w dist
python -c "import zipfile, glob; print(len(zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist()))"
```

---

## Project Structure

```
explainable-anomaly-detection/
├── backend/                  # FastAPI backend
│   ├── main.py               # Application entrypoint & middleware
│   ├── state.py              # Session object: construction, defaults, reset
│   ├── services/             # State behaviour, split by concern
│   │   ├── dataset.py        # Sample/CSV ingestion, preprocessing, profiling
│   │   ├── detection.py      # Single-model and ensemble scoring
│   │   ├── explainability.py # Local/global explanations and counterfactuals
│   │   ├── evaluation.py     # Leakage-safe held-out evaluation
│   │   ├── analytics.py      # PCA, correlations, benchmarks, drift, lab
│   │   ├── reporting.py      # Audit report assembly & experiment history
│   │   └── base.py           # Shared state typing and the mutation lock
│   ├── session.py            # LRU-bounded session registry for isolation
│   ├── config.py             # Environment-driven settings (CORS, auth, limits)
│   ├── security.py           # Optional API-key authentication
│   ├── schemas.py            # Pydantic v2 request/response models
│   ├── experiments.py        # SHA-256 experiment tracker
│   └── routers/              # Modular API endpoints
│       ├── dataset.py        # Upload, profile, preprocess
│       ├── detection.py      # Run detection & ensemble
│       ├── anomalies.py      # Table queries & single-observation investigation
│       ├── explainability.py # Global SHAP, stability, feature distributions
│       ├── drift.py          # PSI & KS test drift evaluation
│       ├── benchmarks.py     # Synthetic anomaly lab benchmarks
│       ├── experiments.py    # Persistent audit history
│       ├── evaluation.py     # Out-of-sample holdout metrics
│       └── reports.py        # Audit summary generation
├── src/                      # Core ML & Mathematical Engine
│   ├── data/                 # Profiler & leakage-safe preprocessing
│   ├── models/               # 8 Unsupervised anomaly detectors & registry
│   ├── detection/            # Intelligent ensemble & thresholding
│   ├── explainability/       # Faithfulness validation & constrained counterfactuals
│   └── evaluation/           # Synthetic lab, drift monitor, holdout validator
├── frontend/                 # Next.js 16 Web Application
│   ├── src/
│   │   ├── app/              # Next.js App Router (page.tsx, layout.tsx)
│   │   ├── components/
│   │   │   ├── layout/       # Sidebar, Topbar
│   │   │   ├── ui/           # Design system (Cards, StatCards, Badges, Tabs)
│   │   │   ├── charts/       # SVG visualizations (down-sampled)
│   │   │   ├── views/        # All 12 application screens
│   │   │   │   ├── DatasetView.tsx        # Profile & Preprocessing
│   │   │   │   ├── DetectionView.tsx      # 8 Models & Intelligent Ensemble
│   │   │   │   ├── DriftMonitorView.tsx   # PSI & KS Drift Diagnostics
│   │   │   │   ├── BenchmarkLabView.tsx   # Synthetic Anomaly Lab
│   │   │   │   ├── ExperimentsView.tsx    # Reproducibility Registry
│   │   │   │   ├── InvestigateView.tsx    # Counterfactuals & Verification
│   │   │   │   ├── RigorView.tsx          # Holdout & Stability
│   │   │   │   ├── rigor/widgets.tsx      # Formatters & presentational primitives
│   │   │   │   └── ...
│   │   ├── context/          # App-wide state with cancellable fetches
│   │   └── lib/              # Typed API client, down-sampling, navigation
│   ├── e2e/                  # Playwright tests against a production build
│   └── src/**/__tests__/     # Vitest unit & component tests
└── tests/                    # 300 backend unit, scientific and API tests
```

---

## License

This project is licensed under the Apache 2.0 License.

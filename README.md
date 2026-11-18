<p align="center">
  <img src="docs/sentinel-logo.svg" alt="SENTINEL Wordmark Logo" width="520"/>
</p>

# SENTINEL — Autonomous Anomaly Intelligence Platform

> **Tagline:** Detect. Explain. Verify. Act.  
> **SENTINEL:** *Scientific, Explainable, Next-gen Threat & Irregularity Neutralization and Evaluation Layer*  
> **Core Philosophy:** Mathematical correctness over flattering metrics. Zero data leakage. Scores are ranking indices, not probabilities. Every claim backed by reproducible cryptographic evidence.

[![CI Status](https://github.com/rohan1252030019-netizen/explainable-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/rohan1252030019-netizen/explainable-anomaly-detection/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](backend/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16.3%20(Turbopack)-black.svg)](frontend/)
[![Pytest Suite](https://img.shields.io/badge/pytest-353%20passed%20(100%25)-brightgreen.svg)](tests/)
[![Vitest](https://img.shields.io/badge/vitest-36%20passed%20(100%25)-brightgreen.svg)](frontend/)

---

## Executive Overview

**SENTINEL** is an autonomous, research-grade anomaly intelligence platform engineered for complex tabular data operating **without pre-existing ground-truth labels**. While conventional anomaly tools rely on crude distance heuristics or present uncalibrated anomaly scores as "probabilities", SENTINEL treats anomaly detection as a rigorous, end-to-end scientific workflow:

1. **Detect**: 17 production algorithms spanning classical, copula, ensemble, deep autoencoders, and supervised gradient-boosted surrogates.
2. **Explain**: Faithful multi-method attribution (TreeSHAP, KernelSHAP, LIME), pairwise 2nd-order non-linear interaction coupling, and automated natural-language forensic investigation narratives.
3. **Verify**: Finite-sample conformal $p$-values, Benjamini-Hochberg False Discovery Rate (FDR) control, Deletion/Insertion-AUC faithfulness curves, and sub-population fairness/robustness audits.
4. **Act**: Diverse, actionable counterfactual recommendations computed strictly in original physical units with $L_0$ sparsity and LOF distribution plausibility checks; human-in-the-loop active learning triage with zero-leakage semi-supervised retraining.

---

## End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Preprocessing"]
        A[Raw Tabular CSV / Streams] --> B[Schema Profiler & Quality Rubric]
        B --> C[Leakage-Safe Preprocessor]
        C -->|Fit on Reference Only| D[Standardized Matrix X]
        C -->|SHA-256 Digest| E[Cryptographic Fingerprint]
    end

    subgraph DetectionEngine ["2. Multi-Paradigm Detection Engine"]
        D --> F1[Classical Models: IForest, LOF, OCSVM, EE, kNN, PCA, HBOS, GMM, ECOD, COPOD, Mahalanobis, DBSCAN, EIF]
        D --> F2[Deep Learning: Autoencoder, VAE, Deep SVDD]
        D --> F3[Supervised / Stacking: XGBoost Surrogate, Classifier, Hybrid Stacker]
        F1 & F2 & F3 --> G[Rank-Borda Ensemble Consensus]
    end

    subgraph Uncertainty ["3. Calibrated Uncertainty & FDR"]
        G --> H[Split / Mondrian Conformal Calibration]
        H --> I[Finite-Sample Valid p-Values]
        I --> J[Benjamini-Hochberg FDR Control q <= 0.05]
    end

    subgraph Explainability ["4. Advanced Explainability & Verification"]
        J --> K1[TreeSHAP & KernelSHAP Attribution]
        J --> K2[Pairwise 2nd-Order Feature Interactions]
        J --> K3[LIME Cross-Method Agreement Jaccard / Spearman]
        K1 --> L1[Deletion-AUC & Insertion-AUC Faithfulness]
        K1 --> L2[Constrained & Diverse DiCE Counterfactuals in Physical Units]
        K1 --> L3[Automated Investigation Forensic Narrative]
    end

    subgraph Operations ["5. Streaming, Active Learning & Governance"]
        D --> M1[Online Half-Space Trees & Sliding PSI Drift]
        M1 -->|PSI >= 0.20| M2[Human Recalibration Approval Gate]
        J --> N1[Active Learning Triage Queue: Uncertainty + Disagreement]
        N1 --> N2[Analyst Feedback Store 0/1]
        N2 --> N3[Semi-Supervised Retraining on Independent Holdout]
        K1 --> O1[HDBSCAN SHAP Root-Cause Failure Archetypes]
        O1 --> P1[Mitchell et al. Model Cards & Versioned Registry]
        P1 --> P2[Prometheus Metrics, Rate Limiter & Standalone Audit Reports]
    end
```

---

## The SENTINEL Scientific Philosophy (Non-Negotiable Rules)

1. **Zero Fake Metrics:** In unsupervised domains where labels are absent, true ROC-AUC and PR-AUC are mathematically undefined. SENTINEL never fabricates mock metrics. Precision, recall, and ROC-AUC are produced exclusively on genuine historical holdouts or controlled synthetic injection benchmarks with clear `"synthetic benchmark"` disclosures.
2. **Strict Zero Data Leakage:** Preprocessing transformations, imputers, quantile scalers, and decision thresholds are fitted strictly on baseline/reference splits and applied forward without peeking. Pipeline tests guarantee no future information leaks into training.
3. **Calibrated Conformal Uncertainty:** Anomaly scores are ranking indices, not probabilities. When probabilistic guarantees are required, SENTINEL computes conformal $p$-values with finite-sample coverage guarantees ($P(p \le \alpha) \le \alpha$) alongside Benjamini-Hochberg FDR bounds.
4. **Actionable, In-Distribution Counterfactuals:** Recourse recommendations are presented in the dataset's original units, enforce immutable feature locks (e.g., account age, timestamp), minimize intervention cost, and verify through LOF density checks that counterfactual states fall within valid data distributions.
5. **Cryptographic Provenance:** Every model execution logs the full dataset SHA-256 fingerprint, hyperparameters, environment state, and random seed to ensure 100% reproducible execution history.

---

## 20-Detector Model Zoo

SENTINEL incorporates **20 registered production detectors** covering five distinct mathematical paradigms. Detectors marked with `†` are evaluated in the ULB Credit Card Fraud benchmark table below:

| Category | Detector Name | Registry Key | Benchmark Table | Algorithmic Formulation | Computational Complexity | Compute Profile |
|---|---|---|:---:|---|---|---|
| **Classical & Distance** | Isolation Forest | `Isolation Forest` | Yes † | Recursive random hyperplane partitioning; path depth | $O(N \log N)$ | Low |
| **Classical & Distance** | Local Outlier Factor | `Local Outlier Factor` | Yes † | Local reachability density relative to $k$-nearest neighbors | $O(N^2)$ | Medium |
| **Classical & Distance** | One-Class SVM | `One-Class SVM` | Yes † | Kernel max-margin boundary separating dense origin | $O(N^2 - N^3)$ | High |
| **Classical & Distance** | Elliptic Envelope | `Elliptic Envelope` | Yes † | FastMCD robust covariance Gaussian Mahalanobis distance | $O(N \cdot P^2)$ | Low |
| **Classical & Distance** | $k$-Nearest Neighbors | `k-Nearest Neighbors` | Yes † | Distance to $k$-th nearest neighbor in standardized space | $O(N^2)$ | Medium |
| **Classical & Distance** | PCA Reconstruction | `PCA Reconstruction` | Yes † | Orthogonal residual subspace reconstruction error | $O(N \cdot P^2)$ | Low |
| **Classical & Distance** | Histogram-Based Outlier Score (HBOS) | `Histogram-Based Outlier Score` | Yes † | Inverse log marginal histogram density (parameter-free) | $O(N \cdot P)$ | Ultra-Low |
| **Classical & Distance** | Gaussian Mixture Model | `Gaussian Mixture Model` | Yes † | Expectation-Maximization negative log-likelihood | $O(N \cdot K \cdot P)$ | Medium |
| **Classical & Fast Subspace** | ECOD | `ECOD` | No | Empirical Cumulative Distribution Function tail probabilities | $O(N \cdot P)$ | Ultra-Low |
| **Classical & Fast Subspace** | COPOD | `COPOD` | No | Empirical copula transform with joint tail probabilities | $O(N \cdot P)$ | Ultra-Low |
| **Classical & Fast Subspace** | Shrinkage Mahalanobis | `Shrinkage Mahalanobis` | No | Covariance shrinkage distance robust to collinearity ($\kappa < 30$) | $O(N \cdot P^2)$ | Low |
| **Classical & Fast Subspace** | DBSCAN Noise Points | `DBSCAN Noise` | No | Density-based spatial clustering core-point noise isolation | $O(N \log N)$ | Medium |
| **Classical & Fast Subspace** | Extended Isolation Forest | `Extended Isolation Forest` | No | Random hyperplane cuts at arbitrary slopes (resolves axis bias) | $O(N \log N)$ | Medium |
| **Classical & Fast Subspace** | SUOD Fast Ensemble | `SUOD Fast Ensemble` | No | Subspace fast approximation with Borda consensus | $O(N \cdot P)$ | Low |
| **Deep Learning** | PyTorch Autoencoder | `Deep Autoencoder` | No | Symmetric bottleneck neural network reconstruction MSE | $O(E \cdot N)$ | Medium (CPU) |
| **Deep Learning** | Variational Autoencoder | `Variational Autoencoder` | No | Probabilistic encoder-decoder optimized via ELBO loss | $O(E \cdot N)$ | Medium (CPU) |
| **Deep Learning** | Deep SVDD | `Deep SVDD` | No | Neural representation mapping into minimum-volume hypersphere | $O(E \cdot N)$ | Medium (CPU) |
| **Supervised & Surrogate** | Pseudo-Label Surrogate | `Pseudo-Label Surrogate` | No | XGBoost surrogate trained on out-of-fold pseudo-labels | $O(T \cdot N \cdot P)$ | Low |
| **Supervised & Surrogate** | Supervised XGBoost | `Supervised XGBoost` | No | Focal-weighted classifier with cost-matrix thresholding | $O(T \cdot N \cdot P)$ | Medium |
| **Supervised & Surrogate** | Hybrid Stacking Meta-Learner | `Hybrid Stacking Meta-Learner` | No | Boosted meta-learner over out-of-fold unsupervised scores | $O(K \cdot N \cdot (M + T \cdot P))$ | Medium |

*(Note: The Consensus Ensemble evaluated in the benchmark table aggregates the 8 Classical detectors via Rank Borda count).*

*Deep learning models run on CPU by default with zero GPU requirements, supported by optional PyTorch extras (`pip install -e .[deep]`).*

---

## Supervised & Surrogate Gradient Boosting (Feature 1)

For high-throughput inference and TreeSHAP acceleration, SENTINEL provides three dedicated gradient boosted workflows (`src/models/supervised/`):

1. **Pseudo-Label Surrogate Model (`surrogate.py`)**: Trains an XGBoost classifier on ensemble consensus pseudo-labels using **$K$-Fold Out-Of-Fold (OOF)** prediction. Scalers and thresholds are fit strictly inside training folds. Delivers sub-millisecond inference on streaming rows with exact TreeSHAP attribution and surrogate fidelity reports (Spearman rank correlation, Top-$K$ overlap).
2. **Supervised Classifier (`classifier.py`)**: Deployed when ground-truth labels are present. Employs focal loss and class-weighting (`scale_pos_weight`), stratified time-aware cross-validation, PR-AUC, ROC-AUC, Brier score, Expected Calibration Error (ECE), and threshold selection derived from asymmetric cost matrices:
   $$\text{Cost} = C_{FP} \cdot FP + C_{FN} \cdot FN$$
3. **Hybrid Stacking Meta-Learner (`stacking.py`)**: Stacks the rank-normalized out-of-fold scores of the unsupervised detectors as meta-features for an XGBoost meta-estimator, capturing non-linear consensus patterns.
4. **Optuna Bayesian Tuning (`tuning.py`)**: Nested cross-validation hyperparameter search with fixed seeds, logging parameters and Pareto fronts into the experiment registry.

---

## Empirical Benchmarks on Real-World Datasets

### 1. ULB Credit Card Fraud Benchmark Sample (20,000 Transactions, 492 Frauds)

Evaluated under the **honest time-split protocol**: sorted chronologically by `Time`, fitted on the initial 70% window (14,000 transactions), and scored on the remaining 30% holdout window (6,000 transactions). Feature scaling and decision thresholds are computed strictly on the reference window ($Class$ column excluded during training; contamination set to training window prevalence of 2.74%).

**Dataset Attribution & Sampling Provenance**:
* **Source Dataset**: Real credit card transactions made by European cardholders in September 2013, curated by the Machine Learning Group at Université Libre de Bruxelles (MLG-ULB) and Worldline (Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi, *"Calibrating Probability with Undersampling for Unbalanced Classification"*, IEEE SSCI, 2015).
* **Sample vs. Full Dataset**: The full dataset contains 284,807 transactions with 492 frauds (**0.172% fraud rate**). To satisfy memory, compute, and quadratic distance budgets for all detectors (e.g. LOF row limits), `scripts/prepare_ulb_dataset.py` creates a deterministic stratified subsample (`data/sample/creditcard_fraud_sample.csv`, SHA-256: `ede5aa5ca3656a79e724735dee2b02085fdcf6c60aa37d9d9792cbdfd12e7907`) of 20,000 rows containing **all 492 frauds and 19,508 sampled normal records** (**2.46% sample fraud rate**).
* **Random-Baseline PR-AUC**: The random classifier baseline PR-AUC equals the test holdout class prevalence: **0.0180 (1.80%)** (108 frauds in 6,000 test transactions).
* **Comparability Notice**: Because the sample fraud prevalence (2.46%) is ~14× higher than the full dataset (0.172%), **PR-AUC numbers reported below are NOT directly comparable to published literature evaluated on the full 284k dataset**.

```bash
# Run on the included 20k sample:
python scripts/run_ulb_benchmark.py

# Or download the full dataset (creditcard.csv) from Kaggle and run:
python scripts/run_ulb_benchmark.py --data path/to/creditcard.csv
```

For statistical guarantees, assumptions, and exchangeability limitations, see [Statistical Assumptions & Limitations](docs/STATISTICAL_ASSUMPTIONS.md).

| Detector | PR-AUC | ROC-AUC | Precision | Recall | Precision@Top-1% | Fit Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| **Histogram-Based Outlier Score (HBOS)** | **0.3765** | 0.9343 | 0.3589 | **0.6944** | **0.5000** | **0.02** |
| **Isolation Forest** | 0.3108 | 0.9340 | **0.3867** | 0.5370 | 0.4167 | 0.28 |
| **Elliptic Envelope** | 0.1983 | 0.9028 | 0.1856 | 0.1667 | 0.1333 | 1.96 |
| **$k$-Nearest Neighbors** | 0.1754 | 0.9147 | 0.2081 | 0.2870 | 0.1333 | 0.20 |
| **Consensus Ensemble (Rank Borda)** | 0.1634 | 0.9033 | 0.2450 | 0.3810 | 0.3000 | 0.85 |
| **One-Class SVM** | 0.1543 | 0.8792 | 0.1860 | 0.4167 | 0.2667 | 0.70 |
| **PCA Reconstruction** | 0.1528 | 0.8414 | 0.2035 | 0.3241 | 0.3000 | 0.01 |
| **Gaussian Mixture Model** | 0.1510 | 0.8281 | 0.2360 | 0.3889 | 0.1500 | 0.75 |
| **Local Outlier Factor (LOF)** | 0.0510 | 0.6743 | 0.0844 | 0.1852 | 0.1667 | 0.21 |

> [!IMPORTANT]
> **Ensemble Honesty Finding**: The Consensus Ensemble (Rank Borda) achieved a PR-AUC of **0.1634**, **failing to beat either HBOS (0.3765) or Isolation Forest (0.3108)**. Ensembling is not a magic bullet: indiscriminately averaging rank orders with under-performing density/distance estimators (e.g., LOF with PR-AUC 0.0510) severely degraded the final consensus ranking. In production, practitioners should prune low-performing algorithms before ensembling.

> [!NOTE]
> **Protocol Matters More Than the Model**: Under a naive stratified random split, the same Isolation Forest yields **PR-AUC 0.6471** and **ROC-AUC 0.9426**. Random splitting leaks future transaction patterns across identical days into the training set. SENTINEL defaults to strict time-aware splitting to reflect true production deployment performance.

---

## Honest Technical Limitations & Governance Disclaimers

1. **Non-Causal Attribution Disclaimers**: SHAP, LIME, and permutation importances quantify **correlational attribution within the trained model's decision surface**, NOT physical real-world causation. A high SHAP attribution value for `transaction_amount` indicates the model utilized that feature to flag the observation, not that modifying the amount will causally prevent fraudulent behavior.
2. **Exchangeability Assumption in Conformal Guarantees**: Finite-sample conformal validity guarantees require that reference calibration samples and test points are **exchangeable** (identically distributed). Under significant non-stationary concept drift ($\text{PSI} \ge 0.25$), conformal bounds degrade. SENTINEL includes real-time PSI drift gates to alert operators when exchangeability is broken.
3. **High-Dimensional Distance Concentration**: Density- and distance-based detectors ($k$-NN, LOF, DBSCAN) suffer from the curse of dimensionality when $P > 50$, as pairwise Euclidean distances converge to identical values. In high dimensions, subspace and projection methods (PCA, Isolation Forest, ECOD, Autoencoders) are recommended.
4. **Computational Complexity at Scale**: Algorithms with $O(N^2)$ complexity (LOF, $k$-NN, OCSVM) are constrained by a default ceiling of $N = 20,000$ rows. For datasets with $N > 100,000$, SENTINEL automatically triggers sub-sampling or recommends linear-time estimators (HBOS, ECOD, COPOD, SUOD).
5. **Label Noise in Real-World Benchmarks**: In production credit fraud benchmarks (such as ULB), a label of `Class = 0` denotes that no chargeback or dispute was filed within the observation window; it does not constitute mathematical proof of legitimacy.

---

## Installation & Quickstart

### Prerequisites
- Python 3.11+
- Node.js 20+ & npm (for Web UI)
- Docker & Docker Compose (optional for containerized deployment)

### 1. Local Setup

```bash
# Clone the repository
git clone https://github.com/rohan1252030019-netizen/explainable-anomaly-detection.git
cd explainable-anomaly-detection

# Create and activate Python virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Optional: Install deep learning and gradient boosting extras
pip install -e ".[deep,supervised,optuna]"

# Launch the FastAPI backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be accessible at `http://localhost:8000/docs`.

### 2. Frontend Web Interface

```bash
cd frontend

# Install Node dependencies
npm install

# Start Next.js development server with Turbopack
npm run dev
```
Open `http://localhost:3000` to launch the SENTINEL intelligence console.

### 3. Docker Deployment

Deploy the complete stack (FastAPI Backend + Next.js Frontend + Prometheus Monitoring) with Docker Compose:

```bash
docker-compose up --build
```
- Web Console: `http://localhost:3000`
- API Surface: `http://localhost:8000/docs`
- Prometheus Telemetry: `http://localhost:8000/metrics`

---

## API Surface (v1 & v2 Highlights)

SENTINEL exposes clean, modular, typed REST and WebSocket endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | System health, loaded dataset, and detector status |
| `/metrics` | `GET` | Prometheus telemetry (request count, inference latency, drift metrics) |
| `/api/v2/supervised/surrogate/train` | `POST` | Train K-fold OOF XGBoost surrogate with TreeSHAP and fidelity report |
| `/api/v2/supervised/stacking/train` | `POST` | Train hybrid stacking meta-learner using out-of-fold unsupervised scores |
| `/api/v2/conformal/calibrate` | `POST` | Calibrate split/Mondrian conformal non-conformity scores |
| `/api/v2/conformal/score` | `POST` | Compute conformal $p$-values and Benjamini-Hochberg FDR control |
| `/api/v2/explainability/interactions` | `POST` | Compute 2nd-order non-linear pairwise SHAP feature interactions |
| `/api/v2/explainability/agreement` | `POST` | Quantify cross-method explanation agreement (SHAP vs LIME) |
| `/api/v2/explainability/diverse-counterfactuals`| `POST` | Generate $K$ diverse counterfactuals with LOF plausibility & $L_0$ sparsity |
| `/api/v2/explainability/narrative` | `POST` | Generate forensic natural-language investigation report with SHA-256 fingerprint |
| `/api/v2/timeseries/changepoints` | `POST` | Detect regime shifts using PELT ($O(N)$) and Bayesian Online CPD |
| `/api/v2/streaming/ingest` | `POST` | Score streaming rows via Half-Space Trees with sliding PSI drift monitor |
| `/api/v2/streaming/recalibrate` | `POST` | Human-in-the-loop approval gate for streaming reference mass recalibration |
| `/api/v2/active-learning/queue` | `POST` | Active learning triage queue (uncertainty + ensemble disagreement sampling) |
| `/api/v2/active-learning/retrain-semi-supervised`| `POST` | Label spreading update with held-out, zero-leakage evaluation metrics |
| `/api/v2/root-cause/cluster` | `POST` | HDBSCAN density clustering on SHAP vectors to extract failure archetypes |
| `/api/v2/robustness/audit` | `POST` | Subgroup demographic parity gap and noise perturbation sensitivity audit |
| `/api/v2/mlops/model-card/generate` | `POST` | Auto-generate standardized Mitchell et al. Model Card |
| `/api/v2/mlops/registry/save` | `POST` | Save versioned model artifacts with cryptographic dataset hashes |
| `/api/v2/reports/html` | `GET` | Export complete standalone, audit-ready HTML investigation report |

---

## Test Suite & Verification

SENTINEL enforces rigorous automated quality gates:

```bash
# 1. Run complete Python test suite (353 unit, integration, and scientific tests)
pytest tests/ -v

# 2. Run Ruff linting and Mypy static type checking
python run.py --lint

# 3. Run frontend unit and component tests (36 Vitest tests)
npm --prefix frontend test

# 4. Compile optimized frontend production build
npm --prefix frontend run build
```

---

## Repository Governance & Suggested Topics

```
anomaly-detection  explainable-ai  xgboost  shap  conformal-prediction  fastapi  nextjs  mlops
```

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

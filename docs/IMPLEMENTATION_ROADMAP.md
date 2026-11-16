# IMPLEMENTATION ROADMAP — XAUD 2.0

**Project:** Explainable Anomaly Detection Framework (XAUD 2.0)  
**Execution Strategy:** Incremental, Test-Driven, Zero Regression, Production Real Implementations Only

---

## Stage-by-Stage Implementation Plan

### Stage 1: Repository Audit & Architecture Assessment
- [x] Full codebase audit of `src/`, `backend/`, `frontend/`, `tests/`.
- [x] Verification of existing test suite baseline (197/197 passing).
- [x] Verification of frontend production build (`npm run build` passing).
- [x] Creation of `docs/TECHNICAL_AUDIT.md`.
- [x] Creation of `docs/ARCHITECTURE_V2.md`.
- [x] Creation of `docs/IMPLEMENTATION_ROADMAP.md`.

---

### Stage 2: Data Intelligence & Profiling Engine
- **Files to Create/Modify:**
  - `src/data/profiler.py` (New): Automated schema discovery (numeric, categorical, datetime, ID candidates, binary), missingness patterns, high-cardinality flags, near-constant features, multicollinearity diagnostics (correlation condition number, pairwise collinear pairs), and evidence-based Dataset Health Score.
  - `backend/routers/dataset.py`: Add `/dataset/profile` endpoint returning the rich profiling report.
  - `backend/schemas.py`: Add Pydantic schemas for data intelligence profiling.
  - `tests/test_profiler.py`: Automated tests for schema discovery, multicollinearity, and health score mathematics.

---

### Stage 3: Expanded Model Suite & Capability Registry
- **Files to Create/Modify:**
  - `src/models/knn_detector.py` (New): k-Nearest Neighbors distance-based anomaly detector (average k-NN distance / k-th distance).
  - `src/models/pca_detector.py` (New): PCA Reconstruction Error subspace anomaly detector.
  - `src/models/hbos_detector.py` (New): Histogram-Based Outlier Score (HBOS) for fast, linear-time univariate histogram density aggregation.
  - `src/models/gmm_detector.py` (New): Gaussian Mixture Model log-likelihood density detector.
  - `src/models/base_detector.py`: Add capability metadata interface.
  - `src/models/model_manager.py`: Register new detectors and implement the Model Capability Registry.
  - `tests/test_new_models.py`: Unit tests for KNN, PCA, HBOS, and GMM detectors (fit, score, predict, input validation).

---

### Stage 4: Intelligent Ensemble & Consensus Engine
- **Files to Create/Modify:**
  - `src/detection/ensemble.py` (New): Multi-detector ensemble orchestration:
    - Consensus detection (agreement count $C(x)$).
    - Model disagreement metrics (pairwise Jaccard distance, Spearman rank correlation).
    - Aggregation strategies: weighted score averaging, rank-based Borda count consensus.
  - `backend/routers/detection.py`: Add `/detection/ensemble` endpoint and update `/models` router.
  - `backend/state.py`: Integrate ensemble execution and results caching.
  - `tests/test_ensemble.py`: Tests for consensus calculation, rank aggregation, and disagreement metrics.

---

### Stage 5: Advanced Explainability & Constrained Counterfactuals
- **Files to Create/Modify:**
  - `src/explainability/validation.py`: Add **Insertion-AUC** and marginal replacement baseline alongside Deletion-AUC.
  - `src/explainability/counterfactuals.py`: Add **immutable feature constraints** (exclude IDs, timestamps, specified protected features) and L1/L2 normalized intervention cost calculation.
  - `src/explainability/pca_explainer.py` (New): Feature-level reconstruction error decomposition for the PCA detector.
  - `backend/routers/explainability.py`: Expose `/explainability/fidelity/{idx}` (both deletion and insertion curves) and constrained counterfactual options.
  - `tests/test_fidelity_and_counterfactuals.py`: Unit tests for insertion curves and immutable constraints.

---

### Stage 6: Scientific Validation & Synthetic Anomaly Injection Lab
- **Files to Create/Modify:**
  - `src/evaluation/synthetic_lab.py` (New): Controlled anomaly injection engine:
    - Extreme-value anomalies.
    - Multivariate correlation shift anomalies.
    - Contextual anomalies.
    - Cluster anomalies.
    - Benchmark evaluator computing precision, recall, ROC-AUC, PR-AUC against injected masks across difficulty levels.
  - `backend/routers/evaluation.py` & `backend/routers/benchmarks.py` (New): Expose benchmark injection endpoints.
  - `tests/test_synthetic_lab.py`: Unit tests for controlled injection and benchmark evaluation.

---

### Stage 7: Drift & Monitoring Engine
- **Files to Create/Modify:**
  - `src/evaluation/drift.py` (New):
    - Population Stability Index (PSI) per feature and on anomaly scores.
    - Two-sample Kolmogorov-Smirnov (KS) test for continuous features.
    - Jensen-Shannon divergence for categorical features.
    - Score drift quantifier between baseline and production batches.
  - `backend/routers/drift.py` (New): Expose `/drift/evaluate` and `/drift/features` endpoints.
  - `tests/test_drift.py`: Unit tests for PSI, KS tests, and categorical drift math.

---

### Stage 8: Experiment Lineage & Reproducibility
- **Files to Create/Modify:**
  - `backend/experiments.py` (New): Experiment lineage tracker recording experiment ID, dataset SHA-256 fingerprint, model hyperparameters, random seeds, execution timing, and performance metrics.
  - `backend/routers/experiments.py` (New): Expose `/experiments/history` and `/experiments/{id}/export`.
  - `tests/test_experiments.py`: Tests for fingerprinting and experiment logging.

---

### Stage 9: Backend API Integration & Schema Validation
- **Files to Create/Modify:**
  - `backend/main.py`: Register new routers (`drift`, `benchmarks`, `experiments`).
  - `backend/schemas.py`: Add Pydantic v2 schemas for all new capabilities.
  - `backend/state.py`: Support multi-model ensemble state, drift reference batches, and benchmark results.

---

### Stage 10: Frontend Integration & Advanced Investigation Workflows
- **Files to Create/Modify:**
  - `frontend/src/lib/api.ts`: Add typed API methods for profiling, ensemble, drift, benchmarks, and experiments.
  - `frontend/src/lib/navigation.ts`: Add navigation tabs for Data Intelligence, Drift Monitor, Benchmark Lab, and Experiments.
  - `frontend/src/components/views/`:
    - `DataIntelligenceView.tsx` (New)
    - `DriftMonitorView.tsx` (New)
    - `BenchmarkLabView.tsx` (New)
    - `ExperimentsView.tsx` (New)
    - Update `DetectionView.tsx` and `InvestigateView.tsx` with ensemble consensus and insertion curves.
  - Run `npm run build` in `frontend/` to guarantee zero compile or TypeScript errors.

---

### Stage 11: Verification & Final Quality Assurance
- Run full pytest test suite across all new and existing tests.
- Verify test coverage and absence of regressions.
- Update `README.md` with comprehensive XAUD 2.0 capabilities, API documentation, and usage examples.

# SENTINEL — Statistical Assumptions, Guarantees & Limitations

This document details the rigorous mathematical foundations, formal assumptions, and known limitations of the statistical procedures implemented in **SENTINEL 3.0**.

---

## 1. Conformal Prediction & Calibrated Uncertainty

### 1.1 The Exchangeability Hypothesis
Split-conformal and Mondrian conformal anomaly detectors provide finite-sample marginal coverage guarantees:
$$\mathbb{P}(p(X_{n+1}) \le \alpha) \le \alpha$$
under the null hypothesis $H_0$ that reference calibration instances $Z_1, \dots, Z_n$ and future test instances $Z_{n+1}$ are **exchangeable** (drawn identically from an exchangeable joint distribution).

**Violation Warning:**
In real-world non-stationary streaming applications, concept drift, regime changes, or temporal dependency break the exchangeability assumption.
- When Population Stability Index ($\text{PSI}$) exceeds $0.25$, exchangeability is considered violated.
- Operators must trigger the automated recalibration protocol (`/api/v2/streaming/recalibrate`) when drift alarms fire.

### 1.2 Unsupervised Calibration Contamination
In purely unsupervised settings, the reference calibration split $D_{\text{cal}}$ is drawn from unlabelled data and may contain true undetected anomalies:
- **Theoretical consequence:** If $D_{\text{cal}}$ contains contaminated anomalies with high non-conformity scores, the empirical $(1-\alpha)$ quantile threshold $\hat{q}$ is shifted **upward (rightward)**.
- **Direction of Error:** This contamination makes conformal $p$-values **conservative** — the empirical Type-I false alarm rate is bounded strictly **below** $\alpha$ ($\text{FPR} \le \alpha$).
- The validity guarantee is never anti-conservative (it will not inflate false alarms beyond the budget).

### 1.3 Mondrian Subgroup Calibration & Sample Size
Conditional group validity requires sufficient calibration support per group $g$:
$$n_g \ge \left\lceil \frac{1}{\alpha} \right\rceil - 1$$
For $\alpha = 0.05$, a minimum of $n_g \ge 19$ reference points is required for group $g$ to reject the null at level $\alpha$.
- If $n_g < \left\lceil \frac{1}{\alpha} \right\rceil - 1$, the group falls back to the pooled global calibrator.
- SENTINEL explicitly tracks and reports all fallback groups in the API response (`fallback_groups`).

---

## 2. Pseudo-Label Provenance & Surrogate Models

### 2.1 Inductive Surrogate vs. Ground Truth
The XGBoost surrogate model (`src/models/supervised/surrogate.py`) is trained on pseudo-labels derived from the consensus ensemble:
- **Purpose:** Provide exact TreeSHAP explanations, millisecond inference, and feature attribution.
- **Provenance:** The surrogate is strictly an inductive approximation of the unsupervised ensemble, **never ground truth**.
- **Metrics Convention:** All performance metrics evaluating the surrogate against pseudo-labels are named `agreement_*` (e.g., `agreement_spearman`, `agreement_accuracy`, `agreement_pr_auc`, `agreement_f1`) with metadata flag `is_ground_truth=False`.

### 2.2 Refusal of Pseudo-Label Targets in Stacking
Hybrid stacking meta-learners (`src/models/supervised/stacking.py`) train a meta-classifier on out-of-fold scores from 8 unsupervised base models:
- **Circular Leakage Constraint:** Training a meta-learner on pseudo-labels derived from the same base models is mathematically circular and causes degenerate self-reinforcing leakage.
- **Hardening Policy:** `HybridStackingDetector` strictly refuses pseudo-label targets (`HTTP 422 Unprocessable Entity`), requiring either:
  1. A user-provided ground-truth label column (`label_column`), or
  2. Ground-truth benchmark labels generated in the synthetic injection lab.

---

## 3. Explainability Limitations

### 3.1 Non-Causal Attribution
SHAP (Shapley Additive exPlanations), LIME, and permutation importances compute **game-theoretic attribution with respect to the model's learned decision surface**:
- They quantify how much a feature value shifted the anomaly score relative to expected value.
- They **do not prove physical real-world causation**.
- High attribution on `transaction_amount` does not mathematically prove that modifying the amount prevents fraudulent behaviour.

### 3.2 Counterfactual Plausibility Bounds
Constrained counterfactuals (`src/explainability/counterfactuals.py`) find minimal feature adjustments $x \to x'$ such that $f(x') \le \tau$:
- Counterfactuals are subject to local density constraints ($k$-NN / GMM in-distribution checks) to prevent recommending out-of-distribution adversarial vectors.
- Actionability constraints (e.g. immutable attributes such as age or historical dates) must be respected.

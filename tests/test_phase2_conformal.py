"""
tests/test_phase2_conformal.py

Rigorous test suite for SENTINEL Phase 2: Calibrated Uncertainty and Conformal Prediction.

Verifies:
1. Exact finite-sample validity theorem: empirical false-positive rate (FPR) ≈ alpha on pure nulls.
2. Mondrian group-conditional validity across heterogeneous subgroups.
3. Benjamini-Hochberg False Discovery Rate (FDR) control and q-value monotonicity.
4. Strict anti-leakage isolation between calibration set and evaluation set.
5. FastAPI /api/v2/conformal REST endpoints.
"""

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from src.uncertainty.conformal import (
    ConformalAnomalyCalibrator,
    MondrianConformalCalibrator,
)
from src.uncertainty.fdr import benjamini_hochberg_fdr


def test_conformal_finite_sample_validity_on_pure_nulls():
    """
    MATHEMATICAL PROOF TEST:
    Under H_0 (pure nominal exchangeability), conformal p-values satisfy:
        P(p(X) <= alpha) <= alpha
    For N_test = 1,000 pure nominal test points and N_cal = 1,000 calibration points,
    the empirical False Positive Rate (FPR) must be statistically indistinguishable from alpha.
    """
    rng = np.random.RandomState(42)
    n_cal = 1000
    n_test = 1000

    # Null distribution: standard normal
    cal_scores = rng.normal(loc=0.0, scale=1.0, size=n_cal)
    test_scores = rng.normal(loc=0.0, scale=1.0, size=n_test)

    for alpha in [0.05, 0.10, 0.20]:
        calibrator = ConformalAnomalyCalibrator(alpha=alpha)
        calibrator.fit_calibration(cal_scores)

        p_values = calibrator.compute_p_values(test_scores)
        empirical_fpr = float(np.mean(p_values <= alpha))

        # Standard error of binomial proportion: sqrt(alpha * (1 - alpha) / n_test)
        se = np.sqrt(alpha * (1.0 - alpha) / n_test)
        margin = 3.0 * se  # 99.7% confidence interval

        # Empirical false positive rate must be bounded within theoretical window
        assert abs(empirical_fpr - alpha) <= margin, (
            f"FPR {empirical_fpr:.4f} deviated from target alpha {alpha} by more than 3 SE ({margin:.4f})"
        )


def test_conformal_calibrated_threshold_matches_alpha():
    """
    Verify calibrated non-conformity threshold flags exactly alpha fraction of calibration nulls.
    """
    rng = np.random.RandomState(123)
    cal_scores = rng.exponential(scale=2.0, size=500)

    for alpha in [0.05, 0.10]:
        calibrator = ConformalAnomalyCalibrator(alpha=alpha)
        calibrator.fit_calibration(cal_scores)

        # Predict on calibration points
        res = calibrator.predict(cal_scores)
        flagged_rate = float(np.mean(res.is_anomaly))

        # Under finite-sample quantile formula, flagged rate on cal should be <= alpha
        assert flagged_rate <= alpha + 0.02


def test_mondrian_group_conditional_validity():
    """
    Verify Mondrian conformal prediction guarantees validity conditionally within each group,
    even when groups have vastly different score variances and scales.
    """
    rng = np.random.RandomState(99)
    n_per_group = 800

    # Group A: Low variance scale
    scores_A_cal = rng.normal(loc=0.0, scale=1.0, size=n_per_group)
    scores_A_test = rng.normal(loc=0.0, scale=1.0, size=n_per_group)

    # Group B: High variance scale
    scores_B_cal = rng.normal(loc=5.0, scale=4.0, size=n_per_group)
    scores_B_test = rng.normal(loc=5.0, scale=4.0, size=n_per_group)

    cal_scores = np.concatenate([scores_A_cal, scores_B_cal])
    cal_groups = np.array(["group_A"] * n_per_group + ["group_B"] * n_per_group)

    test_scores = np.concatenate([scores_A_test, scores_B_test])
    test_groups = np.array(["group_A"] * n_per_group + ["group_B"] * n_per_group)

    alpha = 0.05
    mondrian = MondrianConformalCalibrator(alpha=alpha)
    mondrian.fit_calibration(cal_scores, cal_groups)

    res = mondrian.predict(test_scores, test_groups, alpha=alpha)

    # Verify each group conditionally respects Type-I error budget
    fpr_A = res["group_breakdown"]["group_A"]["anomaly_rate"]
    fpr_B = res["group_breakdown"]["group_B"]["anomaly_rate"]

    se = np.sqrt(alpha * (1.0 - alpha) / n_per_group)
    margin = 4.0 * se  # Finite-sample deviation window across calibration & test draws

    assert abs(fpr_A - alpha) <= margin, f"Group A conditional FPR {fpr_A} exceeds margin {margin}"
    assert abs(fpr_B - alpha) <= margin, f"Group B conditional FPR {fpr_B} exceeds margin {margin}"


def test_benjamini_hochberg_fdr_control():
    """
    Verify Benjamini-Hochberg guarantees FDR <= target_fdr on multiple tests.
    """
    rng = np.random.RandomState(42)

    # 900 True Nulls (uniform p-values), 100 True Anomalies (small p-values)
    p_null = rng.uniform(0.0, 1.0, 900)
    p_anom = rng.beta(0.1, 10.0, 100)  # Skewed towards 0
    p_values = np.concatenate([p_null, p_anom])
    true_labels = np.array([0] * 900 + [1] * 100)

    target_fdr = 0.05
    res = benjamini_hochberg_fdr(p_values, target_fdr=target_fdr)

    discoveries = np.asarray(res["is_discovery"])
    q_values = np.asarray(res["q_values"])

    # 1. Monotonicity check
    order = np.argsort(p_values)
    sorted_q = q_values[order]
    assert np.all(np.diff(sorted_q) >= -1e-12), "BH q-values must be non-decreasing with respect to p-values."

    # 2. FDR verification
    total_discoveries = int(np.sum(discoveries))
    if total_discoveries > 0:
        false_discoveries = int(np.sum(discoveries & (true_labels == 0)))
        empirical_fdr = false_discoveries / total_discoveries
        assert empirical_fdr <= target_fdr + 0.03, (
            f"Empirical FDR {empirical_fdr:.4f} violated target {target_fdr}"
        )


def test_no_leakage_between_calibration_and_test():
    """
    Ensure calibration scores and thresholds are calculated strictly without test data.
    """
    cal_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    calibrator = ConformalAnomalyCalibrator(alpha=0.10)
    calibrator.fit_calibration(cal_scores)

    orig_thresh = calibrator.threshold_at_alpha_

    # Scoring test scores must never modify internal calibration parameters
    test_scores = np.array([100.0, 200.0, 300.0])
    calibrator.compute_p_values(test_scores)

    assert calibrator.threshold_at_alpha_ == orig_thresh
    assert calibrator.n_calibration_ == 10


def test_api_v2_conformal_endpoints():
    """
    Test FastAPI /api/v2/conformal REST endpoints.
    """
    client = TestClient(app)

    # Load benchmark dataset first
    client.post("/api/dataset/sample")

    # 1. Calibrate
    r_cal = client.post(
        "/api/v2/conformal/calibrate",
        json={"alpha": 0.05, "split_ratio": 0.3},
    )
    assert r_cal.status_code == 200
    data_cal = r_cal.json()
    assert data_cal["is_calibrated"] is True
    assert data_cal["alpha"] == 0.05
    assert data_cal["confidence_level"] == 0.95
    assert "coverage_statement" in data_cal

    # 2. Check Status
    r_stat = client.get("/api/v2/conformal/status")
    assert r_stat.status_code == 200
    assert r_stat.json()["is_calibrated"] is True

    # 3. Score with Conformal & FDR
    r_score = client.post(
        "/api/v2/conformal/score",
        json={"alpha": 0.05, "target_fdr": 0.05},
    )
    assert r_score.status_code == 200
    data_score = r_score.json()
    assert "coverage_statement" in data_score
    assert "fdr_statement" in data_score
    assert len(data_score["items"]) > 0
    assert "conformal_p_value" in data_score["items"][0]
    assert "fdr_q_value" in data_score["items"][0]

    # 4. Standalone FDR Control
    r_fdr = client.post(
        "/api/v2/conformal/fdr",
        json={"p_values": [0.001, 0.01, 0.04, 0.2, 0.8], "target_fdr": 0.05},
    )
    assert r_fdr.status_code == 200
    data_fdr = r_fdr.json()
    assert len(data_fdr["q_values"]) == 5
    assert data_fdr["n_discoveries"] >= 1

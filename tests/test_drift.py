"""
tests/test_drift.py

Unit tests for DriftMonitor:
- PSI calculation
- Two-sample KS test on numerical features
- Categorical JS divergence
- Overall severity classification
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.drift import DriftMonitor


@pytest.fixture
def baseline_and_shifted_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 200

    # Reference dataset
    df_ref = pd.DataFrame({
        "stable_num": rng.normal(10, 2, n),
        "drifting_num": rng.normal(50, 5, n),
        "stable_cat": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "drifting_cat": rng.choice(["X", "Y"], n, p=[0.8, 0.2]),
    })

    # Current dataset (drifting_num has mean 75 instead of 50; drifting_cat inverted)
    df_cur = pd.DataFrame({
        "stable_num": rng.normal(10, 2, n),
        "drifting_num": rng.normal(75, 5, n),
        "stable_cat": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "drifting_cat": rng.choice(["X", "Y"], n, p=[0.1, 0.9]),
    })

    return df_ref, df_cur


def test_calculate_psi():
    monitor = DriftMonitor()
    rng = np.random.default_rng(42)

    # Identical distributions should have near-zero PSI
    a = rng.normal(0, 1, 500)
    b = rng.normal(0, 1, 500)
    psi_stable = monitor.calculate_psi(a, b)
    assert psi_stable < 0.10

    # Shifted distribution should have significant PSI
    c = rng.normal(5, 1, 500)
    psi_shifted = monitor.calculate_psi(a, c)
    assert psi_shifted > 0.50


def test_evaluate_drift_reports(baseline_and_shifted_dfs):
    df_ref, df_cur = baseline_and_shifted_dfs
    monitor = DriftMonitor()

    ref_scores = np.random.uniform(0.1, 0.4, 200)
    cur_scores = np.random.uniform(0.5, 0.9, 200)  # Drifted scores

    report = monitor.evaluate_drift(df_ref, df_cur, ref_scores=ref_scores, cur_scores=cur_scores)

    assert report["overall_severity"] in ("MODERATE", "HIGH")
    assert "drifting_num" in report["features_high_drift"] or "drifting_num" in report["features_moderate_drift"]
    assert report["score_drift"] is not None
    assert report["score_drift"]["drift_status"] == "HIGH"
    assert len(report["actions"]) > 0

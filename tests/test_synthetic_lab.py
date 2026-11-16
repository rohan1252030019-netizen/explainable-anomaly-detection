"""
tests/test_synthetic_lab.py

Unit tests for SyntheticInjectionLab:
- Injection of point, multivariate, and cluster anomalies
- Verified benchmark evaluation metrics (ROC-AUC, PR-AUC, Precision@k)
"""

import numpy as np
import pytest

from src.evaluation.synthetic_lab import SyntheticInjectionLab


@pytest.fixture
def clean_matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    # 200 normal samples with correlated features
    z = rng.normal(0, 1, size=(200, 2))
    X = np.zeros((200, 4))
    X[:, :2] = z
    X[:, 2] = z[:, 0] * 1.5 + z[:, 1]
    X[:, 3] = z[:, 0] - z[:, 1] * 2.0
    return X


def test_synthetic_lab_injection_types(clean_matrix: np.ndarray):
    lab = SyntheticInjectionLab(random_state=42)

    for a_type in ("point", "multivariate", "cluster"):
        res = lab.inject_anomalies(clean_matrix, anomaly_type=a_type, contamination=0.10)
        assert res["n_injected"] == 20
        assert res["injected_mask"].sum() == 20
        assert res["X_injected"].shape == clean_matrix.shape
        assert res["anomaly_type"] == a_type


def test_synthetic_lab_benchmark_execution(clean_matrix: np.ndarray):
    lab = SyntheticInjectionLab(random_state=42)
    models = ["Isolation Forest", "k-Nearest Neighbors", "PCA Reconstruction"]

    report = lab.benchmark_detectors(
        clean_matrix,
        anomaly_type="point",
        contamination=0.08,
        models_to_test=models,
    )

    assert report["n_samples"] == 200
    assert report["n_injected"] == 16
    assert len(report["results_table"]) >= 2

    # Check metric ranges in table
    top_entry = report["results_table"][0]
    assert 0.0 <= top_entry["roc_auc"] <= 1.0
    assert 0.0 <= top_entry["pr_auc"] <= 1.0
    assert 0.0 <= top_entry["precision_at_k"] <= 1.0

"""
tests/test_new_models.py

Unit tests for newly introduced XAUD 2.0 detectors:
- KNNDetector
- PCADetector
- HBOSDetector
- GMMDetector
- Model Capabilities in ModelManager
"""

import numpy as np
import pytest

from src.models.gmm_detector import GMMDetector
from src.models.hbos_detector import HBOSDetector
from src.models.knn_detector import KNNDetector
from src.models.model_manager import ModelManager
from src.models.pca_detector import PCADetector


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """100 normal samples + 5 injected outliers."""
    rng = np.random.default_rng(42)
    # Normal data lies primarily on a 2D subspace in 4D
    basis = rng.normal(0, 1, size=(100, 2))
    X_normal = np.zeros((100, 4))
    X_normal[:, 0] = basis[:, 0]
    X_normal[:, 1] = basis[:, 1]
    X_normal[:, 2] = basis[:, 0] + basis[:, 1] + rng.normal(0, 0.05, 100)
    X_normal[:, 3] = basis[:, 0] - basis[:, 1] + rng.normal(0, 0.05, 100)

    # Outliers deviate from subspace and density
    X_outliers = rng.uniform(8, 12, size=(5, 4))
    X = np.vstack([X_normal, X_outliers])
    return X, X_outliers


def test_knn_detector(synthetic_data: tuple[np.ndarray, np.ndarray]):
    X, X_outliers = synthetic_data
    detector = KNNDetector(n_neighbors=5, contamination=0.05)
    detector.fit(X)

    assert detector.is_fitted()
    scores = detector.score_samples(X)
    assert scores.shape == (105,)

    normal_scores = scores[:100]
    outlier_scores = scores[100:]
    assert np.mean(outlier_scores) < np.mean(normal_scores)

    preds = detector.predict(X)
    assert set(np.unique(preds)).issubset({-1, 1})
    assert np.sum(preds[100:] == -1) >= 4


def test_pca_detector(synthetic_data: tuple[np.ndarray, np.ndarray]):
    X, X_outliers = synthetic_data
    detector = PCADetector(n_components=2, contamination=0.05)
    detector.fit(X)

    assert detector.is_fitted()
    scores = detector.score_samples(X)
    assert scores.shape == (105,)

    normal_scores = scores[:100]
    outlier_scores = scores[100:]
    assert np.mean(outlier_scores) < np.mean(normal_scores)

    feat_errors = detector.feature_reconstruction_errors(X)
    assert feat_errors.shape == (105, 4)
    assert np.all(feat_errors >= 0)

    preds = detector.predict(X)
    assert np.sum(preds[100:] == -1) >= 4


def test_hbos_detector(synthetic_data: tuple[np.ndarray, np.ndarray]):
    X, X_outliers = synthetic_data
    detector = HBOSDetector(n_bins=10, contamination=0.05)
    detector.fit(X)

    assert detector.is_fitted()
    scores = detector.score_samples(X)
    assert scores.shape == (105,)

    normal_scores = scores[:100]
    outlier_scores = scores[100:]
    assert np.mean(outlier_scores) < np.mean(normal_scores)

    preds = detector.predict(X)
    assert np.sum(preds[100:] == -1) >= 4


def test_gmm_detector(synthetic_data: tuple[np.ndarray, np.ndarray]):
    X, X_outliers = synthetic_data
    detector = GMMDetector(n_components=2, contamination=0.05, random_state=42)
    detector.fit(X)

    assert detector.is_fitted()
    scores = detector.score_samples(X)
    assert scores.shape == (105,)

    normal_scores = scores[:100]
    outlier_scores = scores[100:]
    assert np.mean(outlier_scores) < np.mean(normal_scores)

    preds = detector.predict(X)
    assert np.sum(preds[100:] == -1) >= 3


def test_model_manager_expanded_registry(synthetic_data: tuple[np.ndarray, np.ndarray]):
    X, _ = synthetic_data
    manager = ModelManager()

    available = manager.get_available_models()
    assert "k-Nearest Neighbors" in available
    assert "PCA Reconstruction" in available
    assert "Histogram-Based Outlier Score" in available
    assert "Gaussian Mixture Model" in available

    caps = manager.get_model_capabilities("PCA Reconstruction")
    assert caps["family"] == "Subspace Projection"
    assert "Reconstruction Error Decomposition" in caps["explanation_methods"]

    res = manager.fit("Histogram-Based Outlier Score", X)
    assert res["success"] is True
    assert res["scores"] is not None
    assert len(res["scores"]) == 105

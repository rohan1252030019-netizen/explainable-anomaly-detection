"""
tests/test_phase3_detectors.py

Comprehensive test suite for SENTINEL Phase 3: Classical and Deep Learning Detectors.

Covers:
1. ECOD (Empirical Cumulative Distribution Functions)
2. COPOD (Copula-Based Outlier Detection)
3. Shrinkage Mahalanobis (Ledoit-Wolf covariance shrinkage on collinear data)
4. DBSCAN Noise Point Detector (Euclidean distance to core samples)
5. Extended Isolation Forest (Random hyperplane isolation cuts)
6. SUOD Fast Ensemble (Random subspace projection and Borda aggregation)
7. Deep Autoencoder (PyTorch, early stopping, native per-feature attribution)
8. Variational Autoencoder (PyTorch, latent ELBO, per-feature attribution)
9. Deep SVDD (PyTorch, hypersphere center distance)
10. ModelManager and FastAPI detection pipeline integration.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.models.classical import (
    COPODDetector,
    DBSCANDetector,
    ECODDetector,
    ExtendedIsolationForestDetector,
    MahalanobisDetector,
    SUODDetector,
)
from src.models.deep import (
    AutoencoderDetector,
    DeepSVDDDetector,
    VAEDetector,
)
from src.models.model_manager import ModelManager


@pytest.fixture
def synthetic_data():
    """Deterministic synthetic data with an injected extreme outlier."""
    rng = np.random.RandomState(42)
    # 150 normal points
    X_norm = rng.normal(loc=0.0, scale=1.0, size=(150, 5))
    # 5 anomalous points far in tail
    X_anom = rng.normal(loc=6.0, scale=1.0, size=(5, 5))
    X = np.vstack([X_norm, X_anom])
    return X


def test_ecod_detector(synthetic_data):
    """Verify ECOD fits, scores, and correctly isolates extreme tail anomalies."""
    detector = ECODDetector(contamination=0.05)
    detector.fit(synthetic_data)

    assert detector.is_fitted()
    scores = detector.score_samples(synthetic_data)
    preds = detector.predict(synthetic_data)

    assert len(scores) == len(synthetic_data)
    assert set(np.unique(preds)).issubset({-1, 1})

    # Outliers at the end must have lower scores (more anomalous) than normals at the beginning
    assert np.mean(scores[-5:]) < np.mean(scores[:150])
    assert preds[-1] == -1


def test_copod_detector(synthetic_data):
    """Verify COPOD copula tail probabilities and detection."""
    detector = COPODDetector(contamination=0.05)
    detector.fit(synthetic_data)

    assert detector.is_fitted()
    scores = detector.score_samples(synthetic_data)
    preds = detector.predict(synthetic_data)

    assert len(scores) == len(synthetic_data)
    assert np.mean(scores[-5:]) < np.mean(scores[:150])
    assert preds[-1] == -1


def test_shrinkage_mahalanobis_handles_collinear_features():
    """
    CRITICAL STABILITY TEST:
    Generate perfectly collinear features where empirical covariance is singular.
    Verify Ledoit-Wolf shrinkage inverts successfully where naive EllipticEnvelope fails.
    """
    rng = np.random.RandomState(42)
    x1 = rng.normal(0, 1, 100)
    x2 = x1 * 2.0  # Perfect collinearity: rank deficient!
    x3 = rng.normal(0, 1, 100)
    X_collinear = np.column_stack([x1, x2, x3])

    detector = MahalanobisDetector(contamination=0.05)
    # Must fit cleanly without LinAlgError
    detector.fit(X_collinear)

    assert detector.is_fitted()
    scores = detector.score_samples(X_collinear)
    assert not np.any(np.isnan(scores))
    assert not np.any(np.isinf(scores))


def test_dbscan_noise_detector(synthetic_data):
    """Verify DBSCAN identifies noise points and scores continuous distances."""
    detector = DBSCANDetector(eps=1.5, min_samples=5, contamination=0.05)
    detector.fit(synthetic_data)

    assert detector.is_fitted()
    scores = detector.score_samples(synthetic_data)
    preds = detector.predict(synthetic_data)

    assert len(scores) == len(synthetic_data)
    assert len(preds) == len(synthetic_data)
    # Outliers should be further from core
    assert np.mean(scores[-5:]) < np.mean(scores[:150])


def test_extended_isolation_forest(synthetic_data):
    """Verify EIF cuts along random hyperplanes and isolates anomalies."""
    detector = ExtendedIsolationForestDetector(
        n_estimators=30, max_samples=64, contamination=0.05, random_state=42
    )
    detector.fit(synthetic_data)

    assert detector.is_fitted()
    scores = detector.score_samples(synthetic_data)
    preds = detector.predict(synthetic_data)

    assert len(scores) == len(synthetic_data)
    assert len(preds) == len(synthetic_data)
    assert np.mean(scores[-5:]) < np.mean(scores[:150])


def test_suod_fast_ensemble(synthetic_data):
    """Verify SUOD combines heterogeneous models across random subspaces."""
    detector = SUODDetector(subspace_size=0.8, contamination=0.05, random_state=42)
    detector.fit(synthetic_data)

    assert detector.is_fitted()
    scores = detector.score_samples(synthetic_data)
    preds = detector.predict(synthetic_data)

    assert len(scores) == len(synthetic_data)
    assert len(preds) == len(synthetic_data)
    assert np.mean(scores[-5:]) < np.mean(scores[:150])


def test_deep_autoencoder_and_attribution(synthetic_data):
    """
    Verify PyTorch Autoencoder early stopping and native per-feature error attribution.
    """
    detector = AutoencoderDetector(
        latent_dim=2, hidden_dim=16, epochs=15, batch_size=32, device="cpu", random_state=42
    )
    detector.fit(synthetic_data)

    assert detector.is_fitted()
    scores = detector.score_samples(synthetic_data)
    preds = detector.predict(synthetic_data)

    assert len(scores) == len(synthetic_data)
    assert len(preds) == len(synthetic_data)
    assert np.mean(scores[-5:]) < np.mean(scores[:150])

    # Check native per-feature attribution
    attr = detector.get_feature_attribution(synthetic_data[:10])
    assert attr.shape == (10, 5)
    assert np.all(attr >= 0.0)


def test_variational_autoencoder(synthetic_data):
    """Verify PyTorch VAE fits and reconstructs nominal manifold."""
    detector = VAEDetector(
        latent_dim=2, hidden_dim=16, epochs=15, batch_size=32, device="cpu", random_state=42
    )
    detector.fit(synthetic_data)

    assert detector.is_fitted()
    scores = detector.score_samples(synthetic_data)
    preds = detector.predict(synthetic_data)

    assert len(scores) == len(synthetic_data)
    assert len(preds) == len(synthetic_data)
    assert np.mean(scores[-5:]) < np.mean(scores[:150])


def test_deep_svdd_detector(synthetic_data):
    """Verify Deep SVDD pulls representations toward fixed center c."""
    detector = DeepSVDDDetector(
        rep_dim=4, hidden_dim=16, epochs=15, batch_size=32, device="cpu", random_state=42
    )
    detector.fit(synthetic_data)

    assert detector.is_fitted()
    scores = detector.score_samples(synthetic_data)
    preds = detector.predict(synthetic_data)

    assert len(scores) == len(synthetic_data)
    assert len(preds) == len(synthetic_data)
    assert np.mean(scores[-5:]) < np.mean(scores[:150])


def test_model_manager_registry_coverage():
    """Verify all 17 detectors are registered and instantiable via ModelManager."""
    manager = ModelManager()
    available = manager.get_available_models()
    assert len(available) == 17

    for model_name in [
        "ECOD",
        "COPOD",
        "Shrinkage Mahalanobis",
        "DBSCAN Noise",
        "Extended Isolation Forest",
        "SUOD Fast Ensemble",
        "Deep Autoencoder",
        "Variational Autoencoder",
        "Deep SVDD",
    ]:
        assert model_name in available
        det = manager.create_detector(model_name)
        assert det is not None
        params = manager.get_default_params(model_name)
        assert isinstance(params, dict)


def test_api_detection_with_new_detectors():
    """Test running detection via FastAPI using new classical and deep models."""
    client = TestClient(app)
    # Load benchmark dataset
    client.post("/api/dataset/sample")

    # 1. Run detection with ECOD
    r_ecod = client.post(
        "/api/detection/run",
        json={"model_name": "ECOD", "contamination": 0.05},
    )
    assert r_ecod.status_code == 200
    assert r_ecod.json()["overview"]["model_name"] == "ECOD"

    # 2. Run detection with Deep Autoencoder
    r_ae = client.post(
        "/api/detection/run",
        json={
            "model_name": "Deep Autoencoder",
            "contamination": 0.05,
            "params": {"epochs": 10, "hidden_dim": 16, "latent_dim": 2},
        },
    )
    assert r_ae.status_code == 200
    assert r_ae.json()["overview"]["model_name"] == "Deep Autoencoder"

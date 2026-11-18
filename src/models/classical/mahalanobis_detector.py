"""
src/models/classical/mahalanobis_detector.py

Robust Mahalanobis Anomaly Detector with Ledoit-Wolf Covariance Shrinkage.

Mathematical Rationale:
-----------------------
Classical Gaussian distance (EllipticEnvelope) relies on empirical covariance
inversion, which catastrophically breaks down (singular matrix, division by zero)
when features are collinear, ill-conditioned, or when n < d.

The Ledoit-Wolf shrinkage estimator guarantees positive definiteness:
    Sigma_LW = (1 - rho) * Sigma_empirical + rho * (Tr(Sigma) / d) * I
where the shrinkage intensity rho in [0, 1] is analytically optimal under Frobenius norm loss.

Anomaly scoring:
    D_M(x) = sqrt( (x - mu)^T * Sigma_LW^{-1} * (x - mu) )
Score convention: returns -D_M(x) so that lower = larger distance = more anomalous.
"""

import time
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from src.models.base_detector import BaseDetector


class MahalanobisDetector(BaseDetector):
    """
    Robust Mahalanobis Anomaly Detector using Ledoit-Wolf Covariance Shrinkage.
    """

    MODEL_NAME = "Shrinkage Mahalanobis"

    def __init__(self, contamination: float = 0.05):
        super().__init__()
        self.contamination = contamination
        self._lw_model: LedoitWolf | None = None
        self._mean: np.ndarray | None = None
        self._precision: np.ndarray | None = None
        self._shrinkage: float = 0.0
        self._threshold: float = 0.0

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        n_samples, n_features = X_arr.shape
        self._lw_model = LedoitWolf(assume_centered=False)
        self._lw_model.fit(X_arr)

        self._mean = self._lw_model.location_
        self._precision = self._lw_model.get_precision()
        self._shrinkage = float(getattr(self._lw_model, "shrinkage_", 0.0))

        self._n_train_samples = n_samples
        self._n_features = n_features
        self._is_fitted = True

        train_scores = self.score_samples(X_arr)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative Mahalanobis distance.
        Lower = further from distribution centroid = more anomalous.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        diff = X_arr - self._mean  # shape (n, d)
        prec = self._precision     # shape (d, d)

        # Vectorized Mahalanobis distance: sqrt(sum(diff * (diff @ prec), axis=1))
        left = diff @ prec
        dist_sq = np.sum(left * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 0.0)
        dists = np.sqrt(dist_sq)

        return -dists

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self._threshold, 1, -1)

    def get_params(self) -> dict[str, Any]:
        return {
            "contamination": self.contamination,
            "shrinkage_intensity": round(self._shrinkage, 4),
        }

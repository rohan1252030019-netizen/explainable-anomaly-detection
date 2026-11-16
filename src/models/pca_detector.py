"""
src/models/pca_detector.py

PCA Reconstruction Error Anomaly Detector for XAUD 2.0.

Concept:
Normal observations typically lie along a low-dimensional linear subspace
defined by the principal components of maximum variance. Anomalous observations
frequently deviate from this principal hyperplane and display high reconstruction error:
    E(x) = ||x - x_reconstructed||^2

Score Convention:
In XAUD, lower score = more anomalous.
We return -reconstruction_error.
"""

import time
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from src.models.base_detector import BaseDetector


class PCADetector(BaseDetector):
    """
    Principal Component Analysis (PCA) Reconstruction Error Outlier Detector.

    Parameters
    ----------
    n_components : int or float, default=0.80
        Number of components to keep or fraction of variance explained (0.0 < float < 1.0).
    contamination : float, default=0.05
        Expected anomaly proportion for binary thresholding.
    random_state : int, default=42
        Seed for reproducibility.
    """

    MODEL_NAME = "PCA Reconstruction"

    def __init__(
        self,
        n_components: int | float = 0.80,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_components = n_components
        self.contamination = contamination
        self.random_state = random_state

        self._pca: PCA | None = None
        self._threshold: float | None = None
        self._mean: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCADetector":
        """
        Fit PCA model on training feature matrix X.
        """
        self._validate_input(X)
        t0 = time.perf_counter()

        max_possible = min(X.shape[0], X.shape[1])
        if isinstance(self.n_components, int) and self.n_components > max_possible:
            actual_components = max_possible
        else:
            actual_components = self.n_components

        self._pca = PCA(n_components=actual_components, random_state=self.random_state)
        self._pca.fit(X)

        self._mean = self._pca.mean_
        self._n_train_samples = X.shape[0]
        self._n_features = X.shape[1]
        self._is_fitted = True

        # Compute threshold on training data
        train_scores = self.score_samples(X)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative squared reconstruction error.
        Lower score = higher error = more anomalous.
        """
        self._check_fitted()
        self._validate_input(X)
        pca = self._fitted_model(self._pca)
        X_projected = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_projected)

        # Squared reconstruction error per sample
        error = np.sum((X - X_reconstructed) ** 2, axis=1)
        return -error

    def feature_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Return feature-level squared reconstruction errors, shape (n_samples, n_features).
        Used directly for native subspace explainability without approximations.
        """
        self._check_fitted()
        self._validate_input(X)
        pca = self._fitted_model(self._pca)
        X_projected = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_projected)
        return (X - X_reconstructed) ** 2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Binary prediction: +1 = normal, -1 = anomaly.
        """
        self._check_fitted()
        scores = self.score_samples(X)
        preds = np.ones(X.shape[0], dtype=int)
        preds[scores <= self._threshold] = -1
        return preds

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }

"""
src/models/classical/dbscan_detector.py

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) Anomaly Detector.

Principles:
- Discovers arbitrary non-linear cluster topologies based on local point density.
- Outliers are identified as noise points unable to join any high-density core cluster.
- Continuous scoring: evaluates Euclidean distance to the nearest core sample.
- Score convention: returns -distance so that lower = further from clusters = more anomalous.
"""

import time
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from src.models.base_detector import BaseDetector


class DBSCANDetector(BaseDetector):
    """
    DBSCAN Density-Based Noise Point Anomaly Detector.
    """

    MODEL_NAME = "DBSCAN Noise"

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        contamination: float = 0.05,
    ):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.contamination = contamination
        self._dbscan: DBSCAN | None = None
        self._core_samples: np.ndarray | None = None
        self._nn_core: NearestNeighbors | None = None
        self._threshold: float = 0.0

    def fit(self, X: np.ndarray) -> "DBSCANDetector":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        n_samples, n_features = X_arr.shape
        self._dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        labels = self._dbscan.fit_predict(X_arr)

        core_indices = self._dbscan.core_sample_indices_
        if len(core_indices) > 0:
            self._core_samples = X_arr[core_indices]
            self._nn_core = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self._core_samples)
        else:
            # Fallback if no core points formed (sparse data): use whole dataset
            self._core_samples = X_arr
            self._nn_core = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_arr)

        self._n_train_samples = n_samples
        self._n_features = n_features
        self._is_fitted = True

        train_scores = self.score_samples(X_arr)
        # Use either explicit noise labels or contamination quantile
        noise_mask = labels == -1
        if np.sum(noise_mask) > 0:
            self._threshold = float(np.max(train_scores[noise_mask]))
        else:
            self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative distance to nearest core sample.
        Lower = further from dense clusters = more anomalous.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        if self._nn_core is None:
            return np.zeros(len(X_arr))

        distances, _ = self._nn_core.kneighbors(X_arr)
        return -distances.ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self._threshold, 1, -1)

    def get_params(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "contamination": self.contamination,
        }

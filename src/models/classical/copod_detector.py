"""
src/models/classical/copod_detector.py

Copula-Based Outlier Detection (COPOD).

Mathematical Reference:
Li, Z., Zhao, Y., Guo, X., & Shen, C. (2020).
"COPOD: Copula-Based Outlier Detection."
IEEE International Conference on Data Mining (ICDM).

Principles:
- Fast, parameter-free, deterministic O(n * d) computation.
- Models multivariate tail dependence via empirical copulas.
- Avoids distance computation and matrix inversions.
- Score convention: returns -COPOD(x) so that lower = more anomalous.
"""

import time
from typing import Any

import numpy as np

from src.models.base_detector import BaseDetector


class COPODDetector(BaseDetector):
    """
    Copula-Based Outlier Detection (COPOD) Anomaly Detector.
    """

    MODEL_NAME = "COPOD"

    def __init__(self, contamination: float = 0.05):
        super().__init__()
        self.contamination = contamination
        self._sorted_features: list[np.ndarray] = []
        self._threshold: float = 0.0
        self._epsilon: float = 1e-10

    def fit(self, X: np.ndarray) -> "COPODDetector":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        n_samples, n_features = X_arr.shape
        self._sorted_features = [np.sort(X_arr[:, j]) for j in range(n_features)]

        self._n_train_samples = n_samples
        self._n_features = n_features
        self._is_fitted = True

        train_scores = self.score_samples(X_arr)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative COPOD score.
        Lower = higher COPOD copula tail extremity = more anomalous.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        n_train = self._n_train_samples

        left_tail_log = np.zeros(n_samples, dtype=np.float64)
        right_tail_log = np.zeros(n_samples, dtype=np.float64)

        for j in range(n_features):
            col = X_arr[:, j]
            sorted_col = self._sorted_features[j]

            left_counts = np.searchsorted(sorted_col, col, side="right")
            f_left = np.clip(left_counts / float(n_train), self._epsilon, 1.0)

            right_counts = n_train - np.searchsorted(sorted_col, col, side="left")
            f_right = np.clip(right_counts / float(n_train), self._epsilon, 1.0)

            left_tail_log += -np.log(f_left)
            right_tail_log += -np.log(f_right)

        # Joint empirical copula score
        copod_scores = np.maximum(left_tail_log, right_tail_log)
        return -copod_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self._threshold, 1, -1)

    def get_params(self) -> dict[str, Any]:
        return {"contamination": self.contamination}

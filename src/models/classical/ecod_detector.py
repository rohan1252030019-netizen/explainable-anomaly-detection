"""
src/models/classical/ecod_detector.py

Empirical Cumulative Distribution Functions for Outlier Detection (ECOD).

Mathematical Reference:
Li, Z., Zhao, Y., Botta, N., Ionescu, C., & Hu, X. (2022).
"ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions."
IEEE Transactions on Knowledge and Data Engineering.

Principles:
- Fast, parameter-free, deterministic O(n * d) computation.
- Estimates sample tail probabilities in left, right, and skew-weighted directions.
- Feature tail probabilities:
    O_left(x)  = -sum_{j=1}^d log( F_j(x_j) )
    O_right(x) = -sum_{j=1}^d log( 1 - F_j(x_j) )
    ECOD(x)    = max( O_left(x), O_right(x), O_skew(x) )
- Score convention: returns -ECOD(x) so that lower = more anomalous.
"""

import time
from typing import Any

import numpy as np
from scipy.stats import skew

from src.models.base_detector import BaseDetector


class ECODDetector(BaseDetector):
    """
    Empirical Cumulative Distribution Functions (ECOD) Anomaly Detector.
    """

    MODEL_NAME = "ECOD"

    def __init__(self, contamination: float = 0.05):
        super().__init__()
        self.contamination = contamination
        self._sorted_features: list[np.ndarray] = []
        self._skews: np.ndarray | None = None
        self._threshold: float = 0.0
        self._epsilon: float = 1e-10

    def fit(self, X: np.ndarray) -> "ECODDetector":
        """
        Fit ECOD by sorting each feature column to enable O(log n) empirical CDF evaluation.
        """
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        n_samples, n_features = X_arr.shape
        self._sorted_features = [np.sort(X_arr[:, j]) for j in range(n_features)]
        self._skews = np.asarray([skew(X_arr[:, j]) for j in range(n_features)])

        self._n_train_samples = n_samples
        self._n_features = n_features
        self._is_fitted = True

        # Compute threshold on training set
        train_scores = self.score_samples(X_arr)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative ECOD anomaly score.
        Lower = higher ECOD tail extremity = more anomalous.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        n_train = self._n_train_samples

        left_tail_sum = np.zeros(n_samples, dtype=np.float64)
        right_tail_sum = np.zeros(n_samples, dtype=np.float64)
        skew_tail_sum = np.zeros(n_samples, dtype=np.float64)

        for j in range(n_features):
            col = X_arr[:, j]
            sorted_col = self._sorted_features[j]

            # Empirical CDF: F(x) = count(sorted <= x) / n_train
            left_counts = np.searchsorted(sorted_col, col, side="right")
            f_left = np.clip(left_counts / float(n_train), self._epsilon, 1.0)

            # Empirical Right Tail: S(x) = count(sorted >= x) / n_train
            right_counts = n_train - np.searchsorted(sorted_col, col, side="left")
            f_right = np.clip(right_counts / float(n_train), self._epsilon, 1.0)

            left_tail_sum += -np.log(f_left)
            right_tail_sum += -np.log(f_right)

            # Skewness direction
            is_pos_skew = self._skews[j] >= 0.0 if self._skews is not None else True
            f_skew = f_right if is_pos_skew else f_left
            skew_tail_sum += -np.log(f_skew)

        # Maximize over left, right, and skew
        ecod_scores = np.maximum(np.maximum(left_tail_sum, right_tail_sum), skew_tail_sum)
        # Invert so lower = more anomalous
        return -ecod_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self._threshold, 1, -1)

    def get_params(self) -> dict[str, Any]:
        return {"contamination": self.contamination}

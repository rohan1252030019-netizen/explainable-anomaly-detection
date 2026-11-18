"""
src/models/hbos_detector.py

Histogram-Based Outlier Score (HBOS) Anomaly Detector for SENTINEL.

Concept:
HBOS is a fast, linear-time O(N * D) unsupervised outlier detection algorithm
that assumes feature independence. It constructs univariate histograms for each
feature and calculates the inverse log density:
    HBOS(x) = sum_{j=1}^D log(1 / (p_j(x_j) + eps))

Observations falling into low-frequency bins across multiple features accumulate
a large positive HBOS score.

Score Convention:
In SENTINEL, lower score = more anomalous.
We return -HBOS(x).
"""

import time
from typing import Any

import numpy as np

from src.models.base_detector import BaseDetector


class HBOSDetector(BaseDetector):
    """
    Histogram-Based Outlier Score (HBOS) Detector.

    Parameters
    ----------
    n_bins : int, default=10
        Number of equal-width or quantile histogram bins per feature.
    contamination : float, default=0.05
        Expected anomaly proportion.
    alpha : float, default=0.1
        Regularization smoothing parameter to prevent division by zero in empty bins.
    """

    MODEL_NAME = "Histogram-Based Outlier Score"

    def __init__(
        self,
        n_bins: int = 10,
        contamination: float = 0.05,
        alpha: float = 0.1,
    ):
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}.")
        if not 0.0 < contamination <= 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {contamination}.")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}.")

        self.n_bins = n_bins
        self.contamination = contamination
        self.alpha = alpha

        # Bin edges and density heights per feature
        self._bin_edges: list[np.ndarray] = []
        self._bin_densities: list[np.ndarray] = []
        self._density_floor: float = 1e-12
        self._threshold: float | None = None

    def fit(self, X: np.ndarray) -> "HBOSDetector":
        """
        Build univariate histograms for each feature in X.
        """
        self._validate_input(X)
        t0 = time.perf_counter()

        n_samples, n_features = X.shape
        self._bin_edges = []
        self._bin_densities = []

        for j in range(n_features):
            col = X[:, j]
            # Compute histogram
            counts, edges = np.histogram(col, bins=self.n_bins)
            # Add smoothing alpha
            smoothed_counts = counts.astype(np.float64) + self.alpha
            # Normalize to empirical probability density: sum = 1.0
            densities = smoothed_counts / smoothed_counts.sum()

            self._bin_edges.append(edges)
            self._bin_densities.append(densities)

        # Density the model assigns to a bin it never observed: what an empty
        # bin would have received after smoothing.  Used for values beyond the
        # fitted range (see score_samples).
        denominator = n_samples + self.alpha * self.n_bins
        self._density_floor = float(
            max(self.alpha / denominator if denominator > 0 else 0.0, 1e-12)
        )

        self._n_train_samples = n_samples
        self._n_features = n_features
        self._is_fitted = True

        # Compute threshold on training set
        train_scores = self.score_samples(X)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative HBOS score.
        Lower score = higher HBOS = lower probability density = more anomalous.
        """
        self._check_fitted()
        self._validate_input(X)

        n_samples, n_features = X.shape
        hbos_scores = np.zeros(n_samples, dtype=np.float64)

        for j in range(n_features):
            col = X[:, j]
            edges = self._bin_edges[j]
            densities = self._bin_densities[j]

            # Find bin indices (1 to n_bins)
            bin_idx = np.digitize(col, edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(densities) - 1)

            p_vals = densities[bin_idx]

            # ``np.digitize`` folds out-of-range values into the first / last
            # bin, so a point far beyond the fitted range was scored as if it sat
            # in the densest edge bin — an extreme outlier got a *normal* score
            # and was never flagged.  The histogram has no support there: use
            # the empty-bin density so the value is scored as maximally
            # improbable.
            outside = (col < edges[0]) | (col > edges[-1])
            if np.any(outside):
                p_vals = np.where(outside, self._density_floor, p_vals)

            hbos_scores += -np.log(p_vals)

        # Invert so lower score = more anomalous
        return -hbos_scores

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
            "n_bins": self.n_bins,
            "contamination": self.contamination,
            "alpha": self.alpha,
        }

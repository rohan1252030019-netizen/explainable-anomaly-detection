"""
src/models/knn_detector.py

k-Nearest Neighbors (k-NN) Distance-Based Anomaly Detector for XAUD 2.0.

Concept:
Anomalous points lie far from their nearest neighbors in feature space.
For each sample, we find its k nearest neighbors in the training data and
compute the distance metric:
- method="largest": Euclidean distance to the k-th neighbor.
- method="mean": Average Euclidean distance across all k neighbors.

Score Convention:
In XAUD, score_samples(X) follows the convention: lower score = more anomalous.
Since larger distance indicates greater abnormality, we return -distance.
"""

import time
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.models.base_detector import BaseDetector


class KNNDetector(BaseDetector):
    """
    k-Nearest Neighbors Outlier Detector.

    Parameters
    ----------
    n_neighbors : int, default=10
        Number of neighbors to evaluate.
    contamination : float, default=0.05
        Expected proportion of anomalies in the dataset.
    method : {"largest", "mean"}, default="largest"
        Measure distance to the k-th neighbor ("largest") or average distance ("mean").
    metric : str, default="euclidean"
        Distance metric for nearest neighbor search.
    """

    MODEL_NAME = "k-Nearest Neighbors"

    def __init__(
        self,
        n_neighbors: int = 10,
        contamination: float = 0.05,
        method: str = "largest",
        metric: str = "euclidean",
    ):
        super().__init__()
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")
        if not 0.0 < contamination <= 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {contamination}.")
        if method not in ("largest", "mean"):
            raise ValueError(f"method must be 'largest' or 'mean', got '{method}'.")

        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.method = method
        self.metric = metric

        self._nn: NearestNeighbors | None = None
        self._threshold: float | None = None
        self._X_train: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KNNDetector":
        """
        Fit the nearest neighbors search structure on training data X.
        """
        self._validate_input(X)
        t0 = time.perf_counter()

        # Ensure n_neighbors <= number of samples
        k = min(self.n_neighbors, X.shape[0])
        self._nn = NearestNeighbors(n_neighbors=k, metric=self.metric, n_jobs=-1)
        self._nn.fit(X)

        self._X_train = X.copy()
        self._n_train_samples = X.shape[0]
        self._n_features = X.shape[1]

        # Compute threshold on training set
        train_scores = self.score_samples(X)
        # Contamination quantile of scores: lower score = more anomalous
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative k-NN distance as raw score.
        Lower score = larger distance = more anomalous.
        """
        if self._nn is None:
            self._check_fitted()

        self._validate_input(X)
        # Ask for one extra neighbour.  A row that exists in the training set is
        # its own nearest neighbour at distance 0, so its "distance to the k-th
        # neighbour" is one rank too small: training rows looked systematically
        # more normal than they are, which placed the contamination threshold
        # below the real anomaly tail.
        k = min(self.n_neighbors, max(1, self._n_train_samples - 1))
        n_query = min(k + 1, self._n_train_samples)
        nn = self._fitted_model(self._nn)
        distances, _ = nn.kneighbors(X, n_neighbors=n_query)

        self_match = distances[:, 0] <= 1e-12
        if n_query > 1:
            # Shift the self-match to the end and keep the k real neighbours.
            distances = np.where(
                self_match[:, None],
                np.roll(distances, -1, axis=1),
                distances,
            )
        distances = distances[:, :k]

        if self.method == "largest":
            dist = distances[:, -1]
        else:
            dist = np.mean(distances, axis=1)

        # Invert distance so lower = more anomalous
        return -dist

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
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "method": self.method,
            "metric": self.metric,
        }

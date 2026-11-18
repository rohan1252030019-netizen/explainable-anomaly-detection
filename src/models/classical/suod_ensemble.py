"""
src/models/classical/suod_ensemble.py

Scalable Unsupervised Outlier Detection (SUOD) Ensemble.

Mathematical Reference:
Zhao, Y., Hu, X., Cheng, C., Wang, C., Wan, C., Wang, W., ... & Akoglu, L. (2021).
"SUOD: Accelerating Large-Scale Unsupervised Heterogeneous Outlier Detection."
Proceedings of Machine Learning and Systems (MLSys), 3, 463-478.

Principles:
- Subspace projection via random projections to accelerate quadratic detectors.
- Combines heterogeneous base models across randomized feature subspaces.
- Rank-normalizes individual detector outputs and computes consensus scores.
- Score convention: returns -consensus_score so that lower = more anomalous.
"""

import time
from typing import Any

import numpy as np
from scipy.stats import rankdata

from src.models.base_detector import BaseDetector
from src.models.classical.copod_detector import COPODDetector
from src.models.classical.ecod_detector import ECODDetector
from src.models.hbos_detector import HBOSDetector
from src.models.isolation_forest import IsolationForestDetector


class SUODDetector(BaseDetector):
    """
    Scalable Unsupervised Outlier Detection (SUOD) Fast Ensemble.
    """

    MODEL_NAME = "SUOD Fast Ensemble"

    def __init__(
        self,
        subspace_size: float = 0.75,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__()
        self.subspace_size = subspace_size
        self.contamination = contamination
        self.random_state = random_state
        self._estimators: list[tuple[Any, np.ndarray]] = []
        self._threshold: float = 0.0

    def fit(self, X: np.ndarray) -> "SUODDetector":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        n_samples, n_features = X_arr.shape
        rng = np.random.RandomState(self.random_state)

        # Base estimators
        factories = [
            lambda seed: IsolationForestDetector(n_estimators=50, random_state=seed),
            lambda _seed: ECODDetector(),
            lambda _seed: COPODDetector(),
            lambda _seed: HBOSDetector(),
        ]

        n_sub_features = max(1, int(np.ceil(n_features * self.subspace_size)))
        self._estimators = []

        for i, factory in enumerate(factories):
            seed = self.random_state + i
            feat_idx = rng.choice(n_features, size=n_sub_features, replace=False)
            model = factory(seed)
            model.fit(X_arr[:, feat_idx])
            self._estimators.append((model, feat_idx))

        self._n_train_samples = n_samples
        self._n_features = n_features
        self._is_fitted = True

        train_scores = self.score_samples(X_arr)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative SUOD consensus rank score.
        Lower = higher consensus outlier rank = more anomalous.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)
        rank_matrix = np.zeros((n_samples, len(self._estimators)), dtype=np.float64)

        for idx, (model, feat_idx) in enumerate(self._estimators):
            raw_scores = model.score_samples(X_arr[:, feat_idx])
            # In base detectors, lower score = more anomalous.
            # rankdata on negative score so that most anomalous gets rank n
            anom_ranks = rankdata(-raw_scores) / float(n_samples)
            rank_matrix[:, idx] = anom_ranks

        # Average Borda rank across subspace estimators
        avg_rank = np.mean(rank_matrix, axis=1)
        # Return negative rank so lower = more anomalous
        return -avg_rank

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self._threshold, 1, -1)

    def get_params(self) -> dict[str, Any]:
        return {
            "subspace_size": self.subspace_size,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "n_base_estimators": len(self._estimators),
        }

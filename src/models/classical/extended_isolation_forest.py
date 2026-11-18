"""
src/models/classical/extended_isolation_forest.py

Extended Isolation Forest (EIF) with Hyperplane Cuts.

Mathematical Reference:
Hariri, S., Kind, M. C., & Brunner, R. J. (2019).
"Extended Isolation Forest."
IEEE Transactions on Knowledge and Data Engineering, 33(4), 1479-1489.

Principles:
- Solves the axis-parallel cut artifact of standard Isolation Forest.
- Slices the feature space using random hyperplanes with arbitrary slopes:
    (x - p) . n <= 0
  where n ~ N(0, I) is a random unit normal and p is a random point in the data box.
- Anomaly score s(x, n) = 2^(-E[h(x)] / c(n)).
- Score convention: returns -s(x, n) so that lower = shorter path = more anomalous.
"""

import time
from typing import Any

import numpy as np

from src.models.base_detector import BaseDetector


def _c_factor(n: int) -> float:
    """Average path length of unsuccessful search in Binary Search Tree."""
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / float(n))


class _EIFNode:
    def __init__(
        self,
        left=None,
        right=None,
        n_vector=None,
        p_point=None,
        size: int = 0,
        is_leaf: bool = False,
        depth: int = 0,
    ):
        self.left = left
        self.right = right
        self.n_vector = n_vector
        self.p_point = p_point
        self.size = size
        self.is_leaf = is_leaf
        self.depth = depth


class ExtendedIsolationForestDetector(BaseDetector):
    """
    Extended Isolation Forest (EIF) Anomaly Detector.
    """

    MODEL_NAME = "Extended Isolation Forest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self._trees: list[_EIFNode] = []
        self._threshold: float = 0.0
        self._c: float = 1.0

    def fit(self, X: np.ndarray) -> "ExtendedIsolationForestDetector":
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_arr.ndim}D.")

        n_samples, n_features = X_arr.shape
        rng = np.random.RandomState(self.random_state)
        sample_size = min(n_samples, self.max_samples)
        self._c = _c_factor(sample_size)
        max_depth = int(np.ceil(np.log2(max(sample_size, 2))))

        self._trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n_samples, size=sample_size, replace=False)
            tree = self._build_tree(X_arr[idx], current_depth=0, max_depth=max_depth, rng=rng)
            self._trees.append(tree)

        self._n_train_samples = n_samples
        self._n_features = n_features
        self._is_fitted = True

        train_scores = self.score_samples(X_arr)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        return self

    def _build_tree(
        self,
        X_sub: np.ndarray,
        current_depth: int,
        max_depth: int,
        rng: np.random.RandomState,
    ) -> _EIFNode:
        n_samples, n_features = X_sub.shape
        if current_depth >= max_depth or n_samples <= 1:
            return _EIFNode(size=n_samples, is_leaf=True, depth=current_depth)

        # Random hyperplane normal vector
        n_vec = rng.normal(loc=0.0, scale=1.0, size=n_features)
        norm = np.linalg.norm(n_vec)
        if norm > 1e-10:
            n_vec /= norm

        # Random intercept point between min and max bounds
        mins = np.min(X_sub, axis=0)
        maxs = np.max(X_sub, axis=0)
        p_pt = rng.uniform(mins, maxs)

        # Split: (x - p) . n <= 0
        dots = np.dot(X_sub - p_pt, n_vec)
        left_mask = dots <= 0.0

        if np.all(left_mask) or not np.any(left_mask):
            return _EIFNode(size=n_samples, is_leaf=True, depth=current_depth)

        left = self._build_tree(X_sub[left_mask], current_depth + 1, max_depth, rng)
        right = self._build_tree(X_sub[~left_mask], current_depth + 1, max_depth, rng)

        return _EIFNode(
            left=left,
            right=right,
            n_vector=n_vec,
            p_point=p_pt,
            size=n_samples,
            is_leaf=False,
            depth=current_depth,
        )

    def _path_length(self, x: np.ndarray, node: _EIFNode) -> float:
        if node.is_leaf:
            return float(node.depth + _c_factor(node.size))
        dot = np.dot(x - node.p_point, node.n_vector)
        if dot <= 0.0:
            return self._path_length(x, node.left)
        else:
            return self._path_length(x, node.right)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative EIF anomaly score: -2^(-E[h(x)] / c(n)).
        Lower = more anomalous.
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)
        paths = np.zeros(n_samples, dtype=np.float64)

        for tree in self._trees:
            for i in range(n_samples):
                paths[i] += self._path_length(X_arr[i], tree)

        avg_paths = paths / float(len(self._trees))
        raw_scores = 2.0 ** (-avg_paths / max(1e-8, self._c))
        return -raw_scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self._threshold, 1, -1)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }

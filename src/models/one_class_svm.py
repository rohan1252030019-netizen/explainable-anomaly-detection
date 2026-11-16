"""
src/models/one_class_svm.py

One-Class SVM anomaly detector.

Academic background
-------------------
One-Class SVM (Scholkopf et al., 2001) learns a hypersphere (or general
kernel-based boundary) that encloses the majority of training observations.
New observations that fall outside the learned boundary are classified as
anomalies.

The nu parameter controls the upper bound on the fraction of training errors
(analogous to contamination in other detectors).

Score convention
----------------
sklearn's OneClassSVM.score_samples() returns:
    More negative → more anomalous (signed distance to boundary, negated).

Performance note
----------------
Standard One-Class SVM does not scale well to large datasets (>10k samples)
because it uses quadratic programming.  For large datasets, consider using
sklearn's SGDOneClassSVM.  This implementation uses the standard kernel-based
variant, suitable for academic demonstration datasets.
"""

import time
from typing import Any

import numpy as np
from sklearn.svm import OneClassSVM

from src.models.base_detector import BaseDetector


class OneClassSVMDetector(BaseDetector):
    """
    One-Class SVM anomaly detector.

    Parameters
    ----------
    kernel : str
        Kernel type.  'rbf' (default) works well for most cases.
        Options: 'linear', 'poly', 'rbf', 'sigmoid'.
    nu : float
        Upper bound on the fraction of training errors and lower bound on
        the fraction of support vectors.  Analogous to contamination.
        Range: (0, 1].
    gamma : str or float
        Kernel coefficient.  'scale' (default) uses 1/(n_features * X.var()).
        'auto' uses 1/n_features.
    """

    MODEL_NAME = "One-Class SVM"

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: Any = "scale",
    ):
        super().__init__()
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma

        self._model: OneClassSVM | None = None

    # ── BaseDetector interface ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """
        Fit One-Class SVM on feature matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self._validate_input(X)
        t0 = time.perf_counter()

        self._model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
        )
        self._model.fit(X)

        self._fit_time = time.perf_counter() - t0
        self._is_fitted = True
        self._n_train_samples = X.shape[0]
        self._n_features = X.shape[1]
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores.

        More negative → more anomalous (signed distance to the learned
        boundary, negated).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        self._validate_input(X)
        model = self._fitted_model(self._model)
        return model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary anomaly labels (+1 normal, -1 anomaly).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        self._validate_input(X)
        model = self._fitted_model(self._model)
        return model.predict(X)

    def get_params(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "nu": self.nu,
            "gamma": self.gamma,
        }

    def get_model_info(self) -> dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "algorithm": "One-Class SVM",
            "reference": "Scholkopf et al. (2001)",
            "description": (
                "Learns a kernel-based boundary enclosing the majority of "
                "training observations. Observations outside the boundary "
                "are classified as anomalies. The nu parameter controls the "
                "fraction of allowed training errors."
            ),
            "score_direction": "lower = more anomalous",
            "supports_novelty": True,
            "assumptions": (
                "Assumes training data is predominantly normal (inlier). "
                "Sensitive to feature scaling — always scale before using. "
                "Quadratic complexity: may be slow for very large datasets."
            ),
        })
        return info

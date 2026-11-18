"""
src/models/gmm_detector.py

Gaussian Mixture Model (GMM) Log-Likelihood Anomaly Detector for SENTINEL.

Concept:
Fits a multimodal Gaussian mixture to unlabelled feature distributions.
Samples situated in the tails or low-density regions between clusters
yield low log-likelihood values under the fitted mixture model:
    log p(x) = log sum_{k=1}^K w_k N(x; mu_k, Sigma_k)

Score Convention:
In SENTINEL, lower score = more anomalous.
GMM's log_likelihood(x) naturally satisfies this: lower log-likelihood = more anomalous.
"""

import time
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from src.models.base_detector import BaseDetector


class GMMDetector(BaseDetector):
    """
    Gaussian Mixture Model Density Outlier Detector.

    Parameters
    ----------
    n_components : int, default=3
        Number of mixture components.
    covariance_type : {"full", "tied", "diag", "spherical"}, default="full"
        Type of covariance parameters to use.
    contamination : float, default=0.05
        Expected anomaly proportion.
    random_state : int, default=42
        Seed for reproducibility.
    """

    MODEL_NAME = "Gaussian Mixture Model"

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__()
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}.")
        if not 0.0 < contamination <= 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {contamination}.")
        if covariance_type not in ("full", "tied", "diag", "spherical"):
            raise ValueError(f"Invalid covariance_type '{covariance_type}'.")

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.contamination = contamination
        self.random_state = random_state

        self._gmm: GaussianMixture | None = None
        self._threshold: float | None = None

    def fit(self, X: np.ndarray) -> "GMMDetector":
        """
        Fit Gaussian Mixture Model on feature matrix X.
        """
        self._validate_input(X)
        t0 = time.perf_counter()

        n_samples, n_features = X.shape

        # A full-covariance component has n_features * (n_features + 1) / 2
        # covariance parameters; with fewer observations than that, EM shrinks
        # the covariance towards singular, the log-likelihood is dominated by
        # numerical conditioning, and the resulting "anomaly scores" are noise.
        # Validate the requirement explicitly instead of clamping n_components
        # to n_samples and reporting a successful fit.
        if self.covariance_type == "full":
            observations_per_component = n_features + 1
        else:
            observations_per_component = 2
        required = self.n_components * observations_per_component
        if n_samples < required:
            raise ValueError(
                f"Gaussian Mixture Model needs at least {required} samples for "
                f"{self.n_components} component(s) with "
                f"covariance_type='{self.covariance_type}' and {n_features} "
                f"feature(s), but received {n_samples}. Reduce n_components, "
                f"use a simpler covariance_type, or provide more rows."
            )
        if self.n_components > n_samples:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed the number of "
                f"samples ({n_samples})."
            )

        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self._gmm.fit(X)

        self._n_train_samples = X.shape[0]
        self._n_features = X.shape[1]

        # Compute threshold on training set
        train_scores = self.score_samples(X)
        self._threshold = float(np.quantile(train_scores, self.contamination))

        self._fit_time = time.perf_counter() - t0
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return log-likelihood of each sample.
        Lower log-likelihood = more anomalous.
        """
        if self._gmm is None:
            self._check_fitted()
        self._validate_input(X)
        gmm = self._fitted_model(self._gmm)
        return gmm.score_samples(X)

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
            "covariance_type": self.covariance_type,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }

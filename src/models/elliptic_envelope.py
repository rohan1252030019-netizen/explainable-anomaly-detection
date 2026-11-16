"""
src/models/elliptic_envelope.py

Elliptic Envelope anomaly detector.

Academic background
-------------------
Elliptic Envelope (Rousseeuw & Van Driessen, 1999) fits a robust covariance
estimate to the data, assuming it follows a multivariate Gaussian distribution.
Observations that lie far from the centre of the fitted ellipse (measured by
Mahalanobis distance) are flagged as anomalies.

Assumptions and limitations
----------------------------
- The data must be roughly Gaussian (unimodal, elliptically distributed).
- Not suitable for multi-modal or heavily skewed data.
- Breaks down for very high-dimensional data (p > n).
- Use with caution and always communicate the Gaussian assumption.

Score convention
----------------
sklearn's EllipticEnvelope.score_samples() returns:
    Negative Mahalanobis distance → more negative = more anomalous.
"""

import time
from typing import Any

import numpy as np
from sklearn.covariance import EllipticEnvelope

from src.models.base_detector import BaseDetector


class EllipticEnvelopeDetector(BaseDetector):
    """
    Elliptic Envelope anomaly detector.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies.
    support_fraction : float or None
        Proportion of points included in the raw MCD support.
        If None, sklearn uses the default (approximately (n_samples + n_features + 1) / 2).
    random_state : int
        Seed for reproducibility.
    """

    MODEL_NAME = "Elliptic Envelope"

    def __init__(
        self,
        contamination: float = 0.05,
        support_fraction: float | None = None,
        random_state: int = 42,
    ):
        super().__init__()
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.random_state = random_state

        self._model: EllipticEnvelope | None = None
        self._fit_error: str | None = None

    # ── BaseDetector interface ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "EllipticEnvelopeDetector":
        """
        Fit Elliptic Envelope on feature matrix X.

        Catches singular covariance errors (common with OHE features that
        produce near-collinear columns) and stores the error message.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self._validate_input(X)

        # Elliptic Envelope requires n_samples > n_features
        if X.shape[0] <= X.shape[1]:
            self._fit_error = (
                f"Cannot fit Elliptic Envelope: n_samples ({X.shape[0]}) "
                f"must be greater than n_features ({X.shape[1]}). "
                "Try using fewer features or a different model."
            )
            return self

        t0 = time.perf_counter()
        try:
            self._model = EllipticEnvelope(
                contamination=self.contamination,
                support_fraction=self.support_fraction,
                random_state=self.random_state,
            )
            self._model.fit(X)
            self._fit_error = None
        except Exception as exc:
            self._fit_error = (
                f"Elliptic Envelope failed to fit: {exc}. "
                "This is often caused by near-singular covariance matrices "
                "(e.g. highly correlated or one-hot-encoded features). "
                "Consider using Isolation Forest or LOF instead."
            )
            self._model = None

        self._fit_time = time.perf_counter() - t0
        self._is_fitted = True
        self._n_train_samples = X.shape[0]
        self._n_features = X.shape[1]
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores (negative Mahalanobis distance).

        More negative → more anomalous.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        if self._model is None:
            raise RuntimeError(
                f"Elliptic Envelope is not usable: {self._fit_error}"
            )
        self._validate_input(X)
        return self._model.score_samples(X)

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
        if self._model is None:
            raise RuntimeError(
                f"Elliptic Envelope is not usable: {self._fit_error}"
            )
        self._validate_input(X)
        return self._model.predict(X)

    def is_available(self) -> bool:
        """Return True if the model fitted without error."""
        return self._is_fitted and self._model is not None

    def get_fit_error(self) -> str | None:
        """Return the fit error message, or None if fitting succeeded."""
        return self._fit_error

    def get_params(self) -> dict[str, Any]:
        return {
            "contamination": self.contamination,
            "support_fraction": self.support_fraction,
            "random_state": self.random_state,
        }

    def get_model_info(self) -> dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "algorithm": "Elliptic Envelope",
            "reference": "Rousseeuw & Van Driessen (1999)",
            "description": (
                "Fits a robust covariance estimate to data assumed to follow "
                "a multivariate Gaussian distribution. Anomalies are "
                "observations with high Mahalanobis distance from the centre."
            ),
            "score_direction": "lower = more anomalous",
            "supports_novelty": True,
            "assumptions": (
                "Assumes multivariate Gaussian distribution. "
                "Requires n_samples > n_features. "
                "Sensitive to non-Gaussian and multi-modal distributions. "
                "May fail with near-collinear or OHE features."
            ),
            "available": self.is_available(),
            "fit_error": self._fit_error,
        })
        return info

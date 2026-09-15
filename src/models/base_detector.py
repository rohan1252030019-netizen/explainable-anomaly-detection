"""
src/models/base_detector.py

Abstract base class for all anomaly detection models in the framework.

All concrete detectors (IsolationForest, LOF, OneClassSVM, EllipticEnvelope)
inherit from BaseDetector and implement:
    - fit(X)         : train on unlabelled data
    - score_samples(X): return raw anomaly scores
    - predict(X)     : return binary labels (+1 normal, -1 anomaly)
    - get_params()   : return hyperparameter dict
    - get_model_info(): return description dict

IMPORTANT: Labels must never be passed into fit() for unsupervised training.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseDetector(ABC):
    """Abstract base class for unsupervised anomaly detectors."""

    MODEL_NAME: str = "BaseDetector"

    def __init__(self):
        self._is_fitted: bool = False
        self._fit_time: float = 0.0
        self._n_train_samples: int = 0
        self._n_features: int = 0

    # ── Abstract methods (must be implemented by subclasses) ──────────────────

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector":
        """Fit the detector on feature matrix X (no labels)."""

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores.

        Convention: lower score = more anomalous.
        (Normalisation is handled externally by scoring.py)
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary predictions.

        Returns
        -------
        np.ndarray : +1 = normal, -1 = anomaly
        """

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return a dict of hyperparameters."""

    # ── Concrete helper methods ────────────────────────────────────────────────

    def get_model_info(self) -> Dict[str, Any]:
        """Return a summary dict describing the model."""
        return {
            "name": self.MODEL_NAME,
            "is_fitted": self._is_fitted,
            "fit_time_sec": round(self._fit_time, 4),
            "n_train_samples": self._n_train_samples,
            "n_features": self._n_features,
            "params": self.get_params(),
        }

    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit_time(self) -> float:
        return self._fit_time

    # ── Internal utilities ─────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.MODEL_NAME} is not fitted yet. "
                "Call fit(X) before scoring or predicting."
            )

    @staticmethod
    def _validate_input(X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy ndarray, got {type(X)}.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X has 0 samples.")
        if X.shape[1] == 0:
            raise ValueError("X has 0 features.")
        if np.any(np.isnan(X)):
            raise ValueError(
                "X contains NaN values. Ensure preprocessing (imputation) "
                "ran before passing data to the detector."
            )
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values.")

"""
src/models/supervised/base.py

Abstract base class for supervised and surrogate anomaly models in SENTINEL.

All supervised models adhere to:
- Strict out-of-fold validation to prevent data leakage.
- Clear separation between ground truth and pseudo-labels ("surrogate of ensemble").
- Exact TreeSHAP explanations where tree architectures permit.
- Standardized anomaly output conventions (+1 normal, -1 anomaly; score in [0, 1]).
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseSupervisedDetector(ABC):
    """
    Abstract base class for supervised, surrogate, and stacking detectors in SENTINEL.
    """

    MODEL_TYPE: str = "BaseSupervised"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._is_fitted: bool = False
        self._fit_time: float = 0.0
        self._n_train_samples: int = 0
        self._n_features: int = 0
        self.feature_names_: list[str] = []
        self.metadata_: dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> "BaseSupervisedDetector":
        """
        Fit the model on feature matrix X with target labels y.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,) binary indicators (0=normal, 1=anomaly)
        feature_names : list[str] | None
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary predictions following SENTINEL convention.

        Returns
        -------
        np.ndarray : +1 = normal, -1 = anomaly
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.

        Returns
        -------
        np.ndarray of shape (n_samples, 2): [P(normal), P(anomaly)]
        """

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return continuous anomaly scores in [0, 1].

        Higher score = more anomalous (1.0 = extreme anomaly).
        """

    @abstractmethod
    def explain(
        self,
        X: np.ndarray,
        max_samples: int = 500,
    ) -> dict[str, Any]:
        """
        Compute feature attribution explanations (e.g. TreeSHAP).
        """

    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_type": self.MODEL_TYPE,
            "is_fitted": self._is_fitted,
            "fit_time_sec": round(self._fit_time, 4),
            "n_train_samples": self._n_train_samples,
            "n_features": self._n_features,
            "feature_names": self.feature_names_,
            "random_state": self.random_state,
            **self.metadata_,
        }

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.MODEL_TYPE} is not fitted yet. Call fit() before predicting."
            )

"""
src/models/local_outlier_factor.py

Local Outlier Factor (LOF) anomaly detector.

Academic background
-------------------
LOF (Breunig et al., 2000) computes a local density estimate for each
observation and compares it to those of its k nearest neighbours.
Observations whose local density is significantly lower than their
neighbours' receive high LOF scores (> 1) and are flagged as anomalies.

Two modes
---------
novelty=False (default, inductive use disabled):
    fit_predict() is used during training.  The model scores the training
    set using leave-one-out LOF.  Predicting NEW samples is not directly
    supported in this mode.

novelty=True:
    The model is fitted on training data and can score/predict new samples
    using the training neighbourhood.  Used here so the fitted model can
    later score arbitrary data (e.g. when the user uploads new data).

We default to novelty=True to be consistent with the other detectors.

Score convention
----------------
sklearn's LOF negative_outlier_factor_ / score_samples():
    More negative → more anomalous (same convention as IsolationForest).
"""

import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from src.models.base_detector import BaseDetector


class LOFDetector(BaseDetector):
    """
    Local Outlier Factor anomaly detector.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbours for local density estimation.
        Typical range: 5–50.  Larger values produce smoother boundaries.
    contamination : float
        Expected fraction of anomalies (used for the decision threshold).
    metric : str
        Distance metric.  "euclidean" or "minkowski" are common choices.
    novelty : bool
        If True, the model supports scoring/predicting new samples after fit.
        Set to True (default) for consistency with other detectors.
    """

    MODEL_NAME = "Local Outlier Factor"

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        metric: str = "euclidean",
        novelty: bool = True,
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.novelty = novelty

        self._model: Optional[LocalOutlierFactor] = None

    # ── BaseDetector interface ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "LOFDetector":
        """
        Fit LOF on feature matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self._validate_input(X)

        # LOF requires at least n_neighbors + 1 samples
        if X.shape[0] <= self.n_neighbors:
            adjusted = max(2, X.shape[0] // 2)
            self.n_neighbors = adjusted

        t0 = time.perf_counter()
        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            novelty=self.novelty,
            n_jobs=-1,
        )

        if self.novelty:
            self._model.fit(X)
        else:
            # In non-novelty mode we call fit_predict to trigger score computation
            self._model.fit(X)

        self._fit_time = time.perf_counter() - t0
        self._is_fitted = True
        self._n_train_samples = X.shape[0]
        self._n_features = X.shape[1]
        self._X_train = X  # cache training data for non-novelty fallback
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores.

        Lower (more negative) → more anomalous.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        self._validate_input(X)

        if self.novelty:
            return self._model.score_samples(X)
        else:
            # Non-novelty mode: can only score training data
            # Fall back to negative_outlier_factor_ for training set,
            # or re-fit with novelty=True for new data
            if np.array_equal(X, self._X_train):
                return self._model.negative_outlier_factor_
            else:
                # Score new data by fitting a temporary novelty LOF
                temp_model = LocalOutlierFactor(
                    n_neighbors=self.n_neighbors,
                    contamination=self.contamination,
                    metric=self.metric,
                    novelty=True,
                    n_jobs=-1,
                )
                temp_model.fit(self._X_train)
                return temp_model.score_samples(X)

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

        if self.novelty:
            return self._model.predict(X)
        else:
            # Use scores and contamination threshold
            scores = self.score_samples(X)
            threshold = np.percentile(scores, self.contamination * 100)
            return np.where(scores < threshold, -1, 1)

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "metric": self.metric,
            "novelty": self.novelty,
        }

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "algorithm": "Local Outlier Factor",
            "reference": "Breunig et al. (2000)",
            "description": (
                "Estimates local density for each observation relative to "
                "its k nearest neighbours. Observations in sparser regions "
                "than their neighbours receive high LOF scores and are flagged "
                "as anomalies."
            ),
            "score_direction": "lower = more anomalous",
            "supports_novelty": self.novelty,
            "assumptions": (
                "Sensitive to the choice of n_neighbors. "
                "Works well for local and density-based outliers. "
                "Can struggle in high-dimensional spaces (curse of dimensionality)."
            ),
        })
        return info

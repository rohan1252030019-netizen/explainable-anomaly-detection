"""
src/models/isolation_forest.py

Isolation Forest anomaly detector.

Academic background
-------------------
Isolation Forest (Liu et al., 2008) isolates observations by recursively
partitioning the feature space with random splits.  Anomalies require fewer
splits to be isolated, resulting in shorter average path lengths and therefore
LOWER (more negative) decision_function scores.

Score convention used here
--------------------------
sklearn's IsolationForest.score_samples() returns:
    - More negative  → more anomalous
    - Less negative  → more normal

We expose raw scores as-is and leave normalisation to src/detection/scoring.py.

IMPORTANT: Labels are NEVER used during training.  The model is purely
unsupervised.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.base_detector import BaseDetector


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest–based anomaly detector.

    Parameters
    ----------
    n_estimators : int
        Number of base estimators (trees).  100–200 is usually sufficient.
    contamination : float or "auto"
        Expected fraction of anomalies.  Used only for the decision boundary,
        not to guide training.
    max_samples : int, float, or "auto"
        Samples drawn per tree.
    max_features : float
        Features drawn per tree.
    random_state : int
        Seed for reproducibility.
    """

    MODEL_NAME = "Isolation Forest"

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        max_samples: Any = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

        self._model: Optional[IsolationForest] = None

    # ── BaseDetector interface ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest on feature matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self._validate_input(X)
        t0 = time.perf_counter()

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X)

        self._fit_time = time.perf_counter() - t0
        self._is_fitted = True
        self._n_train_samples = X.shape[0]
        self._n_features = X.shape[1]
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores for each sample.

        Lower (more negative) values indicate more anomalous observations.
        Scores are NOT normalised here; use scoring.normalize_scores().

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Raw scores from sklearn's score_samples() method.
        """
        self._check_fitted()
        self._validate_input(X)
        return self._model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary anomaly labels.

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            +1 = normal, -1 = anomaly  (sklearn convention)
        """
        self._check_fitted()
        self._validate_input(X)
        return self._model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "algorithm": "Isolation Forest",
            "reference": "Liu et al. (2008)",
            "description": (
                "Isolates observations by recursively partitioning the feature "
                "space with random splits. Observations that require fewer "
                "partitions to be isolated receive lower (more anomalous) scores."
            ),
            "score_direction": "lower = more anomalous",
            "supports_novelty": True,
            "assumptions": (
                "No strong distributional assumptions. "
                "Works well for global outliers. "
                "Performance may degrade for very high-dimensional data."
            ),
        })
        return info

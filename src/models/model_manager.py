"""
src/models/model_manager.py

Central model manager for the Explainable Anomaly Detection Framework.

Responsibilities
----------------
- Maintain a registry of available detector classes.
- Create and configure detector instances with user-supplied parameters.
- Fit detectors on preprocessed data.
- Store fit results and timing.
- Persist / restore fitted models using joblib.
- Return a unified results structure containing scores, predictions, fit time.

IMPORTANT: Labels are NEVER passed to any detector during training.
"""

import os
import time
import json
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib

from src.models.base_detector import BaseDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.local_outlier_factor import LOFDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.elliptic_envelope import EllipticEnvelopeDetector


# ── Registry ───────────────────────────────────────────────────────────────────

DETECTOR_REGISTRY: Dict[str, type] = {
    "Isolation Forest": IsolationForestDetector,
    "Local Outlier Factor": LOFDetector,
    "One-Class SVM": OneClassSVMDetector,
    "Elliptic Envelope": EllipticEnvelopeDetector,
}

# Default hyperparameters for each model
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "Isolation Forest": {
        "n_estimators": 100,
        "contamination": 0.05,
        "max_samples": "auto",
        "max_features": 1.0,
        "random_state": 42,
    },
    "Local Outlier Factor": {
        "n_neighbors": 20,
        "contamination": 0.05,
        "metric": "euclidean",
        "novelty": True,
    },
    "One-Class SVM": {
        "kernel": "rbf",
        "nu": 0.05,
        "gamma": "scale",
    },
    "Elliptic Envelope": {
        "contamination": 0.05,
        "support_fraction": None,
        "random_state": 42,
    },
}


class ModelManager:
    """
    Manages the lifecycle of anomaly detection models.

    Parameters
    ----------
    models_dir : str
        Directory where fitted models are saved.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        # name -> fitted BaseDetector instance
        self._detectors: Dict[str, BaseDetector] = {}
        # name -> result dict (scores, predictions, fit_time, etc.)
        self._results: Dict[str, Dict[str, Any]] = {}
        # Experiment log
        self._experiments: List[Dict[str, Any]] = []

    # ── Model creation ─────────────────────────────────────────────────────────

    def create_detector(
        self,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> BaseDetector:
        """
        Instantiate a detector with the given (or default) parameters.

        Parameters
        ----------
        model_name : str
            One of the keys in DETECTOR_REGISTRY.
        params : dict or None
            Hyperparameters to override the defaults.

        Returns
        -------
        BaseDetector
        """
        if model_name not in DETECTOR_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(DETECTOR_REGISTRY.keys())}"
            )

        merged_params = {**DEFAULT_PARAMS.get(model_name, {})}
        if params:
            merged_params.update(params)

        cls = DETECTOR_REGISTRY[model_name]
        return cls(**merged_params)

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(
        self,
        model_name: str,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fit a detector and store scores + predictions.

        Parameters
        ----------
        model_name : str
        X : np.ndarray, shape (n_samples, n_features)
            Preprocessed feature matrix.  NO labels included.
        feature_names : list of str or None
        params : dict or None

        Returns
        -------
        result : dict with keys:
            model_name, scores, predictions, fit_time_sec,
            n_anomalies, anomaly_fraction, model_info
        """
        detector = self.create_detector(model_name, params)
        detector.fit(X)

        # Check for Elliptic Envelope fit failure
        if model_name == "Elliptic Envelope":
            ee: EllipticEnvelopeDetector = detector  # type: ignore
            if not ee.is_available():
                result = {
                    "model_name": model_name,
                    "scores": None,
                    "predictions": None,
                    "fit_time_sec": detector.fit_time(),
                    "n_anomalies": None,
                    "anomaly_fraction": None,
                    "model_info": detector.get_model_info(),
                    "error": ee.get_fit_error(),
                    "success": False,
                }
                self._detectors[model_name] = detector
                self._results[model_name] = result
                self._log_experiment(model_name, detector, X, result)
                return result

        scores = detector.score_samples(X)
        predictions = detector.predict(X)

        n_anomalies = int((predictions == -1).sum())
        anomaly_fraction = round(n_anomalies / len(predictions), 4)

        result = {
            "model_name": model_name,
            "scores": scores,
            "predictions": predictions,
            "fit_time_sec": round(detector.fit_time(), 4),
            "n_anomalies": n_anomalies,
            "anomaly_fraction": anomaly_fraction,
            "model_info": detector.get_model_info(),
            "feature_names": feature_names or [],
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "success": True,
            "error": None,
        }

        self._detectors[model_name] = detector
        self._results[model_name] = result
        self._log_experiment(model_name, detector, X, result)
        return result

    def fit_all(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        params_override: Optional[Dict[str, Dict[str, Any]]] = None,
        include_elliptic: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit all registered detectors on X.

        Parameters
        ----------
        X : np.ndarray
        feature_names : list of str or None
        params_override : dict of {model_name: params} or None
        include_elliptic : bool
            Whether to attempt fitting Elliptic Envelope.

        Returns
        -------
        dict of {model_name: result}
        """
        models_to_fit = list(DETECTOR_REGISTRY.keys())
        if not include_elliptic:
            models_to_fit = [m for m in models_to_fit if m != "Elliptic Envelope"]

        all_results = {}
        for name in models_to_fit:
            params = (params_override or {}).get(name)
            try:
                result = self.fit(name, X, feature_names, params)
            except Exception as exc:
                result = {
                    "model_name": name,
                    "scores": None,
                    "predictions": None,
                    "fit_time_sec": 0.0,
                    "n_anomalies": None,
                    "anomaly_fraction": None,
                    "model_info": {},
                    "error": str(exc),
                    "success": False,
                }
            all_results[name] = result

        return all_results

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Score new data using an already-fitted detector."""
        if model_name not in self._detectors:
            raise KeyError(f"Model '{model_name}' has not been fitted yet.")
        return self._detectors[model_name].score_samples(X)

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data using an already-fitted detector."""
        if model_name not in self._detectors:
            raise KeyError(f"Model '{model_name}' has not been fitted yet.")
        return self._detectors[model_name].predict(X)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_model(self, model_name: str) -> str:
        """
        Persist a fitted detector to disk using joblib.

        Returns
        -------
        path : str
            Path to the saved model file.
        """
        if model_name not in self._detectors:
            raise KeyError(f"Model '{model_name}' is not fitted.")
        safe_name = model_name.replace(" ", "_").lower()
        path = os.path.join(self.models_dir, f"{safe_name}.joblib")
        joblib.dump(self._detectors[model_name], path)
        return path

    def load_model(self, model_name: str) -> BaseDetector:
        """Load a previously saved detector from disk."""
        safe_name = model_name.replace(" ", "_").lower()
        path = os.path.join(self.models_dir, f"{safe_name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at: {path}")
        detector = joblib.load(path)
        self._detectors[model_name] = detector
        return detector

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_result(self, model_name: str) -> Optional[Dict[str, Any]]:
        return self._results.get(model_name)

    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._results)

    def get_fitted_model_names(self) -> List[str]:
        return [
            name for name, r in self._results.items()
            if r.get("success", False)
        ]

    def get_available_models(self) -> List[str]:
        return list(DETECTOR_REGISTRY.keys())

    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        return dict(DEFAULT_PARAMS.get(model_name, {}))

    def get_experiments(self) -> List[Dict[str, Any]]:
        return list(self._experiments)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _log_experiment(
        self,
        model_name: str,
        detector: BaseDetector,
        X: np.ndarray,
        result: Dict[str, Any],
    ) -> None:
        """Record an experiment entry for reproducibility."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model_name,
            "parameters": detector.get_params(),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "fit_time_sec": result.get("fit_time_sec", 0.0),
            "n_anomalies": result.get("n_anomalies"),
            "anomaly_fraction": result.get("anomaly_fraction"),
            "success": result.get("success", False),
            "error": result.get("error"),
            "random_state": 42,
        }
        self._experiments.append(entry)

    def save_experiment_log(self, path: str) -> None:
        """Save the experiment log to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._experiments, f, indent=2)

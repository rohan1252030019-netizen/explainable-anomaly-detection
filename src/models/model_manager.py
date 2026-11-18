"""
src/models/model_manager.py

Central model manager and capability registry for SENTINEL.

Responsibilities
----------------
- Maintain a registry of available detector classes and capability metadata.
- Create and configure detector instances with user-supplied parameters.
- Fit detectors on preprocessed data (NO labels permitted).
- Persist / restore fitted models using joblib.
- Unified access to scores, predictions, fit times, and model capability info.
"""

import datetime
import hashlib
import json
import os
import platform
from typing import Any

import joblib
import numpy as np
import sklearn

from src.models.base_detector import BaseDetector
from src.models.classical.copod_detector import COPODDetector
from src.models.classical.dbscan_detector import DBSCANDetector
from src.models.classical.ecod_detector import ECODDetector
from src.models.classical.extended_isolation_forest import (
    ExtendedIsolationForestDetector,
)
from src.models.classical.mahalanobis_detector import MahalanobisDetector
from src.models.classical.suod_ensemble import SUODDetector
from src.models.deep.autoencoder import AutoencoderDetector
from src.models.deep.deep_svdd import DeepSVDDDetector
from src.models.deep.vae import VAEDetector
from src.models.elliptic_envelope import EllipticEnvelopeDetector
from src.models.gmm_detector import GMMDetector
from src.models.hbos_detector import HBOSDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.knn_detector import KNNDetector
from src.models.local_outlier_factor import LOFDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.pca_detector import PCADetector
from src.models.supervised.classifier import SupervisedClassifierDetector
from src.models.supervised.stacking import HybridStackingDetector
from src.models.supervised.surrogate import SurrogateDetector

# ── Registry ───────────────────────────────────────────────────────────────────

DETECTOR_REGISTRY: dict[str, type] = {
    "Isolation Forest": IsolationForestDetector,
    "Local Outlier Factor": LOFDetector,
    "One-Class SVM": OneClassSVMDetector,
    "Elliptic Envelope": EllipticEnvelopeDetector,
    "k-Nearest Neighbors": KNNDetector,
    "PCA Reconstruction": PCADetector,
    "Histogram-Based Outlier Score": HBOSDetector,
    "Gaussian Mixture Model": GMMDetector,
    # Classical detectors
    "ECOD": ECODDetector,
    "COPOD": COPODDetector,
    "Shrinkage Mahalanobis": MahalanobisDetector,
    "DBSCAN Noise": DBSCANDetector,
    "Extended Isolation Forest": ExtendedIsolationForestDetector,
    "SUOD Fast Ensemble": SUODDetector,
    # Deep learning detectors
    "Deep Autoencoder": AutoencoderDetector,
    "Variational Autoencoder": VAEDetector,
    "Deep SVDD": DeepSVDDDetector,
}

SUPERVISED_REGISTRY: dict[str, type] = {
    "Pseudo-Label Surrogate": SurrogateDetector,
    "Supervised XGBoost": SupervisedClassifierDetector,
    "Hybrid Stacking Meta-Learner": HybridStackingDetector,
}

# Default hyperparameters for each model
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
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
    "k-Nearest Neighbors": {
        "n_neighbors": 10,
        "contamination": 0.05,
        "method": "largest",
        "metric": "euclidean",
    },
    "PCA Reconstruction": {
        "n_components": 0.80,
        "contamination": 0.05,
        "random_state": 42,
    },
    "Histogram-Based Outlier Score": {
        "n_bins": 10,
        "contamination": 0.05,
        "alpha": 0.1,
    },
    "Gaussian Mixture Model": {
        "n_components": 3,
        "covariance_type": "full",
        "contamination": 0.05,
        "random_state": 42,
    },
    "ECOD": {
        "contamination": 0.05,
    },
    "COPOD": {
        "contamination": 0.05,
    },
    "Shrinkage Mahalanobis": {
        "contamination": 0.05,
    },
    "DBSCAN Noise": {
        "eps": 0.5,
        "min_samples": 5,
        "contamination": 0.05,
    },
    "Extended Isolation Forest": {
        "n_estimators": 100,
        "max_samples": 256,
        "contamination": 0.05,
        "random_state": 42,
    },
    "SUOD Fast Ensemble": {
        "subspace_size": 0.75,
        "contamination": 0.05,
        "random_state": 42,
    },
    "Deep Autoencoder": {
        "latent_dim": 4,
        "hidden_dim": 32,
        "epochs": 40,
        "batch_size": 64,
        "lr": 0.005,
        "contamination": 0.05,
    },
    "Variational Autoencoder": {
        "latent_dim": 4,
        "hidden_dim": 32,
        "beta": 0.01,
        "epochs": 40,
        "batch_size": 64,
        "lr": 0.005,
        "contamination": 0.05,
    },
    "Deep SVDD": {
        "rep_dim": 16,
        "hidden_dim": 32,
        "epochs": 40,
        "batch_size": 64,
        "lr": 0.005,
        "contamination": 0.05,
    },
}


def _sha256_file(path: str) -> str:
    """Digest of an artifact's bytes, streamed so large models are not re-read."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Per-detector row ceilings.
#
# The shared MAX_ROWS guard (200k) is a memory bound, not a complexity bound.
# LOF and k-NN build an all-pairs neighbourhood graph, so their cost grows with
# n^2: 200k rows is ~40 billion distance evaluations, which does not finish and
# does not free memory while it runs.  These are the sizes at which a single
# request is still a reasonable thing to ask for.
MODEL_ROW_LIMITS: dict[str, int] = {
    "Isolation Forest": 200_000,
    "Local Outlier Factor": 20_000,
    "One-Class SVM": 50_000,
    "Elliptic Envelope": 50_000,
    "k-Nearest Neighbors": 50_000,
    "PCA Reconstruction": 200_000,
    "Histogram-Based Outlier Score": 200_000,
    "Gaussian Mixture Model": 200_000,
    "ECOD": 200_000,
    "COPOD": 200_000,
    "Shrinkage Mahalanobis": 100_000,
    "DBSCAN Noise": 25_000,
    "Extended Isolation Forest": 100_000,
    "SUOD Fast Ensemble": 200_000,
    "Deep Autoencoder": 100_000,
    "Variational Autoencoder": 100_000,
    "Deep SVDD": 100_000,
}


def row_limit_for(model_name: str) -> int | None:
    """Return the row ceiling for a detector, or ``None`` if unregistered."""
    return MODEL_ROW_LIMITS.get(model_name)


# Model Capability Registry for SENTINEL
MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "Isolation Forest": {
        "family": "Tree-based Ensemble",
        "time_complexity": "O(T * n * log(s))",
        "space_complexity": "O(T * s)",
        "scaling_required": False,
        "sensitive_to_dim": False,
        "explanation_methods": ["TreeExplainer SHAP", "Perturbation (LOFO)"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Fast, robust to irrelevant features, handles high dimensions.",
        "limitations": "Axes-parallel cuts can struggle with complex oblique relationships.",
    },
    "Local Outlier Factor": {
        "family": "Density-based Neighborhood",
        "time_complexity": "O(n^2 * d)",
        "space_complexity": "O(n * d)",
        "scaling_required": True,
        "sensitive_to_dim": True,
        "explanation_methods": ["KernelExplainer SHAP", "Perturbation (LOFO)"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Detects local density anomalies in multi-density clusters.",
        "limitations": "Quadratic distance complexity; requires feature scaling.",
    },
    "One-Class SVM": {
        "family": "Kernel Support Vector Boundary",
        "time_complexity": "O(n^2 * d) to O(n^3)",
        "space_complexity": "O(n_sv * d)",
        "scaling_required": True,
        "sensitive_to_dim": True,
        "explanation_methods": ["KernelExplainer SHAP", "Perturbation (LOFO)"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Effective in non-linear high-dimensional continuous domains.",
        "limitations": "Hyperparameter tuning (nu, gamma) is sensitive; slow on large n.",
    },
    "Elliptic Envelope": {
        "family": "Robust Covariance / Mahalanobis",
        "time_complexity": "O(n * d^2 + d^3)",
        "space_complexity": "O(d^2)",
        "scaling_required": True,
        "sensitive_to_dim": True,
        "explanation_methods": ["KernelExplainer SHAP", "Perturbation (LOFO)"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Statistically optimal for unimodal Gaussian/elliptical structures.",
        "limitations": "Fails when features are collinear (singular covariance) or multi-modal.",
    },
    "k-Nearest Neighbors": {
        "family": "Distance-based Proximity",
        "time_complexity": "O(n^2 * d) (fit O(n*d), query O(k*n*d))",
        "space_complexity": "O(n * d)",
        "scaling_required": True,
        "sensitive_to_dim": True,
        "explanation_methods": ["Distance Decomposition", "Perturbation (LOFO)", "Kernel SHAP"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Transparent geometric distance interpretation; no training optimization required.",
        "limitations": "Distance metrics degrade under high dimensions (curse of dimensionality).",
    },
    "PCA Reconstruction": {
        "family": "Subspace Projection",
        "time_complexity": "O(n * d * min(n, d))",
        "space_complexity": "O(d * k)",
        "scaling_required": True,
        "sensitive_to_dim": False,
        "explanation_methods": ["Reconstruction Error Decomposition", "Kernel SHAP", "Perturbation"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Linear time query, native exact feature-level error decomposition.",
        "limitations": "Assumes anomalies deviate from linear subspaces; blind to non-linear correlations.",
    },
    "Histogram-Based Outlier Score": {
        "family": "Statistical / Marginal Density",
        "time_complexity": "O(n * d)",
        "space_complexity": "O(d * n_bins)",
        "scaling_required": False,
        "sensitive_to_dim": False,
        "explanation_methods": ["Histogram Bin Density", "Perturbation (LOFO)"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Extremely fast, linear complexity O(n*d); highly scalable baseline.",
        "limitations": "Assumes feature independence; misses complex multivariate interactions.",
    },
    "Gaussian Mixture Model": {
        "family": "Probabilistic Mixture Density",
        "time_complexity": "O(iter * K * n * d^2)",
        "space_complexity": "O(K * d^2)",
        "scaling_required": True,
        "sensitive_to_dim": True,
        "explanation_methods": ["Log-Likelihood Decomposition", "Perturbation (LOFO)", "Kernel SHAP"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Captures multi-modal clustered densities with true likelihood scores.",
        "limitations": "EM algorithm can get stuck in local optima; needs adequate samples per cluster.",
    },
    "ECOD": {
        "family": "Empirical Cumulative Distribution Functions",
        "time_complexity": "O(n * d)",
        "space_complexity": "O(n * d)",
        "scaling_required": False,
        "sensitive_to_dim": False,
        "explanation_methods": ["Tail Probability Decomposition", "Kernel SHAP"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Parameter-free, deterministic, ultra-fast linear time O(n*d) computation.",
        "limitations": "Assumes dimensional tail independence; can underestimate complex multivariate joint outliers.",
    },
    "COPOD": {
        "family": "Empirical Copula Tail Dependence",
        "time_complexity": "O(n * d)",
        "space_complexity": "O(n * d)",
        "scaling_required": False,
        "sensitive_to_dim": False,
        "explanation_methods": ["Copula Tail Decomposition", "Kernel SHAP"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Parameter-free, models multivariate tail dependence without Gaussianity assumptions.",
        "limitations": "Empirical copulas degrade when extreme joint tail events have zero training coverage.",
    },
    "Shrinkage Mahalanobis": {
        "family": "Robust Covariance / Regularized Distance",
        "time_complexity": "O(n * d^2 + d^3)",
        "space_complexity": "O(d^2)",
        "scaling_required": True,
        "sensitive_to_dim": True,
        "explanation_methods": ["Distance Component Decomposition", "Kernel SHAP"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Ledoit-Wolf shrinkage avoids singular covariance breakdown, robust to collinear features.",
        "limitations": "Assumes unimodal elliptical density structure.",
    },
    "DBSCAN Noise": {
        "family": "Density-Based Clustering",
        "time_complexity": "O(n^2 * d) to O(n * log(n) * d)",
        "space_complexity": "O(n * d)",
        "scaling_required": True,
        "sensitive_to_dim": True,
        "explanation_methods": ["Nearest Core Distance Decomposition"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Identifies arbitrary geometric non-linear cluster boundaries and isolates noise points.",
        "limitations": "Sensitive to eps and min_samples hyperparameters; quadratic distance scaling.",
    },
    "Extended Isolation Forest": {
        "family": "Random Hyperplane Isolation Trees",
        "time_complexity": "O(T * n * d)",
        "space_complexity": "O(T * s)",
        "scaling_required": False,
        "sensitive_to_dim": False,
        "explanation_methods": ["Hyperplane Depth Decomposition", "Kernel SHAP"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Overcomes axis-parallel cut artifacts by cutting along random multi-feature hyperplanes.",
        "limitations": "Slightly higher computation per tree node than standard axis-parallel iForest.",
    },
    "SUOD Fast Ensemble": {
        "family": "Scalable Subspace Ensemble",
        "time_complexity": "O(T * n * (d * subspace_size))",
        "space_complexity": "O(T * s)",
        "scaling_required": False,
        "sensitive_to_dim": False,
        "explanation_methods": ["Subspace Model Consensus Analysis"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Accelerates quadratic base models via random subspace projections and rank consensus.",
        "limitations": "Ensemble variance depends on random subspace sampling coverage.",
    },
    "Deep Autoencoder": {
        "family": "Deep Neural Reconstruction",
        "time_complexity": "O(epochs * n * d * hidden)",
        "space_complexity": "O(weights)",
        "scaling_required": True,
        "sensitive_to_dim": False,
        "explanation_methods": ["Native Per-Feature Reconstruction Error", "Kernel SHAP"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Non-linear manifold bottleneck; exact native per-feature error attribution without perturbation.",
        "limitations": "Requires gradient descent optimization; hyperparameter tuning (epochs, lr, latent dim).",
    },
    "Variational Autoencoder": {
        "family": "Deep Probabilistic Latent Generative",
        "time_complexity": "O(epochs * n * d * hidden)",
        "space_complexity": "O(weights)",
        "scaling_required": True,
        "sensitive_to_dim": False,
        "explanation_methods": ["Native Per-Feature Reconstruction Error", "Latent KL Divergence"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Learns continuous probabilistic latent distribution; combines reconstruction with KL density.",
        "limitations": "Stochastic latent sampling; sensitive to beta trade-off parameter.",
    },
    "Deep SVDD": {
        "family": "Deep One-Class Hypersphere",
        "time_complexity": "O(epochs * n * d * hidden)",
        "space_complexity": "O(weights)",
        "scaling_required": True,
        "sensitive_to_dim": False,
        "explanation_methods": ["Representation Center Distance Decomposition"],
        "score_direction": "lower_is_anomalous",
        "strengths": "Directly minimizes latent hypersphere volume around nominal representations.",
        "limitations": "Requires careful representation collapse prevention (fixed center, no bias).",
    },
    "Pseudo-Label Surrogate": {
        "family": "Gradient Boosted Trees (Surrogate)",
        "provenance": "surrogate of ensemble",
        "time_complexity": "O(T * n * d)",
        "space_complexity": "O(T * 2^depth)",
        "scaling_required": False,
        "sensitive_to_dim": False,
        "explanation_methods": ["TreeExplainer SHAP", "Global Importance"],
        "score_direction": "higher_is_anomalous",
        "strengths": "Exact millisecond TreeSHAP, distills 8-model ensemble into single fast estimator.",
        "limitations": "Learns ensemble consensus, not ground truth; fidelity bounded by ensemble quality.",
    },
    "Supervised XGBoost": {
        "family": "Supervised Gradient Boosted Trees",
        "provenance": "ground_truth_supervised",
        "time_complexity": "O(T * n * d)",
        "space_complexity": "O(T * 2^depth)",
        "scaling_required": False,
        "sensitive_to_dim": False,
        "explanation_methods": ["TreeExplainer SHAP", "Global Importance"],
        "score_direction": "higher_is_anomalous",
        "strengths": "Direct PR-AUC/ROC-AUC optimization, scale_pos_weight for severe imbalance, cost-matrix threshold tuning.",
        "limitations": "Requires trustworthy ground truth labels; risks overfitting on small label sets.",
    },
    "Hybrid Stacking Meta-Learner": {
        "family": "Stacking Ensemble Meta-Learner",
        "provenance": "hybrid_stacking_ensemble",
        "time_complexity": "O(K * n * (models + T * d))",
        "space_complexity": "O(n * 8 + T * 2^depth)",
        "scaling_required": False,
        "sensitive_to_dim": False,
        "explanation_methods": ["TreeExplainer SHAP (Meta-Attribution)"],
        "score_direction": "higher_is_anomalous",
        "strengths": "Fuses 8 algorithm families via out-of-fold meta-features into boosted decision boundary.",
        "limitations": "Higher training cost due to out-of-fold multi-model cross-validation.",
    },
}


class ModelManager:
    """
    Manages the lifecycle and capabilities of anomaly detection models.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        self._detectors: dict[str, BaseDetector] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._experiments: list[dict[str, Any]] = []

    def create_detector(
        self,
        model_name: str,
        params: dict[str, Any] | None = None,
    ) -> BaseDetector:
        """
        Instantiate a detector with given or default parameters.
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

    def fit(
        self,
        model_name: str,
        X: np.ndarray,
        feature_names: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fit a detector on preprocessed feature matrix X (no labels).
        """
        detector = self.create_detector(model_name, params)
        detector.fit(X)

        # Check for Elliptic Envelope fit failure
        if model_name == "Elliptic Envelope":
            ee: EllipticEnvelopeDetector = detector  # type: ignore
            if not ee.is_available():
                result: dict[str, Any] = {
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
            "fit_time_sec": detector.fit_time(),
            "n_anomalies": n_anomalies,
            "anomaly_fraction": anomaly_fraction,
            "model_info": detector.get_model_info(),
            "feature_names": feature_names or [f"f{i}" for i in range(X.shape[1])],
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
        feature_names: list[str] | None = None,
        include_elliptic: bool = True,
        params_override: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fit all registered models on X."""
        results = {}
        overrides = params_override or {}
        for name in DETECTOR_REGISTRY:
            if not include_elliptic and name == "Elliptic Envelope":
                continue
            p = overrides.get(name)
            results[name] = self.fit(name, X, feature_names=feature_names, params=p)
        return results

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

    def get_detector(self, model_name: str) -> BaseDetector | None:
        return self._detectors.get(model_name)

    def save_model(self, model_name: str, filename: str | None = None) -> str:
        """
        Persist a fitted detector plus a reproducibility manifest.

        ``joblib.dump`` alone saves the object but not the context needed to
        trust it later: which features it was trained on, which hyperparameters
        were used, what threshold was in force, or which library versions
        produced it.  A pickle reloaded against a different feature order scores
        silently wrong.  A ``<artifact>.meta.json`` sidecar is written next to
        the model with that context and a SHA-256 of the artifact bytes, so
        ``load_model`` can detect a corrupted or swapped file.
        """
        if model_name not in self._detectors:
            raise KeyError(f"Model '{model_name}' is not fitted.")
        detector = self._detectors[model_name]
        if not detector.is_fitted():
            raise RuntimeError(f"Model '{model_name}' is not fitted. Cannot save.")

        safe_name = model_name.replace(" ", "_").lower()
        fname = filename or f"{safe_name}.joblib"
        path = os.path.join(self.models_dir, fname)
        joblib.dump(detector, path)

        result = self._results.get(model_name, {})
        info = detector.get_model_info()
        manifest = {
            "model_name": model_name,
            "artifact": os.path.basename(path),
            "artifact_sha256": _sha256_file(path),
            "feature_names": result.get("feature_names"),
            "hyperparameters": info.get("params", info.get("hyperparameters")),
            "random_state": (info.get("params") or {}).get("random_state")
            if isinstance(info.get("params"), dict)
            else None,
            "threshold": info.get("threshold"),
            "n_anomalies": result.get("n_anomalies"),
            "anomaly_fraction": result.get("anomaly_fraction"),
            "saved_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "environment": {
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
            },
        }
        manifest_path = path + ".meta.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=str)
        return path

    def load_model(self, model_name: str, path: str | None = None) -> BaseDetector:
        """
        Load a persisted detector, verifying its manifest when one exists.

        A mismatch between the recorded artifact digest and the file on disk is
        an error, not a warning: the whole point of the sidecar is to refuse to
        score data with an artifact that is not the one that was validated.
        """
        safe_name = model_name.replace(" ", "_").lower()
        p = path or os.path.join(self.models_dir, f"{safe_name}.joblib")
        if not os.path.exists(p):
            raise FileNotFoundError(f"No saved model found at: {p}")

        manifest_path = p + ".meta.json"
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            expected = manifest.get("artifact_sha256")
            if expected and expected != _sha256_file(p):
                raise RuntimeError(
                    f"Model artifact '{os.path.basename(p)}' does not match its "
                    "manifest digest; refusing to load a modified file."
                )
            if manifest.get("model_name") not in (None, model_name):
                raise RuntimeError(
                    f"Manifest describes '{manifest.get('model_name')}' but "
                    f"'{model_name}' was requested."
                )

        detector = joblib.load(p)
        self._detectors[model_name] = detector
        return detector

    def read_manifest(self, model_name: str, path: str | None = None) -> dict[str, Any] | None:
        """Return the saved manifest for a detector, if one was written."""
        safe_name = model_name.replace(" ", "_").lower()
        p = path or os.path.join(self.models_dir, f"{safe_name}.joblib")
        manifest_path = p + ".meta.json"
        if not os.path.exists(manifest_path):
            return None
        with open(manifest_path, encoding="utf-8") as handle:
            return json.load(handle)

    def get_result(self, model_name: str) -> dict[str, Any] | None:
        return self._results.get(model_name)

    def get_all_results(self) -> dict[str, dict[str, Any]]:
        return dict(self._results)

    def get_fitted_model_names(self) -> list[str]:
        return [
            name for name, r in self._results.items()
            if r.get("success", False)
        ]

    def get_available_models(self) -> list[str]:
        return list(DETECTOR_REGISTRY.keys())

    def get_available_supervised_models(self) -> list[str]:
        return list(SUPERVISED_REGISTRY.keys())

    def get_default_params(self, model_name: str) -> dict[str, Any]:
        return dict(DEFAULT_PARAMS.get(model_name, {}))

    def get_model_capabilities(self, model_name: str | None = None) -> dict[str, Any]:
        """Return capability metadata for one or all models."""
        if model_name:
            return MODEL_CAPABILITIES.get(model_name, {})
        return dict(MODEL_CAPABILITIES)

    def get_experiments(self) -> list[dict[str, Any]]:
        return list(self._experiments)

    def _log_experiment(
        self,
        model_name: str,
        detector: BaseDetector,
        X: np.ndarray,
        result: dict[str, Any],
    ) -> None:
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._experiments, f, indent=2)

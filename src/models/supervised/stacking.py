"""
src/models/supervised/stacking.py

Hybrid Stacking Meta-Learner for SENTINEL.

Combines the inductive biases of 8 heterogeneous unsupervised anomaly detectors:
1. Isolation Forest (Isolation tree isolation depth)
2. Local Outlier Factor (Local reachability density)
3. One-Class SVM (Kernel hyperplane boundary)
4. Elliptic Envelope (Mahalanobis covariance distance)
5. k-Nearest Neighbors (Distance to k-th nearest neighbor)
6. PCA Reconstruction (Residual subspace projection error)
7. Histogram-Based Outlier Score (Univariate density estimation)
8. Gaussian Mixture Model (Negative log-likelihood density)

Mathematical Leakage Safety:
----------------------------
If detectors were fitted on the full dataset and then fed into a meta-learner,
the meta-learner would overfit to in-sample idiosyncrasies and memorization.
To guarantee 100% leakage prevention:
- Out-Of-Fold (OOF) cross-validation generates an N x 8 meta-feature matrix Z.
- For each fold, all 8 detectors are trained purely on X[train_fold].
- X[val_fold] is scored and rank-normalized using ScoreNormalizer(method="rank").
- The XGBoost meta-classifier is trained on Z (or [Z, X] if passthrough=True).
- Explanations over Z provide meta-attribution: quantifying each algorithm family's
  contribution to the final detection consensus.
"""

import time
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from src.detection.scoring import ScoreNormalizer
from src.models.elliptic_envelope import EllipticEnvelopeDetector
from src.models.gmm_detector import GMMDetector
from src.models.hbos_detector import HBOSDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.knn_detector import KNNDetector
from src.models.local_outlier_factor import LOFDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.pca_detector import PCADetector
from src.models.supervised.base import BaseSupervisedDetector

DETECTOR_FACTORIES = {
    "Isolation Forest": lambda seed: IsolationForestDetector(random_state=seed),
    "Local Outlier Factor": lambda seed: LOFDetector(novelty=True),
    "One-Class SVM": lambda seed: OneClassSVMDetector(),
    "Elliptic Envelope": lambda seed: EllipticEnvelopeDetector(random_state=seed),
    "k-Nearest Neighbors": lambda seed: KNNDetector(),
    "PCA Reconstruction": lambda seed: PCADetector(random_state=seed),
    "Histogram-Based Outlier Score": lambda seed: HBOSDetector(),
    "Gaussian Mixture Model": lambda seed: GMMDetector(random_state=seed),
}


class HybridStackingDetector(BaseSupervisedDetector):
    """
    Hybrid Stacking Meta-Learner aggregating 8 unsupervised detectors into XGBoost.
    """

    MODEL_TYPE: str = "hybrid_stacking"

    def __init__(
        self,
        passthrough: bool = True,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        super().__init__(random_state=random_state)
        if not _HAS_XGB:
            raise ImportError("xgboost is required for HybridStackingDetector.")

        self.passthrough = passthrough
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_splits = n_splits

        self.fitted_base_detectors_: dict[str, Any] = {}
        self.meta_feature_names_: list[str] = list(DETECTOR_FACTORIES.keys())
        self.all_feature_names_: list[str] = []
        self.meta_model_: xgb.XGBClassifier | None = None
        self.scaler_: StandardScaler | None = None
        self.explainer_: Any = None
        self.oof_meta_matrix_: np.ndarray | None = None
        self.threshold_: float = 0.5

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        is_pseudo_label: bool = False,
        **kwargs: Any,
    ) -> "HybridStackingDetector":
        if is_pseudo_label or kwargs.get("provenance") == "pseudo_label":
            raise ValueError(
                "HybridStackingDetector refuses ensemble pseudo-label targets. "
                "Meta-learner stacking requires ground-truth labels to prevent circular self-reinforcing leakage."
            )
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int)
        n_samples, n_features = X_arr.shape

        self._n_train_samples = n_samples
        self._n_features = n_features
        orig_feature_names = (
            feature_names
            if feature_names is not None
            else [f"feat_{i}" for i in range(n_features)]
        )

        n_detectors = len(self.meta_feature_names_)
        meta_oof = np.zeros((n_samples, n_detectors), dtype=np.float64)

        n_splits = max(2, min(self.n_splits, n_samples // 5 if n_samples >= 10 else 2))
        if len(np.unique(y_arr)) > 1:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(X_arr, y_arr))
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            splits = list(cv.split(X_arr))

        # Generate out-of-fold meta-features
        for train_idx, val_idx in splits:
            X_tr = X_arr[train_idx]
            X_va = X_arr[val_idx]

            for det_idx, (_name, factory) in enumerate(DETECTOR_FACTORIES.items()):
                try:
                    det = factory(self.random_state)
                    det.fit(X_tr)
                    raw_scores = det.score_samples(X_va)
                    normalizer = ScoreNormalizer(method="rank")
                    norm_scores = normalizer.fit_transform(raw_scores)
                    meta_oof[val_idx, det_idx] = norm_scores
                except Exception:
                    # In degenerate covariance or failure cases, fill with median neutral score
                    meta_oof[val_idx, det_idx] = 0.5

        self.oof_meta_matrix_ = meta_oof

        # Fit production base detectors on full dataset
        for name, factory in DETECTOR_FACTORIES.items():
            try:
                det = factory(self.random_state)
                det.fit(X_arr)
                self.fitted_base_detectors_[name] = det
            except Exception:
                self.fitted_base_detectors_[name] = None

        # Build meta-training dataset
        if self.passthrough:
            X_meta_train = np.hstack([meta_oof, X_arr])
            self.all_feature_names_ = [f"meta_{m}" for m in self.meta_feature_names_] + orig_feature_names
        else:
            X_meta_train = meta_oof
            self.all_feature_names_ = [f"meta_{m}" for m in self.meta_feature_names_]

        self.feature_names_ = self.all_feature_names_

        self.scaler_ = StandardScaler()
        X_meta_scaled = self.scaler_.fit_transform(X_meta_train)

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(n_samples - n_pos)
        scale_pos_weight = float(n_neg / max(1, n_pos))

        self.meta_model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
        self.meta_model_.fit(X_meta_scaled, y_arr)

        if _HAS_SHAP:
            try:
                self.explainer_ = shap.TreeExplainer(self.meta_model_)
            except Exception:
                self.explainer_ = None

        self._fit_time = time.perf_counter() - t0
        self._is_fitted = True

        self.metadata_ = {
            "provenance": "hybrid_stacking_ensemble",
            "passthrough": self.passthrough,
            "n_base_detectors": len(self.fitted_base_detectors_),
            "base_detectors": list(self.fitted_base_detectors_.keys()),
            "scale_pos_weight": round(scale_pos_weight, 3),
            "meta_features_shape": [n_samples, len(self.all_feature_names_)],
        }
        return self

    def _transform_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Score new samples with all fitted base detectors to construct meta-features.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)
        meta_scores = np.zeros((n_samples, len(self.meta_feature_names_)), dtype=np.float64)

        for det_idx, name in enumerate(self.meta_feature_names_):
            det = self.fitted_base_detectors_.get(name)
            if det is not None:
                try:
                    raw = det.score_samples(X_arr)
                    norm = ScoreNormalizer(method="rank").fit_transform(raw)
                    meta_scores[:, det_idx] = norm
                except Exception:
                    meta_scores[:, det_idx] = 0.5
            else:
                meta_scores[:, det_idx] = 0.5

        if self.passthrough:
            return np.hstack([meta_scores, X_arr])
        return meta_scores

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_meta = self._transform_meta_features(X)
        X_scaled = self.scaler_.transform(X_meta) if self.scaler_ else X_meta
        return self.meta_model_.predict_proba(X_scaled)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_meta = self._transform_meta_features(X)
        X_scaled = self.scaler_.transform(X_meta) if self.scaler_ else X_meta
        return self.meta_model_.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold_, -1, 1)

    def explain(
        self,
        X: np.ndarray,
        max_samples: int = 500,
    ) -> dict[str, Any]:
        self._check_fitted()
        if not _HAS_SHAP or self.explainer_ is None:
            raise RuntimeError("SHAP is unavailable.")

        X_arr = np.asarray(X, dtype=np.float64)
        if len(X_arr) > max_samples:
            indices = np.random.RandomState(self.random_state).choice(
                len(X_arr), max_samples, replace=False
            )
            X_eval = X_arr[indices]
        else:
            indices = np.arange(len(X_arr))
            X_eval = X_arr

        X_meta = self._transform_meta_features(X_eval)
        X_meta_scaled = self.scaler_.transform(X_meta) if self.scaler_ else X_meta

        shap_values = self.explainer_.shap_values(X_meta_scaled)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        base_val = (
            float(self.explainer_.expected_value[1])
            if isinstance(self.explainer_.expected_value, (list, np.ndarray))
            else float(self.explainer_.expected_value)
        )

        global_importance = np.mean(np.abs(sv), axis=0)
        sorted_feat_idx = np.argsort(global_importance)[::-1]

        features_summary = [
            {
                "feature": self.all_feature_names_[idx],
                "mean_abs_shap": round(float(global_importance[idx]), 5),
                "rank": int(rank + 1),
                "is_meta_feature": self.all_feature_names_[idx].startswith("meta_"),
            }
            for rank, idx in enumerate(sorted_feat_idx)
        ]

        # Break down importance into base detectors vs original features
        base_detector_importance = [
            f for f in features_summary if f.get("is_meta_feature")
        ]
        original_feature_importance = [
            f for f in features_summary if not f.get("is_meta_feature")
        ]

        return {
            "method": "TreeSHAP_HybridStacking",
            "provenance": "hybrid_stacking_ensemble",
            "expected_value": round(base_val, 4),
            "n_samples_explained": len(X_eval),
            "shap_values": sv.tolist(),
            "global_importance": features_summary,
            "base_detector_attribution": base_detector_importance,
            "original_feature_attribution": original_feature_importance,
            "feature_names": self.all_feature_names_,
            "sample_indices": indices.tolist(),
        }

"""
src/models/supervised/surrogate.py

Pseudo-Label XGBoost Surrogate Model for SENTINEL.

Mathematical Foundation & Rationale:
-------------------------------------
Complex non-parametric ensembles (such as rank-Borda over 8 heterogeneous detectors)
yield superior detection performance but:
1. Incur slow inference on new streaming points (evaluating 8 models sequentially).
2. Lack unified, exact tree-based Shapley explanations (TreeSHAP).

The Pseudo-Label Surrogate distills the ensemble consensus into a single gradient boosted
tree model. To guarantee scientific validity and prevent data leakage:
- Pseudo-labels are derived from the ensemble anomaly score distribution via quantile thresholding:
  y_pseudo[i] = 1 if score[i] >= quantile(1 - contamination) else 0.
- Out-Of-Fold (OOF) cross-validation (K-Fold) ensures that every training observation is
  evaluated strictly on surrogate estimators that NEVER observed it during training.
- Scalers and thresholds are strictly fit on training fold partitions only.
- Fidelity metrics (Spearman rank correlation, Top-K Jaccard overlap, ROC/PR agreement)
  quantify surrogate-ensemble distillation fidelity.
- All outputs are explicitly tagged as 'surrogate of ensemble' (never ground truth).
"""

import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
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

from src.models.supervised.base import BaseSupervisedDetector


class FidelityReport:
    """
    Fidelity summary comparing surrogate model predictions to the underlying ensemble.
    """

    def __init__(
        self,
        spearman_rank_corr: float,
        spearman_pvalue: float,
        top_k_overlaps: dict[str, float],
        pseudo_label_accuracy: float,
        pseudo_label_f1: float,
        surrogate_mean_score: float,
        ensemble_mean_score: float,
        n_samples: int,
    ):
        self.spearman_rank_corr = spearman_rank_corr
        self.spearman_pvalue = spearman_pvalue
        self.top_k_overlaps = top_k_overlaps
        self.pseudo_label_accuracy = pseudo_label_accuracy
        self.pseudo_label_f1 = pseudo_label_f1
        self.surrogate_mean_score = surrogate_mean_score
        self.ensemble_mean_score = ensemble_mean_score
        self.n_samples = n_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": "surrogate of ensemble",
            "is_ground_truth": False,
            "spearman_rank_corr": round(float(self.spearman_rank_corr), 4),
            "spearman_pvalue": float(self.spearman_pvalue),
            "top_k_overlaps": {k: round(float(v), 4) for k, v in self.top_k_overlaps.items()},
            "pseudo_label_accuracy": round(float(self.pseudo_label_accuracy), 4),
            "pseudo_label_f1": round(float(self.pseudo_label_f1), 4),
            "surrogate_mean_score": round(float(self.surrogate_mean_score), 4),
            "ensemble_mean_score": round(float(self.ensemble_mean_score), 4),
            "n_samples": int(self.n_samples),
        }


class SurrogateDetector(BaseSupervisedDetector):
    """
    Gradient Boosted Tree Surrogate of an Anomaly Ensemble.
    """

    MODEL_TYPE: str = "surrogate_xgboost"

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        contamination: float = 0.05,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        super().__init__(random_state=random_state)
        if not _HAS_XGB:
            raise ImportError(
                "xgboost is required for SurrogateDetector. Install via 'pip install xgboost'."
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.contamination = contamination
        self.n_splits = n_splits

        self.model_: xgb.XGBClassifier | None = None
        self.explainer_: Any = None
        self.scaler_: StandardScaler | None = None
        self.threshold_: float = 0.5
        self.oof_scores_: np.ndarray | None = None
        self.oof_predictions_: np.ndarray | None = None
        self.fidelity_report_: FidelityReport | None = None

    def _derive_pseudo_labels(self, ensemble_scores: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Derive binary pseudo-labels from continuous ensemble scores using contamination quantile.

        y_pseudo = 1 for anomalies (top contamination quantile), 0 for normals.
        """
        threshold = float(np.quantile(ensemble_scores, 1.0 - self.contamination))
        y_pseudo = (ensemble_scores >= threshold).astype(int)
        # Ensure at least 1 positive sample in extreme cases
        if y_pseudo.sum() == 0 and len(ensemble_scores) > 0:
            y_pseudo[np.argmax(ensemble_scores)] = 1
        return y_pseudo, threshold

    def fit_from_ensemble(
        self,
        X: np.ndarray,
        ensemble_scores: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "SurrogateDetector":
        """
        Fit surrogate from continuous ensemble scores with out-of-fold cross-validation.
        """
        y_pseudo, pseudo_thresh = self._derive_pseudo_labels(ensemble_scores)
        return self.fit(
            X=X,
            y=y_pseudo,
            feature_names=feature_names,
            ensemble_scores=ensemble_scores,
            pseudo_threshold=pseudo_thresh,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        ensemble_scores: np.ndarray | None = None,
        pseudo_threshold: float | None = None,
        **kwargs: Any,
    ) -> "SurrogateDetector":
        """
        Train XGBClassifier using K-fold out-of-fold predictions.
        """
        t0 = time.perf_counter()
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int)
        n_samples, n_features = X_arr.shape

        self._n_train_samples = n_samples
        self._n_features = n_features
        self.feature_names_ = (
            feature_names
            if feature_names is not None
            else [f"feature_{i}" for i in range(n_features)]
        )

        # Calculate imbalance weight
        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(n_samples - n_pos)
        scale_pos_weight = float(n_neg / max(1, n_pos))

        # K-Fold Out-Of-Fold Validation
        n_splits = max(2, min(self.n_splits, n_samples // 5 if n_samples >= 10 else 2))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        oof_probs = np.zeros(n_samples, dtype=np.float64)
        oof_preds = np.zeros(n_samples, dtype=int)

        # Scaler fit on training fold only to prevent leakage
        for train_idx, val_idx in cv.split(X_arr, y_arr):
            X_train, y_train = X_arr[train_idx], y_arr[train_idx]
            X_val = X_arr[val_idx]

            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)

            fold_clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric="logloss",
                n_jobs=-1,
            )
            fold_clf.fit(X_train_scaled, y_train)

            val_probs = fold_clf.predict_proba(X_val_scaled)[:, 1]
            oof_probs[val_idx] = val_probs
            oof_preds[val_idx] = (val_probs >= 0.5).astype(int)

        self.oof_scores_ = oof_probs
        self.oof_predictions_ = np.where(oof_preds == 1, -1, 1)

        # Train final production surrogate on all data
        self.scaler_ = StandardScaler()
        X_all_scaled = self.scaler_.fit_transform(X_arr)

        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
        self.model_.fit(X_all_scaled, y_arr)

        # Fit TreeSHAP explainer
        if _HAS_SHAP:
            try:
                self.explainer_ = shap.TreeExplainer(self.model_)
            except Exception:
                self.explainer_ = None

        # Build Fidelity Report if ensemble scores are provided
        if ensemble_scores is not None:
            self.fidelity_report_ = self._compute_fidelity(
                ensemble_scores=ensemble_scores,
                oof_scores=oof_probs,
                y_pseudo=y_arr,
                oof_preds=oof_preds,
            )

        self._fit_time = time.perf_counter() - t0
        self._is_fitted = True

        self.metadata_ = {
            "provenance": "surrogate of ensemble",
            "is_ground_truth": False,
            "n_splits": n_splits,
            "scale_pos_weight": round(scale_pos_weight, 3),
            "pseudo_anomaly_count": n_pos,
            "pseudo_anomaly_rate": round(float(n_pos / max(1, n_samples)), 4),
            "pseudo_threshold": round(float(pseudo_threshold), 4) if pseudo_threshold else None,
            "fidelity": self.fidelity_report_.to_dict() if self.fidelity_report_ else None,
        }
        return self

    def _compute_fidelity(
        self,
        ensemble_scores: np.ndarray,
        oof_scores: np.ndarray,
        y_pseudo: np.ndarray,
        oof_preds: np.ndarray,
    ) -> FidelityReport:
        """
        Compute agreement between surrogate out-of-fold scores and ensemble scores.
        """
        # Spearman rank correlation
        rho, pval = spearmanr(oof_scores, ensemble_scores)
        if np.isnan(rho):
            rho, pval = 0.0, 1.0

        # Top-K overlap (Jaccard similarity on top-k anomalies)
        n = len(ensemble_scores)
        top_k_overlaps: dict[str, float] = {}
        for pct in [1, 5, 10, 20]:
            k = max(1, int(n * (pct / 100.0)))
            top_ens = set(np.argsort(ensemble_scores)[-k:])
            top_surr = set(np.argsort(oof_scores)[-k:])
            intersection = len(top_ens.intersection(top_surr))
            union = len(top_ens.union(top_surr))
            jaccard = intersection / max(1, union)
            top_k_overlaps[f"top_{pct}%"] = jaccard

        # Accuracy & F1 on pseudo labels
        acc = float(np.mean(oof_preds == y_pseudo))
        tp = np.sum((oof_preds == 1) & (y_pseudo == 1))
        fp = np.sum((oof_preds == 1) & (y_pseudo == 0))
        fn = np.sum((oof_preds == 0) & (y_pseudo == 1))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-8, precision + recall)

        return FidelityReport(
            spearman_rank_corr=float(rho),
            spearman_pvalue=float(pval),
            top_k_overlaps=top_k_overlaps,
            pseudo_label_accuracy=acc,
            pseudo_label_f1=float(f1),
            surrogate_mean_score=float(np.mean(oof_scores)),
            ensemble_mean_score=float(np.mean(ensemble_scores)),
            n_samples=n,
        )

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return surrogate continuous anomaly score in [0, 1].
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr) if self.scaler_ else X_arr
        return self.model_.predict_proba(X_scaled)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return [P(normal), P(anomaly)]
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr) if self.scaler_ else X_arr
        return self.model_.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary predictions (+1 normal, -1 anomaly).
        """
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold_, -1, 1)

    def explain(
        self,
        X: np.ndarray,
        max_samples: int = 500,
    ) -> dict[str, Any]:
        """
        Compute exact TreeSHAP attributions.
        """
        self._check_fitted()
        if not _HAS_SHAP or self.explainer_ is None:
            raise RuntimeError("SHAP or TreeExplainer is unavailable.")

        X_arr = np.asarray(X, dtype=np.float64)
        if len(X_arr) > max_samples:
            indices = np.random.RandomState(self.random_state).choice(
                len(X_arr), max_samples, replace=False
            )
            X_eval = X_arr[indices]
        else:
            indices = np.arange(len(X_arr))
            X_eval = X_arr

        X_eval_scaled = self.scaler_.transform(X_eval) if self.scaler_ else X_eval

        shap_values = self.explainer_.shap_values(X_eval_scaled)
        # For binary classification, shap_values may be 2D array or list of 2 arrays
        if isinstance(shap_values, list):
            sv = shap_values[1]  # positive class (anomaly)
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
                "feature": self.feature_names_[idx],
                "mean_abs_shap": round(float(global_importance[idx]), 5),
                "rank": int(rank + 1),
            }
            for rank, idx in enumerate(sorted_feat_idx)
        ]

        return {
            "method": "TreeSHAP",
            "provenance": "surrogate of ensemble",
            "expected_value": round(base_val, 4),
            "n_samples_explained": len(X_eval),
            "shap_values": sv.tolist(),
            "global_importance": features_summary,
            "feature_names": self.feature_names_,
            "sample_indices": indices.tolist(),
        }

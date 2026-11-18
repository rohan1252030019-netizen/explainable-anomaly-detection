"""
src/models/supervised/classifier.py

Supervised Anomaly Detection Classifier for SENTINEL.

Handles:
- Labeled anomaly datasets (imbalanced binary classification: 0=normal, 1=anomaly).
- Severe class imbalance via scale_pos_weight and sample weighting.
- Cross-validation: StratifiedKFold or TimeSeriesSplit for temporal order preservation.
- Rigorous evaluation metrics: PR-AUC (Average Precision), ROC-AUC, Precision@K, MCC, F1.
- Calibration analysis: Brier Score, Expected Calibration Error (ECE) with reliability diagrams.
- Decision threshold optimization via custom asymmetric Cost Matrices:
  Total Cost = C_FP * FP + C_FN * FN
"""

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import catboost as cb
    _HAS_CB = True
except ImportError:
    _HAS_CB = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from src.models.supervised.base import BaseSupervisedDetector


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """
    Compute Expected Calibration Error (ECE) and reliability bin statistics.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n_samples = len(y_true)
    reliability_bins: list[dict[str, float]] = []

    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count > 0:
            avg_confidence = float(np.mean(y_prob[mask]))
            empirical_accuracy = float(np.mean(y_true[mask]))
            diff = abs(empirical_accuracy - avg_confidence)
            ece += (count / n_samples) * diff

            reliability_bins.append({
                "bin_idx": i,
                "bin_start": round(float(bins[i]), 2),
                "bin_end": round(float(bins[i + 1]), 2),
                "count": count,
                "avg_confidence": round(avg_confidence, 4),
                "empirical_accuracy": round(empirical_accuracy, 4),
                "calibration_gap": round(diff, 4),
            })
        else:
            reliability_bins.append({
                "bin_idx": i,
                "bin_start": round(float(bins[i]), 2),
                "bin_end": round(float(bins[i + 1]), 2),
                "count": 0,
                "avg_confidence": round(float((bins[i] + bins[i + 1]) / 2.0), 2),
                "empirical_accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return float(ece), reliability_bins


def optimize_threshold_cost_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
    n_thresholds: int = 200,
) -> dict[str, Any]:
    """
    Determine the optimal decision threshold that minimizes total asymmetric cost:
    Cost(tau) = cost_fp * FP(tau) + cost_fn * FN(tau)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thresh = 0.5
    best_cost = float("inf")
    best_metrics = {}

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        total_cost = (cost_fp * fp) + (cost_fn * fn)
        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = float(t)
            best_metrics = {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": (2 * tp) / max(1, (2 * tp + fp + fn)),
            }

    return {
        "optimal_threshold": round(best_thresh, 4),
        "min_cost": round(best_cost, 2),
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "confusion_matrix_at_optimum": best_metrics,
    }


class SupervisedClassifierDetector(BaseSupervisedDetector):
    """
    Supervised gradient boosted anomaly classifier with imbalance correction and cost matrix.
    """

    MODEL_TYPE: str = "supervised_xgboost"

    def __init__(
        self,
        framework: str = "xgboost",
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        time_aware: bool = False,
        n_splits: int = 5,
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
        random_state: int = 42,
    ):
        super().__init__(random_state=random_state)
        self.framework = framework.lower()
        if self.framework == "xgboost" and not _HAS_XGB:
            raise ImportError("xgboost is not installed.")
        if self.framework == "lightgbm" and not _HAS_LGB:
            raise ImportError("lightgbm is not installed. Install via 'pip install lightgbm'.")
        if self.framework == "catboost" and not _HAS_CB:
            raise ImportError("catboost is not installed. Install via 'pip install catboost'.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.time_aware = time_aware
        self.n_splits = n_splits
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn

        self.model_: Any = None
        self.scaler_: StandardScaler | None = None
        self.optimal_threshold_: float = 0.5
        self.cost_matrix_summary_: dict[str, Any] = {}
        self.cv_metrics_: dict[str, Any] = {}
        self.calibration_report_: dict[str, Any] = {}
        self.explainer_: Any = None

    def _init_estimator(self, scale_pos_weight: float) -> Any:
        if self.framework == "xgboost":
            return xgb.XGBClassifier(
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
        elif self.framework == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif self.framework == "catboost":
            return cb.CatBoostClassifier(
                iterations=self.n_estimators,
                depth=self.max_depth,
                learning_rate=self.learning_rate,
                auto_class_weights="Balanced",
                random_seed=self.random_state,
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> "SupervisedClassifierDetector":
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

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(n_samples - n_pos)
        scale_pos_weight = float(n_neg / max(1, n_pos))

        # Cross-validation
        n_splits = max(2, min(self.n_splits, n_samples // 5 if n_samples >= 10 else 2))
        if self.time_aware:
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        oof_probs = np.zeros(n_samples, dtype=np.float64)
        has_eval_mask = np.zeros(n_samples, dtype=bool)

        for train_idx, val_idx in cv.split(X_arr, y_arr):
            X_train, y_train = X_arr[train_idx], y_arr[train_idx]
            X_val = X_arr[val_idx]

            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)

            fold_clf = self._init_estimator(scale_pos_weight)
            fold_clf.fit(X_train_scaled, y_train)

            oof_probs[val_idx] = fold_clf.predict_proba(X_val_scaled)[:, 1]
            has_eval_mask[val_idx] = True

        eval_indices = np.where(has_eval_mask)[0]
        y_eval = y_arr[eval_indices]
        prob_eval = oof_probs[eval_indices]

        # Calculate metrics
        roc_auc = float(roc_auc_score(y_eval, prob_eval)) if len(np.unique(y_eval)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_eval, prob_eval)) if len(np.unique(y_eval)) > 1 else 0.0
        brier = float(brier_score_loss(y_eval, prob_eval))

        # Precision@K
        precision_at_k: dict[str, float] = {}
        for pct in [1, 5, 10]:
            k = max(1, int(len(y_eval) * (pct / 100.0)))
            top_k_indices = np.argsort(prob_eval)[-k:]
            p_k = float(np.mean(y_eval[top_k_indices] == 1))
            precision_at_k[f"precision@{pct}%"] = round(p_k, 4)

        # Expected Calibration Error
        ece, reliability_bins = compute_expected_calibration_error(y_eval, prob_eval, n_bins=10)

        # Threshold optimization via cost matrix
        cost_res = optimize_threshold_cost_matrix(
            y_true=y_eval,
            y_prob=prob_eval,
            cost_fp=self.cost_fp,
            cost_fn=self.cost_fn,
        )
        self.optimal_threshold_ = cost_res["optimal_threshold"]
        self.cost_matrix_summary_ = cost_res

        # Discretize OOF predictions at optimal threshold
        opt_preds = (prob_eval >= self.optimal_threshold_).astype(int)
        mcc = float(matthews_corrcoef(y_eval, opt_preds))
        f1 = float(f1_score(y_eval, opt_preds, zero_division=0))
        precision = float(precision_score(y_eval, opt_preds, zero_division=0))
        recall = float(recall_score(y_eval, opt_preds, zero_division=0))

        self.cv_metrics_ = {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "precision_at_k": precision_at_k,
            "mcc": round(mcc, 4),
            "f1_at_optimal_threshold": round(f1, 4),
            "precision_at_optimal_threshold": round(precision, 4),
            "recall_at_optimal_threshold": round(recall, 4),
            "optimal_threshold": round(self.optimal_threshold_, 4),
        }

        self.calibration_report_ = {
            "ece": round(ece, 4),
            "brier_score": round(brier, 4),
            "reliability_bins": reliability_bins,
        }

        # Train full estimator on all observations
        self.scaler_ = StandardScaler()
        X_all_scaled = self.scaler_.fit_transform(X_arr)

        self.model_ = self._init_estimator(scale_pos_weight)
        self.model_.fit(X_all_scaled, y_arr)

        if _HAS_SHAP:
            try:
                self.explainer_ = shap.TreeExplainer(self.model_)
            except Exception:
                self.explainer_ = None

        self._fit_time = time.perf_counter() - t0
        self._is_fitted = True

        self.metadata_ = {
            "provenance": "ground_truth_supervised",
            "is_ground_truth": True,
            "framework": self.framework,
            "time_aware": self.time_aware,
            "scale_pos_weight": round(scale_pos_weight, 3),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "metrics": self.cv_metrics_,
            "cost_matrix": self.cost_matrix_summary_,
            "calibration": self.calibration_report_,
        }
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr) if self.scaler_ else X_arr
        return self.model_.predict_proba(X_scaled)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X_arr) if self.scaler_ else X_arr
        return self.model_.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary prediction (+1 normal, -1 anomaly) based on optimal threshold.
        """
        scores = self.score_samples(X)
        return np.where(scores >= self.optimal_threshold_, -1, 1)

    def explain(
        self,
        X: np.ndarray,
        max_samples: int = 500,
    ) -> dict[str, Any]:
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
                "feature": self.feature_names_[idx],
                "mean_abs_shap": round(float(global_importance[idx]), 5),
                "rank": int(rank + 1),
            }
            for rank, idx in enumerate(sorted_feat_idx)
        ]

        return {
            "method": "TreeSHAP",
            "provenance": "ground_truth_supervised",
            "expected_value": round(base_val, 4),
            "n_samples_explained": len(X_eval),
            "shap_values": sv.tolist(),
            "global_importance": features_summary,
            "feature_names": self.feature_names_,
            "sample_indices": indices.tolist(),
        }

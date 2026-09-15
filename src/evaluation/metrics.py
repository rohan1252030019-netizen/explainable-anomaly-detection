"""
src/evaluation/metrics.py

Evaluation metrics for the Explainable Anomaly Detection Framework.

Two evaluation modes
--------------------
1. Supervised (ground truth available):
   Labels are used ONLY here, for post-hoc evaluation.
   They are NEVER used during model training.

   Metrics:
   - Precision, Recall, F1-score
   - ROC-AUC
   - PR-AUC (Average Precision)
   - Confusion Matrix
   - Detection rate (TPR), False Positive Rate (FPR)

2. Unsupervised (no ground truth):
   - Score separation (anomaly vs normal score distributions)
   - Model agreement (fraction of observations all models agree on)
   - Stability across seeds (not computed here; done in model_comparison.py)
   - Anomaly score distribution statistics

IMPORTANT:
- Never fabricate metrics.
- If labels are absent, return None for supervised metrics.
- Clearly document that precision/recall require ground-truth labels.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ── Label conversion helper ────────────────────────────────────────────────────

def _to_binary(labels: np.ndarray) -> np.ndarray:
    """
    Convert labels to binary {0=normal, 1=anomaly}.

    Handles sklearn convention (+1/-1) and common (0/1) convention.
    """
    labels = np.asarray(labels)
    unique = set(labels)

    if unique <= {-1, 1}:
        # sklearn: -1 = anomaly, +1 = normal
        return (labels == -1).astype(int)
    elif unique <= {0, 1}:
        # standard: 1 = anomaly, 0 = normal
        return labels.astype(int)
    else:
        raise ValueError(
            f"Unexpected label values: {unique}. "
            "Expected {{-1, 1}} or {{0, 1}}."
        )


# ── Supervised evaluation ──────────────────────────────────────────────────────

def supervised_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    norm_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute supervised evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.  Convention: 1=anomaly, 0=normal (or -1/+1).
    y_pred : np.ndarray
        Model predictions.  Convention: -1=anomaly, +1=normal (sklearn).
    norm_scores : np.ndarray
        Normalised anomaly scores [0, 1].  Used for AUC computation.

    Returns
    -------
    dict with keys:
        precision, recall, f1, roc_auc, pr_auc,
        confusion_matrix, tp, fp, tn, fn,
        detection_rate, false_positive_rate, false_negative_rate,
        n_true_anomalies, n_detected_anomalies
    """
    y_true_bin = _to_binary(y_true)
    y_pred_bin = _to_binary(y_pred)

    if y_true_bin.sum() == 0:
        return {
            "error": "No anomalies in ground-truth labels. Cannot compute metrics.",
            "precision": None, "recall": None, "f1": None,
            "roc_auc": None, "pr_auc": None,
        }

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true_bin, norm_scores)
    except Exception:
        roc_auc = None

    try:
        pr_auc = average_precision_score(y_true_bin, norm_scores)
    except Exception:
        pr_auc = None

    n_pred_pos = int(y_pred_bin.sum())
    n_true_pos = int(y_true_bin.sum())

    detection_rate = float(tp / n_true_pos) if n_true_pos > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "pr_auc": round(float(pr_auc), 4) if pr_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "detection_rate": round(detection_rate, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "n_true_anomalies": n_true_pos,
        "n_detected_anomalies": n_pred_pos,
        "supervised": True,
    }


# ── Unsupervised validation metrics ───────────────────────────────────────────

def unsupervised_metrics(
    norm_scores: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute unsupervised validation statistics.

    These metrics do NOT require ground-truth labels.
    They evaluate internal properties of the anomaly score distribution.

    Parameters
    ----------
    norm_scores : np.ndarray, shape (n_samples,)
    predictions : np.ndarray, shape (n_samples,)

    Returns
    -------
    dict with unsupervised validation statistics.
    """
    anomaly_mask = predictions == -1
    normal_mask = predictions == 1

    anom_scores = norm_scores[anomaly_mask]
    norm_scores_normal = norm_scores[normal_mask]

    n_anomalies = int(anomaly_mask.sum())
    n_normal = int(normal_mask.sum())
    n_total = len(norm_scores)

    # Score separation: difference between anomaly and normal score means
    # A larger gap indicates better discrimination.
    if len(anom_scores) > 0 and len(norm_scores_normal) > 0:
        score_separation = float(
            np.mean(anom_scores) - np.mean(norm_scores_normal)
        )
        # Normalised separation (Cohen's d equivalent)
        pooled_std = np.sqrt(
            (np.var(anom_scores) + np.var(norm_scores_normal)) / 2
        )
        normalised_separation = (
            float(score_separation / pooled_std)
            if pooled_std > 1e-10 else 0.0
        )
    else:
        score_separation = 0.0
        normalised_separation = 0.0

    return {
        "n_total": n_total,
        "n_anomalies": n_anomalies,
        "n_normal": n_normal,
        "anomaly_fraction": round(n_anomalies / max(n_total, 1), 4),
        "anomaly_score_mean": round(float(np.mean(anom_scores)), 4) if len(anom_scores) > 0 else None,
        "anomaly_score_std": round(float(np.std(anom_scores)), 4) if len(anom_scores) > 0 else None,
        "normal_score_mean": round(float(np.mean(norm_scores_normal)), 4) if len(norm_scores_normal) > 0 else None,
        "normal_score_std": round(float(np.std(norm_scores_normal)), 4) if len(norm_scores_normal) > 0 else None,
        "score_separation": round(score_separation, 4),
        "normalised_separation": round(normalised_separation, 4),
        "overall_mean_score": round(float(np.mean(norm_scores)), 4),
        "overall_score_std": round(float(np.std(norm_scores)), 4),
        "supervised": False,
        "note": (
            "No ground-truth labels available. "
            "Metrics reflect internal score distribution properties only. "
            "Objective evaluation requires labelled data."
        ),
    }


# ── Convenience wrapper ────────────────────────────────────────────────────────

def evaluate(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    norm_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Auto-select supervised or unsupervised evaluation based on label availability.

    Parameters
    ----------
    y_true : np.ndarray or None
    y_pred : np.ndarray
    norm_scores : np.ndarray

    Returns
    -------
    dict of evaluation metrics.
    """
    unsup = unsupervised_metrics(norm_scores, y_pred)

    if y_true is not None:
        sup = supervised_metrics(y_true, y_pred, norm_scores)
        return {**unsup, **sup, "has_labels": True}
    else:
        return {**unsup, "has_labels": False}

"""
src/evaluation/model_comparison.py

Model comparison framework for the Explainable Anomaly Detection Framework.

Compares anomaly detection models on:
1. Anomaly count and fraction
2. Runtime
3. Score separation (unsupervised)
4. Model agreement (cross-model consensus)
5. Supervised metrics (if labels available)
6. Stability across random seeds (optional)

IMPORTANT:
- Supervised metrics (F1, ROC-AUC etc.) require ground-truth labels.
- When labels are absent, those columns show N/A.
- Never fabricate or estimate supervised metrics.
- Different algorithms have different assumptions — comparison is
  meant to guide selection, not to declare one model universally best.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.detection.scoring import normalize_scores
from src.evaluation.metrics import evaluate, supervised_metrics, unsupervised_metrics


# ── Main comparison function ───────────────────────────────────────────────────

def compare_models(
    results: Dict[str, Dict[str, Any]],
    y_true: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build a comparison table across all fitted models.

    Parameters
    ----------
    results : dict
        {model_name: result_dict} from ModelManager.get_all_results().
        Each result must contain: scores, predictions, fit_time_sec.
    y_true : np.ndarray or None
        Ground-truth labels for supervised evaluation.

    Returns
    -------
    pd.DataFrame
        One row per model.  Columns depend on whether labels are available.
    """
    rows = []

    for model_name, result in results.items():
        if not result.get("success", False):
            rows.append(_failed_row(model_name, result.get("error", "Unknown error")))
            continue

        raw_scores = result["scores"]
        predictions = result["predictions"]
        fit_time = result.get("fit_time_sec", 0.0)

        norm_scores = normalize_scores(raw_scores)

        # Unsupervised metrics
        unsup = unsupervised_metrics(norm_scores, predictions)

        row = {
            "Model": model_name,
            "Anomalies": result.get("n_anomalies", "N/A"),
            "Anomaly %": f"{result.get('anomaly_fraction', 0)*100:.1f}%",
            "Runtime (s)": f"{fit_time:.3f}",
            "Score Sep.": f"{unsup['score_separation']:.4f}",
            "Norm. Sep.": f"{unsup['normalised_separation']:.3f}",
            "Anom. Score Mean": f"{unsup.get('anomaly_score_mean', 0.0):.3f}",
            "Normal Score Mean": f"{unsup.get('normal_score_mean', 0.0):.3f}",
        }

        # Supervised metrics (only if labels available)
        if y_true is not None:
            sup = supervised_metrics(y_true, predictions, norm_scores)
            row["Precision"] = (
                f"{sup['precision']:.3f}" if sup["precision"] is not None else "N/A"
            )
            row["Recall"] = (
                f"{sup['recall']:.3f}" if sup["recall"] is not None else "N/A"
            )
            row["F1"] = f"{sup['f1']:.3f}" if sup["f1"] is not None else "N/A"
            row["ROC-AUC"] = (
                f"{sup['roc_auc']:.3f}" if sup["roc_auc"] is not None else "N/A"
            )
            row["PR-AUC"] = (
                f"{sup['pr_auc']:.3f}" if sup["pr_auc"] is not None else "N/A"
            )
            row["Detection Rate"] = (
                f"{sup['detection_rate']:.3f}"
                if sup["detection_rate"] is not None else "N/A"
            )
        else:
            for col in ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Detection Rate"]:
                row[col] = "N/A (no labels)"

        row["Status"] = "OK"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def _failed_row(model_name: str, error: str) -> Dict[str, Any]:
    """Return a placeholder row for a model that failed to fit."""
    return {
        "Model": model_name,
        "Anomalies": "N/A",
        "Anomaly %": "N/A",
        "Runtime (s)": "N/A",
        "Score Sep.": "N/A",
        "Norm. Sep.": "N/A",
        "Anom. Score Mean": "N/A",
        "Normal Score Mean": "N/A",
        "Precision": "N/A",
        "Recall": "N/A",
        "F1": "N/A",
        "ROC-AUC": "N/A",
        "PR-AUC": "N/A",
        "Detection Rate": "N/A",
        "Status": f"FAILED: {error[:60]}",
    }


# ── Model agreement ────────────────────────────────────────────────────────────

def compute_model_agreement(
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute agreement between models on anomaly decisions.

    An observation is considered "agreed anomaly" if ALL models flag it.
    An observation is considered "disagreed" if some but not all models flag it.

    Parameters
    ----------
    results : dict
        {model_name: result_dict}

    Returns
    -------
    dict with:
        agreed_anomalies : int (all models agree it is anomalous)
        disagreed : int (some models flag it, not all)
        agreed_normal : int (all models agree it is normal)
        agreement_matrix : pd.DataFrame
        pairwise_jaccard : dict of {(m1, m2): jaccard_similarity}
    """
    successful = {
        name: r for name, r in results.items()
        if r.get("success", False) and r["predictions"] is not None
    }

    if len(successful) < 2:
        return {
            "agreed_anomalies": None,
            "disagreed": None,
            "agreed_normal": None,
            "agreement_matrix": None,
            "note": "Need at least 2 successful models for agreement analysis.",
        }

    # Build binary matrix: (n_samples, n_models), 1=anomaly
    names = list(successful.keys())
    pred_arrays = [
        (successful[n]["predictions"] == -1).astype(int)
        for n in names
    ]
    n_samples = len(pred_arrays[0])
    pred_matrix = np.vstack(pred_arrays).T  # (n_samples, n_models)

    # Agreement counts
    vote_sum = pred_matrix.sum(axis=1)  # number of models flagging each obs
    n_models = len(names)
    agreed_anomaly_mask = vote_sum == n_models
    agreed_normal_mask = vote_sum == 0
    disagreed_mask = (vote_sum > 0) & (vote_sum < n_models)

    # Pairwise Jaccard similarity (intersection over union of anomaly sets)
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = pred_arrays[i].astype(bool)
            b = pred_arrays[j].astype(bool)
            intersection = (a & b).sum()
            union = (a | b).sum()
            jaccard = float(intersection / union) if union > 0 else 0.0
            pairwise[(names[i], names[j])] = round(jaccard, 4)

    # Agreement DataFrame (per observation vote count)
    agreement_df = pd.DataFrame(
        pred_matrix, columns=names
    )
    agreement_df["vote_count"] = vote_sum
    agreement_df["agreed_anomaly"] = agreed_anomaly_mask
    agreement_df["agreed_normal"] = agreed_normal_mask
    agreement_df["disagreed"] = disagreed_mask

    return {
        "agreed_anomalies": int(agreed_anomaly_mask.sum()),
        "disagreed": int(disagreed_mask.sum()),
        "agreed_normal": int(agreed_normal_mask.sum()),
        "n_models": n_models,
        "model_names": names,
        "agreement_df": agreement_df,
        "pairwise_jaccard": pairwise,
    }


# ── Stability test ─────────────────────────────────────────────────────────────

def stability_test(
    model_class,
    X: np.ndarray,
    params: Dict[str, Any],
    seeds: List[int] = None,
) -> Dict[str, Any]:
    """
    Test how stable a model's anomaly decisions are across random seeds.

    Parameters
    ----------
    model_class : class
        A detector class (e.g. IsolationForestDetector).
    X : np.ndarray
    params : dict
        Hyperparameters (without random_state).
    seeds : list of int

    Returns
    -------
    dict with:
        mean_anomaly_fraction, std_anomaly_fraction,
        mean_agreement (fraction of observations consistently flagged),
        stability_score (1 = perfectly stable, 0 = maximally unstable)
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1000]

    prediction_list = []
    fractions = []

    for seed in seeds:
        try:
            p = {**params, "random_state": seed}
            # Remove random_state if not supported
            try:
                model = model_class(**p)
            except TypeError:
                p_no_seed = {k: v for k, v in p.items() if k != "random_state"}
                model = model_class(**p_no_seed)

            model.fit(X)
            preds = model.predict(X)
            prediction_list.append(preds)
            fractions.append(float((preds == -1).mean()))
        except Exception:
            continue

    if not prediction_list:
        return {"error": "All stability test runs failed."}

    fractions_arr = np.array(fractions)
    pred_matrix = np.vstack(
        [(p == -1).astype(int) for p in prediction_list]
    ).T  # (n_samples, n_seeds)

    # Fraction of observations consistently flagged (flagged in all or none)
    vote_fractions = pred_matrix.mean(axis=1)
    consistent = np.sum((vote_fractions == 0) | (vote_fractions == 1))
    stability_score = float(consistent / len(vote_fractions))

    return {
        "seeds_tested": len(prediction_list),
        "mean_anomaly_fraction": round(float(fractions_arr.mean()), 4),
        "std_anomaly_fraction": round(float(fractions_arr.std()), 4),
        "stability_score": round(stability_score, 4),
        "note": (
            "Stability score = fraction of observations consistently "
            "classified (either always anomaly or always normal) across seeds."
        ),
    }

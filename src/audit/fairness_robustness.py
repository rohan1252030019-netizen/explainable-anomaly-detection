"""
src/audit/fairness_robustness.py

Algorithmic fairness, subgroup equity, and adversarial noise robustness auditing.

Evaluates:
1. Subgroup Equity & Disparate Impact: Detects if particular operational or demographic
   cohorts (e.g., regions, device tiers) suffer disproportionate false positive rates.
2. Adversarial Noise Perturbation Sensitivity: Injects controlled jitter into feature
   space and audits whether the detector's rankings are stable or volatile.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def evaluate_subgroup_fairness(
    df: pd.DataFrame,
    group_column: str,
    predictions: np.ndarray,
    norm_scores: np.ndarray,
) -> dict[str, Any]:
    """
    Audit anomaly detection fairness and disparate impact across subgroups.

    Parameters
    ----------
    df : pd.DataFrame
    group_column : str
    predictions : np.ndarray, shape (N,) (-1 for anomaly, 1 for normal)
    norm_scores : np.ndarray, shape (N,) in [0, 1]

    Returns
    -------
    dict with per-group statistics and Disparate Impact Ratio.
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in dataframe.")

    preds = np.asarray(predictions)
    scores = np.asarray(norm_scores, dtype=np.float64)

    is_anomaly = preds == -1
    groups = df[group_column].dropna().unique()

    group_stats = []
    rates = []

    for grp in groups:
        mask = (df[group_column] == grp).to_numpy()
        n_grp = int(np.sum(mask))
        if n_grp == 0:
            continue

        n_anom = int(np.sum(is_anomaly[mask]))
        rate = float(n_anom / n_grp)
        rates.append(rate)
        mean_score = float(np.mean(scores[mask]))

        group_stats.append({
            "group": str(grp),
            "total_count": n_grp,
            "anomaly_count": n_anom,
            "anomaly_rate": round(rate, 4),
            "mean_anomaly_score": round(mean_score, 4),
        })

    # Disparate Impact Ratio: min_rate / max_rate
    min_rate = min(rates) if rates else 0.0
    max_rate = max(rates) if rates else 1.0
    dir_ratio = float(min_rate / max_rate) if max_rate > 1e-6 else 1.0

    # 80% Rule for disparate impact
    has_disparate_impact = bool(dir_ratio < 0.80 and max_rate > 0.02)

    return {
        "group_column": group_column,
        "n_subgroups": len(group_stats),
        "disparate_impact_ratio": round(dir_ratio, 4),
        "has_disparate_impact": has_disparate_impact,
        "subgroups": group_stats,
        "summary": (
            f"Disparate impact ratio is {dir_ratio:.2f}. "
            + ("Subgroup disparity exceeds the 80% equity guideline." if has_disparate_impact else "No severe subgroup disparity detected.")
        ),
    }


def evaluate_noise_robustness(
    X: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    noise_stds: list[float] | None = None,
    n_samples: int = 100,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Audit detector robustness against adversarial/sensor Gaussian noise perturbation.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    score_fn : callable f(X) -> np.ndarray of anomaly scores
    noise_stds : list of float, default=[0.02, 0.05, 0.10]
    n_samples : int, default=100

    Returns
    -------
    dict with rank correlation, mean score delta, and stability verdict.
    """
    if noise_stds is None:
        noise_stds = [0.02, 0.05, 0.10]

    X_sub = np.asarray(X, dtype=np.float64)[:n_samples]
    base_scores = np.asarray(score_fn(X_sub), dtype=np.float64)

    rng = np.random.default_rng(random_state)
    std_feature = np.std(X_sub, axis=0)
    std_feature = np.where(std_feature < 1e-6, 1.0, std_feature)

    levels = []
    for sigma in noise_stds:
        noise = rng.normal(0, sigma, size=X_sub.shape) * std_feature
        perturbed_X = X_sub + noise
        pert_scores = np.asarray(score_fn(perturbed_X), dtype=np.float64)

        # Metrics
        mean_abs_delta = float(np.mean(np.abs(pert_scores - base_scores)))
        spearman_res = stats.spearmanr(base_scores, pert_scores)
        rho = float(spearman_res.statistic) if hasattr(spearman_res, "statistic") else float(spearman_res[0])
        rho = 0.0 if np.isnan(rho) else rho

        levels.append({
            "noise_level_sigma": sigma,
            "mean_score_delta": round(mean_abs_delta, 4),
            "rank_correlation_rho": round(rho, 4),
            "rank_retention_pct": round(max(0.0, rho) * 100, 1),
        })

    # Overall robustness tier based on largest tested noise level
    final_rho = levels[-1]["rank_correlation_rho"]
    if final_rho >= 0.85:
        verdict = "High Robustness"
        explanation = "Model rankings remain highly stable under sensory and adversarial perturbations."
    elif final_rho >= 0.65:
        verdict = "Moderate Sensitivity"
        explanation = "Moderate rank shifts observed under perturbation; monitor sensitive boundaries."
    else:
        verdict = "Vulnerable to Noise"
        explanation = "Significant ranking degradation under perturbation; regularization recommended."

    return {
        "n_tested_samples": len(X_sub),
        "robustness_verdict": verdict,
        "summary": explanation,
        "perturbation_levels": levels,
    }

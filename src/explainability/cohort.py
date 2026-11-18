"""
src/explainability/cohort.py

Cohort-level explanations and subgroup distribution shift analysis.

Enables analysts to understand what characterizes an entire cohort of anomalies
(e.g. all severe anomalies, fraud cluster A, or high-conformal-significance items)
relative to normal reference data.

Computes:
1. Feature Mean Shifts & Standardized Effect Sizes (Cohen's d)
2. Non-parametric Kolmogorov-Smirnov distributional divergence tests
3. Dominant distinguishing features ranked by effect magnitude
4. Cohort archetype summary
"""

from typing import Any

import numpy as np
from scipy import stats


def explain_cohort(
    cohort_X: np.ndarray,
    reference_X: np.ndarray,
    feature_names: list[str],
    cohort_name: str = "Flagged Anomalies",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Produce a cohort-level explanation contrasting cohort_X with reference_X.

    Parameters
    ----------
    cohort_X : np.ndarray, shape (n_cohort, n_features)
    reference_X : np.ndarray, shape (n_ref, n_features)
    feature_names : list of str
    cohort_name : str
    top_k : int

    Returns
    -------
    dict containing:
        - cohort_name: str
        - n_cohort: int
        - n_reference: int
        - feature_shifts: list of dicts with shift statistics per feature
        - dominant_drivers: list of top k most shifted features
        - summary: str natural-language archetype description
    """
    X_c = np.asarray(cohort_X, dtype=np.float64)
    X_r = np.asarray(reference_X, dtype=np.float64)

    if X_c.ndim == 1:
        X_c = X_c.reshape(1, -1)
    if X_r.ndim == 1:
        X_r = X_r.reshape(1, -1)

    n_features = len(feature_names)
    shifts = []

    for j in range(n_features):
        c_vals = X_c[:, j]
        r_vals = X_r[:, j]

        mean_c = float(np.mean(c_vals))
        mean_r = float(np.mean(r_vals))
        std_r = float(np.std(r_vals))
        std_r = std_r if std_r > 1e-6 else 1.0

        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(c_vals) + np.var(r_vals)) / 2.0)
        pooled_std = float(pooled_std) if pooled_std > 1e-6 else 1.0
        effect_size = (mean_c - mean_r) / pooled_std

        # Kolmogorov-Smirnov test for distribution divergence
        ks_res = stats.ks_2samp(c_vals, r_vals)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)

        shifts.append({
            "feature": feature_names[j],
            "cohort_mean": round(mean_c, 3),
            "reference_mean": round(mean_r, 3),
            "standardized_shift": round(float((mean_c - mean_r) / std_r), 3),
            "effect_size_d": round(float(effect_size), 3),
            "ks_statistic": round(ks_stat, 3),
            "ks_pvalue": round(ks_pval, 4),
            "direction": "Elevated" if mean_c > mean_r else "Suppressed",
        })

    # Rank features by absolute effect size
    shifts.sort(key=lambda s: abs(s["effect_size_d"]), reverse=True)
    dominant_drivers = shifts[:top_k]

    # Generate archetype summary
    top_elevated = [d["feature"] for d in dominant_drivers if d["direction"] == "Elevated"]
    top_suppressed = [d["feature"] for d in dominant_drivers if d["direction"] == "Suppressed"]

    summary_parts = [f"Cohort '{cohort_name}' (N={len(X_c)}) shows pronounced distribution shift."]
    if top_elevated:
        summary_parts.append(f"Substantially elevated in: {', '.join(top_elevated)}.")
    if top_suppressed:
        summary_parts.append(f"Substantially suppressed in: {', '.join(top_suppressed)}.")

    return {
        "cohort_name": cohort_name,
        "n_cohort": int(len(X_c)),
        "n_reference": int(len(X_r)),
        "dominant_drivers": dominant_drivers,
        "all_feature_shifts": shifts,
        "summary": " ".join(summary_parts),
    }

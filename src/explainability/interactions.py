"""
src/explainability/interactions.py

Feature interaction analysis for anomaly detection.

Feature interactions reveal when an anomaly is caused by a non-additive,
compound combination of two features rather than a single extreme value
(e.g., normal transaction amount AND normal geographic location individually,
but their combined co-occurrence is anomalous).

Supports:
1. Exact TreeSHAP Interaction Values (for tree models / XGBoost surrogate)
2. Model-Agnostic Pairwise Interaction Index (Friedman's H-statistic proxy / 2nd-order difference)
"""

from typing import Any

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def compute_pairwise_interactions(
    model_or_score_fn: Any,
    X: np.ndarray,
    feature_names: list[str],
    top_n_pairs: int = 5,
    max_samples: int = 50,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Compute pairwise feature interactions on sample rows X.

    Parameters
    ----------
    model_or_score_fn : model object or callable score_fn(X) -> scores
    X : np.ndarray, shape (n_samples, n_features)
    feature_names : list of str
    top_n_pairs : int, default=5
    max_samples : int, default=50
    random_state : int, default=42

    Returns
    -------
    list of dicts describing top interacting feature pairs:
        - feature_1: str
        - feature_2: str
        - interaction_strength: float
        - interpretation: str
    """
    X_sub = np.asarray(X, dtype=np.float64)[:max_samples]
    _, n_features = X_sub.shape

    # 1. Attempt TreeSHAP interaction values if tree model
    if SHAP_AVAILABLE and hasattr(model_or_score_fn, "predict"):
        try:
            tree_explainer = shap.TreeExplainer(model_or_score_fn)
            interactions = tree_explainer.shap_interaction_values(X_sub)
            # interactions shape: (n_samples, n_features, n_features)
            mean_interactions = np.mean(np.abs(interactions), axis=0)

            pairs = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    strength = float(mean_interactions[i, j] + mean_interactions[j, i])
                    pairs.append({
                        "feature_1": feature_names[i],
                        "feature_2": feature_names[j],
                        "interaction_strength": round(strength, 4),
                        "method": "TreeSHAP Interaction",
                    })

            pairs.sort(key=lambda x: x["interaction_strength"], reverse=True)
            for p in pairs[:top_n_pairs]:
                p["interpretation"] = (
                    f"Compound synergy between {p['feature_1']} and {p['feature_2']} "
                    f"amplifies anomaly risk non-linearly."
                )
            return pairs[:top_n_pairs]
        except Exception:
            pass  # Fall back to model-agnostic 2nd-order difference

    # 2. Agnostic 2nd-order interaction via bivariate perturbation
    score_fn = (
        model_or_score_fn
        if callable(model_or_score_fn)
        else getattr(model_or_score_fn, "score_samples", None)
        or getattr(model_or_score_fn, "decision_function", None)
    )
    if score_fn is None:
        return []

    # Estimate empirical feature standard deviations
    stds = np.std(X_sub, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)

    base_scores = np.asarray(score_fn(X_sub), dtype=np.float64)

    pair_strengths = []
    # Test all pairs or random sample of pairs if features are large
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Bivariate second-order difference:
            # Delta_ij = f(x + delta_i + delta_k) - f(x + delta_i) - f(x + delta_j) + f(x)
            delta_i = np.zeros_like(X_sub)
            delta_j = np.zeros_like(X_sub)
            delta_i[:, i] = stds[i] * 0.5
            delta_j[:, j] = stds[j] * 0.5

            X_i = X_sub + delta_i
            X_j = X_sub + delta_j
            X_ij = X_sub + delta_i + delta_j

            s_i = np.asarray(score_fn(X_i), dtype=np.float64)
            s_j = np.asarray(score_fn(X_j), dtype=np.float64)
            s_ij = np.asarray(score_fn(X_ij), dtype=np.float64)

            # Second order interaction magnitude
            interaction = np.mean(np.abs(s_ij - s_i - s_j + base_scores))
            pair_strengths.append({
                "feature_1": feature_names[i],
                "feature_2": feature_names[j],
                "interaction_strength": round(float(interaction), 4),
                "method": "Bivariate 2nd-Order Perturbation",
                "interpretation": (
                    f"Non-additive joint effect between {feature_names[i]} and {feature_names[j]} "
                    f"influences detector sensitivity."
                ),
            })

    pair_strengths.sort(key=lambda x: x["interaction_strength"], reverse=True)
    return pair_strengths[:top_n_pairs]

"""
src/explainability/lime_explainer.py

LIME (Local Interpretable Model-agnostic Explanations) for anomaly scores,
and cross-method explanation agreement scoring.

LIME approximates any non-linear anomaly scoring function locally around an
observation x_0 by fitting an interpretable, distance-weighted linear surrogate:
    min_w sum_i pi(z_i) * (y_i - w^T (z_i - x_0) - b)^2 + alpha * ||w||_2^2

where:
    - z_i are perturbed instances drawn around x_0
    - pi(z_i) = exp(-||z_i - x_0||^2 / sigma^2) is an exponential RBF distance kernel
    - y_i = score_fn(z_i) is the anomaly score of the perturbed instance

The resulting coefficients w represent the local marginal contribution of each
feature to the anomaly score.

Explanation Agreement:
Cross-checks attributions between SHAP and LIME (or statistical perturbations)
using Spearman rank correlation, top-k Jaccard similarity, and sign concordance.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge


class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explainer for anomaly scoring models.

    Parameters
    ----------
    score_fn : callable
        Function f(X) -> np.ndarray of anomaly scores in [0, 1].
    feature_names : list of str
    X_background : np.ndarray
        Background training/reference data to compute feature standard deviations.
    n_samples : int, default=300
        Number of local perturbations to sample.
    kernel_width : float or None, default=None
        Kernel width sigma for exponential weighting. Defaults to 0.75 * sqrt(d).
    random_state : int, default=42
    """

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: list[str],
        X_background: np.ndarray,
        n_samples: int = 300,
        kernel_width: float | None = None,
        random_state: int = 42,
    ):
        self.score_fn = score_fn
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.random_state = random_state

        X_bg = np.asarray(X_background, dtype=np.float64)
        self.d = X_bg.shape[1]
        self.feature_stds = np.std(X_bg, axis=0)
        # Avoid division by zero on constant features
        self.feature_stds = np.where(self.feature_stds < 1e-6, 1.0, self.feature_stds)

        if kernel_width is None:
            self.kernel_width = float(0.75 * np.sqrt(self.d))
        else:
            self.kernel_width = float(kernel_width)

    def explain_instance(
        self,
        x: np.ndarray,
        top_k: int = 8,
    ) -> dict[str, Any]:
        """
        Explain a single observation x locally.

        Parameters
        ----------
        x : np.ndarray of shape (d,) or (1, d)
        top_k : int

        Returns
        -------
        dict containing:
            - attributions: dict of feature_name -> float coefficient
            - top_features: list of (feature_name, coefficient)
            - base_score: float, score of unperturbed x
            - r2_score: float, fidelity of the local linear surrogate
        """
        x_0 = np.asarray(x, dtype=np.float64).ravel()
        if len(x_0) != self.d:
            raise ValueError(f"Instance dimension {len(x_0)} does not match background dimension {self.d}")

        rng = np.random.default_rng(self.random_state)

        # 1. Generate local perturbations centered around x_0
        noise = rng.normal(0, 1.0, size=(self.n_samples, self.d)) * self.feature_stds * 0.5
        perturbations = x_0 + noise
        # Include original point as first row
        perturbations[0] = x_0

        # 2. Evaluate scores
        scores = self.score_fn(perturbations)
        base_score = float(scores[0])

        # 3. Compute distance in standardized space and exponential kernel weights
        std_diff = (perturbations - x_0) / self.feature_stds
        distances = np.linalg.norm(std_diff, axis=1)
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))

        # 4. Fit local weighted Ridge regression
        centered_X = perturbations - x_0
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(centered_X, scores, sample_weight=weights)

        coefs = ridge.coef_
        r2 = float(ridge.score(centered_X, scores, sample_weight=weights))

        attributions = {name: float(coef) for name, coef in zip(self.feature_names, coefs, strict=False)}
        sorted_pairs = sorted(attributions.items(), key=lambda item: abs(item[1]), reverse=True)
        top_features = [{"feature": k, "attribution": v} for k, v in sorted_pairs[:top_k]]

        return {
            "attributions": attributions,
            "top_features": top_features,
            "base_score": base_score,
            "local_r2": max(0.0, r2),
            "method": "LIME",
        }


def compute_explanation_agreement(
    attributions_a: dict[str, float] | Sequence[float],
    attributions_b: dict[str, float] | Sequence[float],
    feature_names: Sequence[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Quantify agreement between two explanation methods (e.g. SHAP vs LIME).

    Computes:
    1. Spearman Rank Correlation (monotonic ordering agreement)
    2. Top-K Jaccard Overlap (consensus on top driver features)
    3. Sign Concordance (% of features where both methods agree on direction)
    4. Categorical Agreement Tier: "High Agreement", "Moderate Agreement", "Low / Divergent"

    Parameters
    ----------
    attributions_a : dict or sequence of floats
    attributions_b : dict or sequence of floats
    feature_names : sequence of str or None
    top_k : int

    Returns
    -------
    dict of metrics and interpretation
    """
    if isinstance(attributions_a, dict) and isinstance(attributions_b, dict):
        keys = [k for k in attributions_a if k in attributions_b]
        vec_a = np.array([attributions_a[k] for k in keys], dtype=np.float64)
        vec_b = np.array([attributions_b[k] for k in keys], dtype=np.float64)
        names = keys
    elif feature_names is not None:
        vec_a = np.asarray(attributions_a, dtype=np.float64)
        vec_b = np.asarray(attributions_b, dtype=np.float64)
        names = list(feature_names)
    else:
        vec_a = np.asarray(attributions_a, dtype=np.float64)
        vec_b = np.asarray(attributions_b, dtype=np.float64)
        names = [f"f{i}" for i in range(len(vec_a))]

    if len(vec_a) != len(vec_b):
        raise ValueError("Attribution vectors must have matching length")

    # Handle degenerate zero-variance cases
    if np.all(vec_a == 0) or np.all(vec_b == 0) or np.std(vec_a) < 1e-9 or np.std(vec_b) < 1e-9:
        spearman_rho = 0.0
        p_value = 1.0
    else:
        res = stats.spearmanr(vec_a, vec_b)
        spearman_rho = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        p_value = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
        if np.isnan(spearman_rho):
            spearman_rho = 0.0

    # Top-K Jaccard similarity based on absolute magnitude
    k_actual = min(top_k, len(names))
    top_indices_a = set(np.argsort(np.abs(vec_a))[-k_actual:])
    top_indices_b = set(np.argsort(np.abs(vec_b))[-k_actual:])

    intersection = top_indices_a.intersection(top_indices_b)
    union = top_indices_a.union(top_indices_b)
    jaccard = len(intersection) / len(union) if union else 1.0

    # Sign concordance
    nonzero = (vec_a != 0) & (vec_b != 0)
    if np.any(nonzero):
        sign_agreement = float(np.mean(np.sign(vec_a[nonzero]) == np.sign(vec_b[nonzero])))
    else:
        sign_agreement = 1.0

    # Agreement consensus tier
    if spearman_rho >= 0.70 and jaccard >= 0.60:
        agreement_level = "High Agreement"
        summary = "Both explanation methods strongly agree on primary anomaly drivers."
    elif spearman_rho >= 0.35 or jaccard >= 0.40:
        agreement_level = "Moderate Agreement"
        summary = "Methods agree on key risk directions but differ in secondary ranking."
    else:
        agreement_level = "Low / Divergent"
        summary = "Explanations diverge; inspect local non-linearities and surrogate fit fidelity."

    top_agreed_features = [names[i] for i in intersection]

    return {
        "spearman_rank_correlation": round(spearman_rho, 4),
        "spearman_p_value": round(p_value, 4),
        "top_k_jaccard_overlap": round(jaccard, 4),
        "sign_concordance": round(sign_agreement, 4),
        "agreement_level": agreement_level,
        "summary": summary,
        "agreed_top_features": top_agreed_features,
    }

"""
src/explainability/explanation_generator.py

High-level explanation generator for the Explainable Anomaly Detection Framework.

This module ties together:
- Statistical deviation analysis
- Perturbation-based feature importance
- SHAP (when available)

And produces:
- Local explanations (per anomalous observation)
- Global explanations (across all anomalies)
- Formatted explanation tables (for display / export)

IMPORTANT:
- We use association language, not causal language.
- Explanations describe model behaviour, not ground truth.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.explainability.feature_contributions import (
    compute_statistical_deviations,
    perturbation_importance,
    combined_feature_contributions,
    global_feature_importance,
)
from src.explainability.shap_explainer import SHAPExplainer


class ExplanationGenerator:
    """
    Generates local and global explanations for detected anomalies.

    Parameters
    ----------
    model_name : str
    sklearn_model : fitted sklearn model object
    X_train : np.ndarray
        Preprocessed training data.
    feature_names : list of str
    reference_stats : dict
        From PreprocessingPipeline.get_reference_stats().
    norm_score_fn : callable
        Function (X: np.ndarray) -> np.ndarray of normalised scores.
        Used for perturbation importance.
    use_shap : bool
        Whether to attempt SHAP explanations.
    top_k : int
        Number of top features to include in each local explanation.
    """

    def __init__(
        self,
        model_name: str,
        sklearn_model: Any,
        X_train: np.ndarray,
        feature_names: List[str],
        reference_stats: Dict[str, Any],
        norm_score_fn: Callable,
        use_shap: bool = True,
        top_k: int = 8,
    ):
        self.model_name = model_name
        self.sklearn_model = sklearn_model
        self.X_train = X_train
        self.feature_names = feature_names
        self.reference_stats = reference_stats
        self.norm_score_fn = norm_score_fn
        self.top_k = top_k

        # Build SHAP explainer
        self._shap_explainer: Optional[SHAPExplainer] = None
        if use_shap:
            self._shap_explainer = SHAPExplainer(
                model_name=model_name,
                sklearn_model=sklearn_model,
                X_train=X_train,
                feature_names=feature_names,
                score_fn=norm_score_fn,
            )

        # Cache global explanations
        self._global_importance: Optional[pd.DataFrame] = None

    # ── Local explanation ──────────────────────────────────────────────────────

    def explain_observation(
        self,
        observation: np.ndarray,
        norm_score: float,
        severity: str,
        row_id: Any = None,
        original_values: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a full local explanation for a single observation.

        Parameters
        ----------
        observation : np.ndarray, shape (n_features,)
            Preprocessed (scaled) feature vector.
        norm_score : float
            Normalised anomaly score [0, 1].
        severity : str
            "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        row_id : any
            Row identifier (e.g. index or transaction_id).
        original_values : dict or None
            Raw (un-preprocessed) feature values for display.

        Returns
        -------
        dict with keys:
            row_id, model_name, norm_score, severity,
            top_features, shap_values, explanation_text,
            confidence_note
        """
        # A. Statistical deviations
        stat_devs = compute_statistical_deviations(
            observation=observation,
            feature_names=self.feature_names,
            reference_stats=self.reference_stats,
        )

        # B. Perturbation importance
        try:
            perturb_imps = perturbation_importance(
                observation=observation,
                score_fn=self.norm_score_fn,
                feature_names=self.feature_names,
            )
        except Exception:
            perturb_imps = None

        # C. Combined contributions
        top_features = combined_feature_contributions(
            stat_devs=stat_devs,
            perturbation_imps=perturb_imps,
            top_k=self.top_k,
        )

        # D. SHAP values (optional)
        shap_values = None
        if self._shap_explainer and self._shap_explainer.is_available():
            try:
                shap_values = self._shap_explainer.explain_single(observation)
            except Exception:
                shap_values = None

        # E. Build feature comparison table
        feature_table = self._build_feature_table(
            top_features, original_values
        )

        # F. Natural language explanation summary
        explanation_text = self._generate_text(
            top_features, norm_score, severity
        )

        return {
            "row_id": row_id,
            "model_name": self.model_name,
            "norm_score": round(float(norm_score), 4),
            "severity": severity,
            "top_features": top_features,
            "feature_table": feature_table,
            "shap_values": shap_values,
            "explanation_text": explanation_text,
            "confidence_note": (
                "This explanation is a model-based approximation. "
                "Feature contributions indicate statistical association, "
                "not causal proof."
            ),
        }

    def explain_batch(
        self,
        X: np.ndarray,
        norm_scores: np.ndarray,
        predictions: np.ndarray,
        severity: np.ndarray,
        original_df: Optional[pd.DataFrame] = None,
        max_explanations: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for all anomalous observations.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        norm_scores : np.ndarray, shape (n_samples,)
        predictions : np.ndarray, shape (n_samples,)
        severity : np.ndarray, shape (n_samples,)
        original_df : pd.DataFrame or None
        max_explanations : int
            Cap to avoid excessive computation.

        Returns
        -------
        list of explanation dicts
        """
        anomaly_indices = np.where(predictions == -1)[0]
        # Sort by score (most anomalous first)
        anomaly_indices = sorted(
            anomaly_indices,
            key=lambda i: norm_scores[i],
            reverse=True,
        )[:max_explanations]

        explanations = []
        for idx in anomaly_indices:
            obs = X[idx]
            score = float(norm_scores[idx])
            sev = str(severity[idx])

            orig_vals = None
            if original_df is not None and idx < len(original_df):
                orig_vals = original_df.iloc[idx].to_dict()

            exp = self.explain_observation(
                observation=obs,
                norm_score=score,
                severity=sev,
                row_id=idx,
                original_values=orig_vals,
            )
            explanations.append(exp)

        return explanations

    # ── Global explanation ─────────────────────────────────────────────────────

    def compute_global_importance(
        self,
        explanations: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Aggregate local explanations into global feature importance.

        Parameters
        ----------
        explanations : list of dicts from explain_batch()

        Returns
        -------
        pd.DataFrame with columns: feature, mean_combined_score, frequency,
                                   mean_z_score, rank
        """
        all_contribs = [exp["top_features"] for exp in explanations]
        if not all_contribs:
            return pd.DataFrame(
                columns=["feature", "mean_combined_score", "frequency",
                         "mean_z_score", "rank"]
            )

        self._global_importance = global_feature_importance(
            anomaly_contributions=all_contribs,
            feature_names=self.feature_names,
        )
        return self._global_importance

    def get_global_importance(self) -> Optional[pd.DataFrame]:
        return self._global_importance

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_feature_table(
        self,
        top_features: List[Dict[str, Any]],
        original_values: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Build a display-ready feature comparison table.

        Columns: Feature | Observed Value | Typical Value | Deviation |
                 Contribution | Direction | Interpretation
        """
        rows = []
        for entry in top_features:
            feat = entry["feature"]
            obs_val = entry["observed_value"]
            ref_val = entry["reference_value"]
            z = entry.get("robust_z_score", 0.0)
            direction = entry["deviation_direction"]
            contribution = entry["combined_score"]

            # Use original (unscaled) values for display when available
            if original_values and feat in original_values:
                raw_val = original_values[feat]
                display_obs = round(float(raw_val), 4) if isinstance(raw_val, (int, float, np.floating, np.integer)) else raw_val
                # reference_stats hold pre-scaling medians for numerical features
                display_ref = round(ref_val, 4) if entry["feature_type"] == "numerical" else "N/A"
            else:
                display_obs = round(obs_val, 4)
                display_ref = round(ref_val, 4)

            # Deviation percentage — use original-space values for numerical
            if entry["feature_type"] == "numerical":
                # obs_numeric and ref_numeric should both be in original space
                obs_numeric = float(display_obs) if isinstance(display_obs, (int, float, np.floating, np.integer)) else obs_val
                ref_numeric = float(display_ref) if isinstance(display_ref, (int, float, np.floating, np.integer)) else ref_val
                if abs(ref_numeric) > 1e-6:
                    pct_dev = (obs_numeric - ref_numeric) / abs(ref_numeric) * 100
                    deviation_str = f"{'+' if pct_dev >= 0 else ''}{pct_dev:.1f}%"
                else:
                    deviation_str = f"z={z:+.2f}"
            else:
                deviation_str = "active" if direction == "active" else "inactive"

            # Contribution label
            if contribution >= 0.7:
                contrib_label = "Very High"
            elif contribution >= 0.4:
                contrib_label = "High"
            elif contribution >= 0.2:
                contrib_label = "Medium"
            else:
                contrib_label = "Low"

            # Interpretation
            interpretation = _interpretation_text(
                feat, direction, abs(z), contribution, entry["feature_type"]
            )

            rows.append({
                "Feature": feat,
                "Observed": display_obs,
                "Typical (Median)": display_ref,
                "Deviation": deviation_str,
                "Contribution": contrib_label,
                "Direction": direction,
                "Interpretation": interpretation,
            })

        return pd.DataFrame(rows)

    def _generate_text(
        self,
        top_features: List[Dict[str, Any]],
        norm_score: float,
        severity: str,
    ) -> str:
        """Generate a concise natural-language explanation."""
        if not top_features:
            return (
                f"This observation received an anomaly score of "
                f"{norm_score:.3f} ({severity} severity). "
                "No dominant contributing features were identified."
            )

        top3 = top_features[:3]
        feat_desc = []
        for f in top3:
            fname = f["feature"]
            direction = f["deviation_direction"]
            z = abs(f.get("robust_z_score", 0.0))

            if f["feature_type"] == "numerical":
                if z >= 3:
                    intensity = "extremely"
                elif z >= 2:
                    intensity = "significantly"
                else:
                    intensity = "somewhat"
                feat_desc.append(
                    f"'{fname}' ({intensity} {direction} typical values, "
                    f"|z|={z:.1f})"
                )
            else:
                feat_desc.append(f"'{fname}' ({direction})")

        feat_str = "; ".join(feat_desc)
        return (
            f"This observation has an anomaly score of {norm_score:.3f} "
            f"(severity: {severity}). "
            f"The primary contributing features are: {feat_str}. "
            "Note: these are model-based feature associations, "
            "not causal explanations."
        )

    def get_shap_status(self) -> Dict[str, Any]:
        if self._shap_explainer:
            return self._shap_explainer.get_status()
        return {"shap_available": False, "is_available": False, "error": "SHAP not initialised"}


# ── Interpretation helper ──────────────────────────────────────────────────────

def _interpretation_text(
    feature: str,
    direction: str,
    z_score: float,
    contribution: float,
    feat_type: str,
) -> str:
    """Generate a short interpretation string for a single feature."""
    if feat_type == "categorical_ohe":
        if direction == "active":
            return f"Category '{feature}' is active; uncommon in normal samples."
        else:
            return f"Category '{feature}' is inactive."

    if z_score >= 4:
        intensity = "extremely unusual"
    elif z_score >= 3:
        intensity = "highly unusual"
    elif z_score >= 2:
        intensity = "noticeably unusual"
    elif z_score >= 1:
        intensity = "mildly unusual"
    else:
        intensity = "within normal range"

    return (
        f"Value is {intensity} compared to training distribution "
        f"({direction} typical range)."
    )

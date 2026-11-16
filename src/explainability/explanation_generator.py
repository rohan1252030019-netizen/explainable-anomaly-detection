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

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.explainability.feature_contributions import (
    combined_feature_contributions,
    compute_statistical_deviations,
    global_feature_importance,
    perturbation_importance,
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
    score_space : dict or None
        Output of ``ScoreNormalizer.describe()``.  Passed to the SHAP
        explainer so attributions are expressed in normalised score units.
    denormalize : callable or None
        ``f(scaled_row) -> raw_row``.  Normally
        ``PreprocessingPipeline.denormalize``.  Required for the statistical
        deviations to be meaningful: ``reference_stats`` is stored in raw units,
        so the observation must be converted before being compared with it.
    score_fn_for_shap : callable or None
        Must return the SAME normalised score the dashboard displays.  The SHAP
        explainer verifies additivity against it and falls back to a
        model-agnostic permutation explainer if a tree explainer's units do not
        line up.
    """

    def __init__(
        self,
        model_name: str,
        sklearn_model: Any,
        X_train: np.ndarray,
        feature_names: list[str],
        reference_stats: dict[str, Any],
        norm_score_fn: Callable,
        use_shap: bool = True,
        top_k: int = 8,
        denormalize: Callable | None = None,
        score_space: dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.sklearn_model = sklearn_model
        self.X_train = X_train
        self.feature_names = feature_names
        self.reference_stats = reference_stats
        self.norm_score_fn = norm_score_fn
        self.top_k = top_k
        self.denormalize = denormalize
        self.score_space = score_space

        # Build SHAP explainer
        self._shap_explainer: SHAPExplainer | None = None
        if use_shap:
            self._shap_explainer = SHAPExplainer(
                model_name=model_name,
                sklearn_model=sklearn_model,
                X_train=X_train,
                feature_names=feature_names,
                score_fn=norm_score_fn,
                score_space=score_space,
            )

        # Cache global explanations
        self._global_importance: pd.DataFrame | None = None

    # ── Local explanation ──────────────────────────────────────────────────────

    def explain_observation(
        self,
        observation: np.ndarray,
        norm_score: float,
        severity: str,
        row_id: Any = None,
        original_values: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        # A. Statistical deviations (in raw units, so the z-scores are real)
        raw_observation = None
        if self.denormalize is not None:
            try:
                raw_observation = self.denormalize(np.asarray(observation))
            except Exception:
                raw_observation = None

        stat_devs = compute_statistical_deviations(
            observation=observation,
            feature_names=self.feature_names,
            reference_stats=self.reference_stats,
            raw_observation=raw_observation,
        )

        # B. Perturbation importance
        #
        # A failure here used to be swallowed by a bare `except Exception`,
        # which silently downgraded every explanation to statistics-only while
        # the UI still claimed a combined contribution score. Failures are now
        # recorded and surfaced.
        perturbation_error = None
        try:
            perturb_imps = perturbation_importance(
                observation=observation,
                score_fn=self.norm_score_fn,
                feature_names=self.feature_names,
            )
        except Exception as exc:
            perturb_imps = None
            perturbation_error = str(exc)

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
            "used_perturbation": perturb_imps is not None,
            "perturbation_error": perturbation_error,
            "confidence_note": (
                "This explanation is a model-based approximation. "
                "Feature contributions indicate statistical association, "
                "not causal proof."
                + (
                    " Perturbation-based importance was unavailable for this "
                    "row, so the ranking rests on statistical deviation alone."
                    if perturbation_error
                    else ""
                )
            ),
        }

    def explain_batch(
        self,
        X: np.ndarray,
        norm_scores: np.ndarray,
        predictions: np.ndarray,
        severity: np.ndarray,
        original_df: pd.DataFrame | None = None,
        max_explanations: int = 200,
    ) -> list[dict[str, Any]]:
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
        explanations: list[dict[str, Any]],
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

    def get_global_importance(self) -> pd.DataFrame | None:
        return self._global_importance

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_feature_table(
        self,
        top_features: list[dict[str, Any]],
        original_values: dict[str, Any] | None,
    ) -> pd.DataFrame:
        """
        Build a display-ready feature comparison table.

        Columns: Feature | Observed Value | Typical Value | Deviation |
                 Contribution | Direction | Interpretation

        ``observed_value`` and ``reference_value`` on each contribution are now
        both in the dataset's original units (see
        compute_statistical_deviations), so the percentage deviation is a real
        percentage.  ``original_values`` is used only as a fallback for
        numerical columns that were somehow missing.
        """
        rows = []
        for entry in top_features:
            feat = entry["feature"]
            obs_val = entry["observed_value"]
            ref_val = entry["reference_value"]
            z = entry.get("robust_z_score", 0.0)
            direction = entry["deviation_direction"]
            contribution = entry["combined_score"]
            is_numerical = entry["feature_type"] == "numerical"

            if is_numerical:
                display_obs = obs_val
                display_ref = ref_val

                if original_values and feat in original_values:
                    raw_val = original_values[feat]
                    if isinstance(
                        raw_val, (int, float, np.floating, np.integer)
                    ):
                        display_obs = round(float(raw_val), 4)

                display_obs = round(float(display_obs), 4)
                display_ref = round(float(display_ref), 4)

                if abs(display_ref) > 1e-6:
                    pct_dev = (display_obs - display_ref) / abs(display_ref) * 100
                    deviation_str = f"{'+' if pct_dev >= 0 else ''}{pct_dev:.1f}%"
                else:
                    deviation_str = f"z={z:+.2f}"
            else:
                display_obs = round(float(obs_val), 4)
                display_ref = "N/A"
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
        top_features: list[dict[str, Any]],
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

    def get_shap_status(self) -> dict[str, Any]:
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

"""
src/evaluation/drift.py

Dataset and Anomaly Score Drift Monitoring Engine for XAUD 2.0.

Provides:
- Population Stability Index (PSI) calculation for numerical features and anomaly scores.
- Two-sample Kolmogorov-Smirnov (KS) test for continuous feature distributions.
- Jensen-Shannon (JS) divergence for categorical distributions.
- Missingness drift tracking.
- Severity classification (NONE, LOW, MODERATE, HIGH) and investigation guidance.

Scientific Principle:
Unsupervised models implicitly assume exchangeability and distribution constancy.
When production data shifts, detection thresholds and score calibrations drift.
Drift metrics quantify distributional divergence between reference baseline
and incoming batches.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


class DriftMonitor:
    """
    Compares a baseline reference dataset against an incoming production batch.

    Parameters
    ----------
    psi_bins : int, default=10
        Number of quantile bins for Population Stability Index estimation.
    alpha_significance : float, default=0.01
        Significance level for Kolmogorov-Smirnov p-value testing.
    """

    def __init__(self, psi_bins: int = 10, alpha_significance: float = 0.01):
        self.psi_bins = psi_bins
        self.alpha_significance = alpha_significance

    def calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int | None = None,
    ) -> float:
        """
        Compute Population Stability Index (PSI) between reference and current.

        Formula:
            PSI = sum_{b=1}^B (P_b - Q_b) * ln(P_b / Q_b)
        """
        ref = np.asarray(reference, dtype=np.float64).ravel()
        cur = np.asarray(current, dtype=np.float64).ravel()

        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]

        if len(ref) < 2 or len(cur) < 2:
            return 0.0

        bins_count = n_bins or self.psi_bins

        # Generate quantile bin edges from reference distribution
        quantiles = np.linspace(0.0, 1.0, bins_count + 1)
        bin_edges = np.percentile(ref, quantiles * 100)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            # Degenerate reference distribution (constant)
            return 0.0 if np.allclose(ref.mean(), cur.mean()) else 1.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        cur_counts, _ = np.histogram(cur, bins=bin_edges)

        eps = 1e-4
        P = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
        Q = (cur_counts + eps) / (len(cur) + eps * len(cur_counts))

        psi_val = np.sum((P - Q) * np.log(P / Q))
        return float(max(0.0, psi_val))

    def evaluate_drift(
        self,
        df_reference: pd.DataFrame,
        df_current: pd.DataFrame,
        ref_scores: np.ndarray | None = None,
        cur_scores: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Run complete dataset and score drift evaluation.

        Parameters
        ----------
        df_reference : pd.DataFrame
            Baseline training or historical dataset.
        df_current : pd.DataFrame
            Incoming production or evaluation dataset.
        ref_scores : np.ndarray or None
            Baseline anomaly scores.
        cur_scores : np.ndarray or None
            Production anomaly scores.

        Returns
        -------
        dict with keys:
            overall_severity : {"NONE", "LOW", "MODERATE", "HIGH"}
            mean_feature_psi : float
            features_drifted_count : int
            feature_reports : list of dicts
            score_drift : dict or None
            missingness_drift : list of dicts
            actions : list of str
        """
        shared_cols = [c for c in df_reference.columns if c in df_current.columns]
        if not shared_cols:
            raise ValueError("Reference and current datasets share no common columns.")

        feature_reports: list[dict[str, Any]] = []
        psi_values: list[float] = []

        for col in shared_cols:
            ref_col = df_reference[col].dropna()
            cur_col = df_current[col].dropna()

            if len(ref_col) < 5 or len(cur_col) < 5:
                continue

            is_numeric = pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col)

            if is_numeric:
                psi = self.calculate_psi(ref_col.values, cur_col.values)
                psi_values.append(psi)

                ks_res = ks_2samp(ref_col.values, cur_col.values)
                ks_stat = float(ks_res.statistic)
                p_val = float(ks_res.pvalue)

                # Classification based on standard PSI thresholds
                if psi >= 0.25:
                    status = "HIGH"
                elif psi >= 0.10:
                    status = "MODERATE"
                else:
                    status = "STABLE"

                feature_reports.append({
                    "feature": col,
                    "type": "numerical",
                    "psi": round(psi, 4),
                    "ks_statistic": round(ks_stat, 4),
                    "p_value": round(p_val, 5),
                    "statistically_significant": bool(p_val < self.alpha_significance),
                    "drift_status": status,
                    "ref_mean": round(float(ref_col.mean()), 4),
                    "cur_mean": round(float(cur_col.mean()), 4),
                })

            else:
                # Categorical JS divergence
                all_cats = list(set(ref_col.unique()).union(set(cur_col.unique())))
                ref_freq = ref_col.value_counts(normalize=True).reindex(all_cats, fill_value=1e-5)
                cur_freq = cur_col.value_counts(normalize=True).reindex(all_cats, fill_value=1e-5)
                js_dist = float(jensenshannon(ref_freq, cur_freq))

                status = "HIGH" if js_dist >= 0.30 else ("MODERATE" if js_dist >= 0.15 else "STABLE")

                feature_reports.append({
                    "feature": col,
                    "type": "categorical",
                    "js_distance": round(js_dist, 4),
                    "drift_status": status,
                    "unique_cats_ref": int(ref_col.nunique()),
                    "unique_cats_cur": int(cur_col.nunique()),
                })

        # Missingness drift analysis
        missingness_drift: list[dict[str, Any]] = []
        for col in shared_cols:
            ref_null_rate = float(df_reference[col].isnull().mean())
            cur_null_rate = float(df_current[col].isnull().mean())
            delta = cur_null_rate - ref_null_rate
            if abs(delta) >= 0.05:  # >= 5% change in missingness
                missingness_drift.append({
                    "feature": col,
                    "ref_missing_rate": round(ref_null_rate, 4),
                    "cur_missing_rate": round(cur_null_rate, 4),
                    "rate_delta": round(delta, 4),
                })

        # Anomaly score drift analysis
        score_drift_report: dict[str, Any] | None = None
        score_psi = 0.0
        if ref_scores is not None and cur_scores is not None:
            score_psi = self.calculate_psi(ref_scores, cur_scores)
            ks_res = ks_2samp(ref_scores, cur_scores)
            score_drift_report = {
                "score_psi": round(score_psi, 4),
                "ks_statistic": round(float(ks_res.statistic), 4),
                "p_value": round(float(ks_res.pvalue), 5),
                "drift_status": "HIGH" if score_psi >= 0.25 else ("MODERATE" if score_psi >= 0.10 else "STABLE"),
                "ref_mean_score": round(float(np.mean(ref_scores)), 4),
                "cur_mean_score": round(float(np.mean(cur_scores)), 4),
            }

        # Overall severity synthesis
        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
        high_drift_features = [f["feature"] for f in feature_reports if f["drift_status"] == "HIGH"]
        mod_drift_features = [f["feature"] for f in feature_reports if f["drift_status"] == "MODERATE"]

        if mean_psi >= 0.25 or len(high_drift_features) >= 3 or score_psi >= 0.25:
            severity = "HIGH"
        elif mean_psi >= 0.10 or len(high_drift_features) >= 1 or len(mod_drift_features) >= 2 or score_psi >= 0.10:
            severity = "MODERATE"
        elif len(mod_drift_features) >= 1:
            severity = "LOW"
        else:
            severity = "NONE"

        # Actionable recommendations
        actions: list[str] = []
        if severity == "HIGH":
            actions.append(
                f"Significant dataset drift detected across {high_drift_features}. Retraining detectors on the updated distribution is strongly recommended."
            )
        elif severity == "MODERATE":
            actions.append(
                f"Moderate feature drift observed in {high_drift_features + mod_drift_features}. Monitor anomaly alert rates for false positive inflation."
            )
        else:
            actions.append("Distributions are stable between reference and current batches. Existing model thresholds remain valid.")

        if missingness_drift:
            actions.append(
                f"Missingness rate shifted substantially in {[m['feature'] for m in missingness_drift]}. Verify upstream ETL ingestion pipelines."
            )

        return {
            "overall_severity": severity,
            "mean_feature_psi": round(mean_psi, 4),
            "n_features_evaluated": len(feature_reports),
            "features_high_drift": high_drift_features,
            "features_moderate_drift": mod_drift_features,
            "feature_reports": feature_reports,
            "score_drift": score_drift_report,
            "missingness_drift": missingness_drift,
            "actions": actions,
        }

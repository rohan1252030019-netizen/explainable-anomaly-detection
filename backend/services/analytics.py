"""
backend/services/analytics.py

Visual analytics, model benchmarking, drift monitoring and the synthetic injection lab.

Extracted verbatim from the former monolithic ``backend/state.py``.  The
methods still read and write the shared :class:`backend.state.AppState`
instance through ``self``; the split is organisational, and it lets a
reviewer read one concern at a time.
"""

from typing import Any

import numpy as np
import pandas as pd

from backend.services.base import ServiceState, serialized
from src.evaluation.drift import DriftMonitor
from src.evaluation.model_comparison import compare_models, compute_model_agreement
from src.evaluation.synthetic_lab import SyntheticInjectionLab
from src.visualization.dimensionality_reduction import reduce_pca


class AnalyticsServiceMixin(ServiceState):
    """Visual analytics, model benchmarking, drift monitoring and the synthetic injection lab."""

    @serialized
    def compute_pca_analytics(self) -> dict[str, Any]:
        """Compute 2D PCA projection for cluster visualization."""
        if self.X is None:
            raise ValueError("Feature matrix is not ready.")

        if self.pca_coordinates is None:
            coords, pca_obj = reduce_pca(self.X, n_components=2)
            self.pca_coordinates = coords
            self.pca_variance_ratio = [float(v) for v in pca_obj.explained_variance_ratio_]

        n_samples = self.pca_coordinates.shape[0]
        max_pts = 1000
        step = max(1, n_samples // max_pts)
        idx_sample = np.arange(0, n_samples, step)

        points = []
        for i in idx_sample:
            is_anom = bool(self.predictions[i] == -1) if self.predictions is not None else False
            score = float(self.norm_scores[i]) if self.norm_scores is not None else 0.0
            sev = str(self.severity[i]) if self.severity is not None else "NORMAL"
            points.append({
                "id": int(i),
                "pc1": round(float(self.pca_coordinates[i, 0]), 4),
                "pc2": round(float(self.pca_coordinates[i, 1]), 4),
                "is_anomaly": is_anom,
                "score": round(score, 4),
                "severity": sev,
            })

        return {
            "variance_explained": self.pca_variance_ratio,
            "points": points,
        }

    def compute_correlations(self, max_features: int = 15) -> dict[str, Any]:
        """Compute correlation matrix for numerical features."""
        if self.df_raw is None:
            return {"features": [], "matrix": []}

        num_cols = [c for c in self.df_raw.columns if pd.api.types.is_numeric_dtype(self.df_raw[c])
                    and c not in ("label", "prediction", "rank", "is_anomaly", "normalized_score", "anomaly_score")][:max_features]

        if len(num_cols) < 2:
            return {"features": num_cols, "matrix": []}

        corr = self.df_raw[num_cols].corr().round(3)
        corr = corr.replace({np.nan: 0.0})

        return {
            "features": num_cols,
            "matrix": corr.values.tolist(),
        }

    @serialized
    def benchmark_models(self, contamination: float = 0.05) -> dict[str, Any]:
        """Run all detectors side-by-side and compute consensus metrics."""
        if self.X is None:
            raise ValueError("Feature matrix is not ready.")

        params_override = {
            "Isolation Forest": {"contamination": contamination},
            "Local Outlier Factor": {"contamination": contamination},
            "One-Class SVM": {"nu": contamination},
            "Elliptic Envelope": {"contamination": contamination},
        }
        all_results = self.model_manager.fit_all(self.X, self.feature_names, params_override=params_override)
        self.all_model_results = all_results

        # compare_models() takes the fit RESULTS dict, not the feature matrix.
        # This previously called a non-existent ModelManager.compare(), so the
        # benchmark endpoint failed with a 400 on every call.
        comp_df = compare_models(all_results, self.y_true)
        self.comparison_df = comp_df

        # Compute consensus agreement
        agreement = compute_model_agreement(all_results)

        # Format pairwise jaccard for JSON
        pj_list = []
        for (m1, m2), val in agreement.get("pairwise_jaccard", {}).items():
            pj_list.append({
                "model_a": m1,
                "model_b": m2,
                "jaccard": round(float(val), 3),
            })

        # Format comparison table records
        records = comp_df.replace({np.nan: None}).to_dict(orient="records")

        # Per-model anomaly counts
        model_counts = []
        for name in agreement.get("model_names", []):
            df_ag = agreement.get("agreement_df")
            cnt = int(df_ag[name].sum()) if df_ag is not None and name in df_ag else 0
            model_counts.append({"model": name, "anomaly_count": cnt})

        return {
            "comparison_table": records,
            "all_model_consensus": agreement.get("agreed_anomalies", 0),
            "disagreed_count": agreement.get("disagreed", 0),
            "agreed_normal": agreement.get("agreed_normal", 0),
            "model_anomaly_counts": model_counts,
            "pairwise_jaccard": pj_list,
        }

    @serialized
    def run_drift_analysis(
        self,
        df_incoming: pd.DataFrame | None = None,
        psi_bins: int = 10,
    ) -> dict[str, Any]:
        """Compute drift between reference baseline and incoming data."""
        if self.df_raw is None:
            raise ValueError("No baseline dataset loaded.")

        monitor = DriftMonitor(psi_bins=psi_bins)

        if df_incoming is not None:
            df_cur = df_incoming
            cur_sc = None
        else:
            n = len(self.df_raw)
            if n < 30:
                raise ValueError("Dataset too small for split drift analysis (need >= 30 rows).")
            split_idx = int(n * 0.6)
            df_ref = self.df_raw.iloc[:split_idx]
            df_cur = self.df_raw.iloc[split_idx:]
            ref_sc = self.norm_scores[:split_idx] if self.norm_scores is not None else None
            cur_sc = self.norm_scores[split_idx:] if self.norm_scores is not None else None
            res = monitor.evaluate_drift(df_ref, df_cur, ref_scores=ref_sc, cur_scores=cur_sc)
            self.drift_report = res
            return res

        res = monitor.evaluate_drift(self.df_raw, df_cur, ref_scores=self.norm_scores, cur_scores=cur_sc)
        self.drift_report = res
        return res

    @serialized
    def run_synthetic_benchmark(
        self,
        anomaly_type: str = "multivariate",
        contamination: float = 0.05,
        models: list[str] | None = None,
        difficulty: str = "medium",
        severity: float = 3.5,
    ) -> dict[str, Any]:
        """Run controlled anomaly injection benchmark on preprocessed feature matrix."""
        if self.X is None:
            raise ValueError("Feature matrix X is not ready. Preprocess dataset first.")

        lab = SyntheticInjectionLab(random_state=42)
        report = lab.benchmark_detectors(
            self.X,
            anomaly_type=anomaly_type,
            contamination=contamination,
            models_to_test=models,
            feature_names=self.feature_names,
            difficulty=difficulty,
            severity=severity,
        )
        self.benchmark_report = report
        return report

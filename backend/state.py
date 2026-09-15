"""
backend/state.py

Thread-safe in-memory state manager for the Explainable Anomaly Detection Framework.
Preserves the exact data flow and orchestrates calls to `src/` modules.
"""

import os
import sys
import threading
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import DataLoader, generate_sample_dataset
from src.data.validator import DataValidator
from src.data.preprocessing import PreprocessingPipeline
from src.models.model_manager import ModelManager, DEFAULT_PARAMS, DETECTOR_REGISTRY
from src.detection.scoring import normalize_scores, ensemble_scores
from src.detection.thresholding import apply_threshold, assign_severity, get_threshold_value
from src.detection.ranking import rank_anomalies, build_results_dataframe
from src.explainability.explanation_generator import ExplanationGenerator
from src.evaluation.model_comparison import compare_models, compute_model_agreement
from src.visualization.dimensionality_reduction import reduce_pca


class AppState:
    """Singleton state manager caching dataset, model, and analysis results."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppState, cls).__new__(cls)
                cls._instance._init_defaults()
            return cls._instance

    def _init_defaults(self):
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_results: Optional[pd.DataFrame] = None
        self.dataset_name: str = ""
        self.loader: Optional[DataLoader] = None
        self.validator_report: Optional[Dict[str, Any]] = None
        self.pipeline: Optional[PreprocessingPipeline] = None
        self.X: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.y_true: Optional[np.ndarray] = None
        self.model_manager: ModelManager = ModelManager(models_dir=os.path.join(PROJECT_ROOT, "models"))
        self.active_model: str = "Isolation Forest"
        self.norm_scores: Optional[np.ndarray] = None
        self.predictions: Optional[np.ndarray] = None
        self.severity: Optional[np.ndarray] = None
        self.threshold_val: float = 0.5
        self.threshold_method: str = "contamination"
        self.contamination: float = 0.05
        self.explanations: List[Dict[str, Any]] = []
        self.global_importance: Optional[pd.DataFrame] = None
        self.all_model_results: Dict[str, Any] = {}
        self.comparison_df: Optional[pd.DataFrame] = None
        self.experiment_history: List[Dict[str, Any]] = []
        self.label_col: Optional[str] = None
        self.id_cols: List[str] = []
        self.scaler_type: str = "standard"
        self.detection_run: bool = False
        self.explanation_run: bool = False
        self.pca_coordinates: Optional[np.ndarray] = None
        self.pca_variance_ratio: Optional[List[float]] = None

    def reset_detection(self):
        """Clear detection and downstream state (keep dataset + preprocessing)."""
        self.df_results = None
        self.norm_scores = None
        self.predictions = None
        self.severity = None
        self.explanations = []
        self.global_importance = None
        self.all_model_results = {}
        self.comparison_df = None
        self.detection_run = False
        self.explanation_run = False
        self.pca_coordinates = None
        self.pca_variance_ratio = None

    def reset_all(self):
        """Clear entire application state."""
        self._init_defaults()

    def load_sample_dataset(self) -> Dict[str, Any]:
        """Load or create the benchmark sample dataset."""
        sample_path = os.path.join(PROJECT_ROOT, "data", "sample", "sample_anomaly_data.csv")
        if not os.path.exists(sample_path):
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            df = generate_sample_dataset(n_normal=800, n_anomalies=80, random_state=42)
            df.to_csv(sample_path, index=False)
        
        loader = DataLoader()
        df = loader.load_csv(sample_path)
        self.df_raw = df
        self.dataset_name = "sample_anomaly_data.csv"
        self.loader = loader
        
        validator = DataValidator()
        self.validator_report = validator.validate(df)
        
        # Reset downstream
        self.reset_detection()
        self.X = None
        self.feature_names = []
        self.pipeline = None
        
        # Automatically fit default preprocessing
        self.preprocess_dataset()
        
        return self.get_dataset_summary()

    def load_user_csv(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Load uploaded CSV file bytes."""
        loader = DataLoader()
        df = loader.load_bytes(file_bytes, filename)
        self.df_raw = df
        self.dataset_name = filename
        self.loader = loader
        
        validator = DataValidator()
        self.validator_report = validator.validate(df)
        
        self.reset_detection()
        self.X = None
        self.feature_names = []
        self.pipeline = None
        
        # Automatically run preprocessing with standard defaults
        self.preprocess_dataset()
        
        return self.get_dataset_summary()

    def preprocess_dataset(
        self,
        scaler_type: str = "standard",
        label_col: Optional[str] = None,
        id_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fit preprocessing pipeline and prepare feature matrix X."""
        if self.df_raw is None:
            raise ValueError("No dataset loaded.")

        self.scaler_type = scaler_type
        self.id_cols = id_cols or []

        # Auto-detect label column if not specified
        if label_col is None:
            summary = self.loader.get_summary() if self.loader else {}
            potential_label = summary.get("potential_label_col")
            if potential_label and potential_label in self.df_raw.columns:
                label_col = potential_label
            elif "label" in self.df_raw.columns:
                label_col = "label"

        self.label_col = label_col
        if label_col and label_col in self.df_raw.columns:
            self.y_true = (self.df_raw[label_col].values == 1).astype(int)
        else:
            self.y_true = None

        pipeline = PreprocessingPipeline(
            label_col=self.label_col,
            scaler_type=self.scaler_type,
            drop_duplicates=False,
            id_cols=self.id_cols,
        )
        X, feature_names = pipeline.fit_transform(self.df_raw)
        self.pipeline = pipeline
        self.X = X
        self.feature_names = feature_names

        # Clear any prior detection
        self.reset_detection()

        return {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": feature_names,
            "scaler_type": scaler_type,
            "label_col": self.label_col,
            "has_labels": self.y_true is not None,
            "pipeline_log": pipeline.get_log(),
        }

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Return dataset summary and column profiling."""
        if self.df_raw is None:
            return {"loaded": False}

        summary = self.loader.get_summary() if self.loader else {}
        col_info = self.loader.get_column_info().to_dict(orient="records") if self.loader else []
        
        # Preview top 10 rows safely
        preview_df = self.df_raw.head(10).copy()
        # Clean NaNs and Infinities for JSON serialization
        preview_records = preview_df.replace({np.nan: None}).to_dict(orient="records")

        return {
            "loaded": True,
            "dataset_name": self.dataset_name,
            "n_rows": int(self.df_raw.shape[0]),
            "n_cols": int(self.df_raw.shape[1]),
            "numerical_cols": summary.get("numerical_cols", []),
            "categorical_cols": summary.get("categorical_cols", []),
            "total_missing": int(summary.get("total_missing", 0)),
            "duplicate_rows": int(summary.get("duplicate_rows", 0)),
            "memory_mb": float(summary.get("memory_mb", 0.0)),
            "columns": col_info,
            "preview": preview_records,
            "feature_matrix_ready": self.X is not None,
            "n_preprocessed_features": len(self.feature_names),
            "label_col": self.label_col,
            "warnings": self.validator_report.get("warnings", []) if self.validator_report else [],
        }

    def run_detection(
        self,
        model_name: str = "Isolation Forest",
        params: Optional[Dict[str, Any]] = None,
        threshold_method: str = "contamination",
        contamination: float = 0.05,
        quantile_val: float = 0.95,
        fixed_val: float = 0.5,
        use_ensemble: bool = False,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Execute unsupervised anomaly detection pipeline."""
        if self.X is None:
            raise ValueError("Feature matrix X is not ready. Preprocess dataset first.")

        self.active_model = model_name
        self.threshold_method = threshold_method
        self.contamination = contamination

        merged_params = {**DEFAULT_PARAMS.get(model_name, {})}
        if params:
            merged_params.update(params)

        fit_result = self.model_manager.fit(model_name, self.X, self.feature_names, merged_params)
        if not fit_result.get("success"):
            raise RuntimeError(f"Training failed: {fit_result.get('error')}")

        raw_scores = fit_result["scores"]
        norm_scores = normalize_scores(raw_scores)

        if use_ensemble:
            weights = ensemble_weights or {
                "Isolation Forest": 0.4,
                "Local Outlier Factor": 0.3,
                "One-Class SVM": 0.3,
            }
            scores_dict = {model_name: norm_scores}
            for m in ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"]:
                if m != model_name:
                    r = self.model_manager.fit(m, self.X, self.feature_names)
                    if r.get("success"):
                        scores_dict[m] = normalize_scores(r["scores"])
            norm_scores = ensemble_scores(scores_dict, weights)

        threshold_val = get_threshold_value(
            norm_scores,
            method=threshold_method,
            contamination=contamination,
            quantile=quantile_val,
            fixed_threshold=fixed_val,
        )
        predictions = apply_threshold(
            norm_scores,
            method=threshold_method,
            contamination=contamination,
            quantile=quantile_val,
            fixed_threshold=fixed_val,
        )
        severity = assign_severity(norm_scores)

        df_results = build_results_dataframe(
            self.df_raw, raw_scores, norm_scores, predictions, severity, model_name
        )

        self.norm_scores = norm_scores
        self.predictions = predictions
        self.severity = severity
        self.df_results = df_results
        self.threshold_val = float(threshold_val)
        self.detection_run = True
        self.explanation_run = False
        self.explanations = []
        self.global_importance = None

        # Record experiment
        import datetime
        n_anom = int((predictions == -1).sum())
        self.experiment_history.insert(0, {
            "id": len(self.experiment_history) + 1,
            "model": model_name,
            "params": merged_params,
            "threshold_method": threshold_method,
            "threshold_val": float(threshold_val),
            "contamination": contamination,
            "n_anomalies": n_anom,
            "anomaly_rate": float(n_anom / len(predictions)),
            "fit_time_sec": float(fit_result.get("fit_time_sec", 0)),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        return self.get_detection_overview()

    def get_detection_overview(self) -> Dict[str, Any]:
        """Return high-level KPIs and distribution bins."""
        if not self.detection_run or self.norm_scores is None:
            return {"ready": False}

        n_total = len(self.predictions)
        n_anom = int((self.predictions == -1).sum())
        anom_mask = self.predictions == -1
        anom_sev = self.severity[anom_mask] if n_anom > 0 else np.array([])
        
        n_crit = int((anom_sev == "CRITICAL").sum())
        n_high = int((anom_sev == "HIGH").sum())
        n_med = int((anom_sev == "MEDIUM").sum())
        n_low = int((anom_sev == "LOW").sum())

        anom_scores = self.norm_scores[anom_mask] if n_anom > 0 else np.array([0.0])
        normal_scores = self.norm_scores[~anom_mask] if (n_total - n_anom) > 0 else np.array([0.0])

        # Compute histogram bins for score distribution
        bins = np.linspace(0.0, 1.0, 31)
        normal_counts, _ = np.histogram(normal_scores, bins=bins)
        anom_counts, _ = np.histogram(anom_scores, bins=bins)
        
        hist_data = []
        for i in range(len(bins) - 1):
            hist_data.append({
                "bin_start": round(float(bins[i]), 3),
                "bin_end": round(float(bins[i + 1]), 3),
                "normal_count": int(normal_counts[i]),
                "anomaly_count": int(anom_counts[i]),
            })

        # Score timeline downsampled if large
        max_scatter_pts = 1000
        step = max(1, n_total // max_scatter_pts)
        idx_sample = np.arange(0, n_total, step)
        timeline = []
        for idx in idx_sample:
            timeline.append({
                "index": int(idx),
                "score": round(float(self.norm_scores[idx]), 4),
                "is_anomaly": bool(self.predictions[idx] == -1),
                "severity": str(self.severity[idx]),
            })

        return {
            "ready": True,
            "model_name": self.active_model,
            "n_total": n_total,
            "n_anomalies": n_anom,
            "anomaly_rate": round(float(n_anom / n_total), 4) if n_total > 0 else 0,
            "avg_score": round(float(np.mean(self.norm_scores)), 4),
            "median_anomaly_score": round(float(np.median(anom_scores)), 4),
            "max_score": round(float(np.max(self.norm_scores)), 4),
            "threshold_val": round(self.threshold_val, 4),
            "threshold_method": self.threshold_method,
            "severity_counts": {
                "CRITICAL": n_crit,
                "HIGH": n_high,
                "MEDIUM": n_med,
                "LOW": n_low,
            },
            "score_histogram": hist_data,
            "score_timeline": timeline,
        }

    def get_anomalies(
        self,
        search: Optional[str] = None,
        severity_filter: Optional[List[str]] = None,
        min_score: float = 0.0,
        max_score: float = 1.0,
        anomalies_only: bool = True,
        sort_by: str = "rank",
        ascending: bool = True,
        page: int = 1,
        page_size: int = 25,
    ) -> Dict[str, Any]:
        """Query and paginate anomaly records."""
        if not self.detection_run or self.df_results is None:
            return {"total": 0, "items": []}

        df = self.df_results.copy()
        if anomalies_only:
            df = df[df["is_anomaly"] == True]

        if severity_filter:
            df = df[df["severity"].isin(severity_filter)]

        df = df[(df["normalized_score"] >= min_score) & (df["normalized_score"] <= max_score)]

        if search:
            search_str = search.lower()
            mask = False
            for col in df.columns:
                mask = mask | df[col].astype(str).str.lower().str.contains(search_str, na=False)
            df = df[mask]

        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)

        total_count = len(df)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_df = df.iloc[start_idx:end_idx]

        # Convert to records safely
        items = paged_df.replace({np.nan: None}).to_dict(orient="records")

        return {
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": int(np.ceil(total_count / page_size)) if total_count > 0 else 1,
            "items": items,
        }

    def investigate_anomaly(self, row_id: int) -> Dict[str, Any]:
        """Provide single-anomaly deep-dive and transparent explanation."""
        if not self.detection_run or self.df_results is None or self.X is None:
            raise ValueError("Detection must be completed first.")

        if row_id < 0 or row_id >= len(self.df_raw):
            raise IndexError(f"Observation ID {row_id} out of bounds.")

        norm_score = float(self.norm_scores[row_id])
        sev = str(self.severity[row_id])
        is_anom = bool(self.predictions[row_id] == -1)
        rank_val = self.df_results.iloc[row_id].get("rank")
        rank = int(rank_val) if pd.notna(rank_val) else None

        obs_X = self.X[row_id]
        orig_vals = self.df_raw.iloc[row_id].replace({np.nan: None}).to_dict()
        ref_stats = self.pipeline.get_reference_stats() if self.pipeline else {}

        # Look in cached explanations or compute on the fly
        exp = next((e for e in self.explanations if e.get("row_id") == row_id), None)
        if exp is None:
            detector = self.model_manager._detectors.get(self.active_model)
            sklearn_model = getattr(detector, "_model", None) if detector else None

            def norm_fn(obs):
                if detector:
                    return normalize_scores(detector.score_samples(obs))
                return np.zeros(len(obs))

            explainer = ExplanationGenerator(
                model_name=self.active_model,
                sklearn_model=sklearn_model,
                X_train=self.X,
                feature_names=self.feature_names,
                reference_stats=ref_stats,
                norm_score_fn=norm_fn,
                use_shap=False,
                top_k=8,
            )
            exp = explainer.explain_observation(
                observation=obs_X,
                norm_score=norm_score,
                severity=sev,
                row_id=row_id,
                original_values=orig_vals,
            )

        # Build clean feature comparison items (observed vs typical median/range)
        comparison_list = []
        ft = exp.get("feature_table")
        if ft is not None and not ft.empty:
            for _, frow in ft.iterrows():
                feat = frow["Feature"]
                obs_val = orig_vals.get(feat, frow.get("Observed"))
                ref_median = ref_stats.get("median", {}).get(feat, frow.get("Typical (Median)"))
                ref_iqr = ref_stats.get("iqr", {}).get(feat, 0.0)
                ref_min = ref_stats.get("min", {}).get(feat)
                ref_max = ref_stats.get("max", {}).get(feat)
                
                comparison_list.append({
                    "feature": feat,
                    "observed": obs_val,
                    "typical_median": ref_median,
                    "typical_iqr": ref_iqr,
                    "typical_min": ref_min,
                    "typical_max": ref_max,
                    "deviation": str(frow.get("Deviation", "")),
                    "direction": str(frow.get("Direction", "")),
                    "contribution": str(frow.get("Contribution", "")),
                    "interpretation": str(frow.get("Interpretation", "")),
                })

        return {
            "row_id": row_id,
            "rank": rank,
            "model_name": self.active_model,
            "normalized_score": norm_score,
            "is_anomaly": is_anom,
            "severity": sev,
            "threshold": self.threshold_val,
            "explanation_text": exp.get("explanation_text", ""),
            "confidence_note": exp.get("confidence_note", ""),
            "top_features": exp.get("top_features", []),
            "feature_comparison": comparison_list,
            "raw_record": orig_vals,
        }

    def generate_all_explanations(self, use_shap: bool = False, top_k: int = 8) -> Dict[str, Any]:
        """Compute global explainability across all flagged anomalies."""
        if not self.detection_run or self.X is None:
            raise ValueError("Detection must be run first.")

        detector = self.model_manager._detectors.get(self.active_model)
        sklearn_model = getattr(detector, "_model", None) if detector else None

        def norm_fn(obs):
            if detector:
                return normalize_scores(detector.score_samples(obs))
            return np.zeros(len(obs))

        explainer = ExplanationGenerator(
            model_name=self.active_model,
            sklearn_model=sklearn_model,
            X_train=self.X,
            feature_names=self.feature_names,
            reference_stats=self.pipeline.get_reference_stats() if self.pipeline else {},
            norm_score_fn=norm_fn,
            use_shap=use_shap,
            top_k=top_k,
        )

        explanations = explainer.explain_batch(
            X=self.X,
            norm_scores=self.norm_scores,
            predictions=self.predictions,
            severity=self.severity,
            original_df=self.df_raw,
            max_explanations=200,
        )
        global_imp = explainer.compute_global_importance(explanations)

        self.explanations = explanations
        self.global_importance = global_imp
        self.explanation_run = True

        imp_records = global_imp.replace({np.nan: None}).to_dict(orient="records") if global_imp is not None else []

        return {
            "n_explained": len(explanations),
            "top_driver": imp_records[0]["feature"] if imp_records else None,
            "global_importance": imp_records,
        }

    def compute_pca_analytics(self) -> Dict[str, Any]:
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

    def compute_correlations(self, max_features: int = 15) -> Dict[str, Any]:
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

    def benchmark_models(self, contamination: float = 0.05) -> Dict[str, Any]:
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

        # Evaluate comparison table
        comp_df = self.model_manager.compare(self.X, self.y_true)
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

    def generate_report(self) -> Dict[str, Any]:
        """Generate presentation-ready academic and technical audit report."""
        import datetime
        summary = self.get_dataset_summary()
        overview = self.get_detection_overview()
        
        anomalies_data = self.get_anomalies(anomalies_only=True, page=1, page_size=10)
        
        global_drivers = []
        if self.global_importance is not None and not self.global_importance.empty:
            global_drivers = self.global_importance.head(5).to_dict(orient="records")

        return {
            "title": "Explainable Anomaly Detection Audit Report",
            "framework": "XAUD Unsupervised Intelligence Framework",
            "generated_at": datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S"),
            "dataset": {
                "name": self.dataset_name,
                "rows": summary.get("n_rows", 0),
                "columns": summary.get("n_cols", 0),
                "features_preprocessed": len(self.feature_names),
                "scaler": self.scaler_type,
            },
            "detection": {
                "model": self.active_model,
                "threshold_method": self.threshold_method,
                "threshold_value": self.threshold_val,
                "total_observations": overview.get("n_total", 0),
                "anomalies_detected": overview.get("n_anomalies", 0),
                "anomaly_rate": overview.get("anomaly_rate", 0),
                "severity_distribution": overview.get("severity_counts", {}),
            },
            "top_drivers": global_drivers,
            "sample_anomalies": anomalies_data.get("items", []),
            "methodology_note": (
                "Unsupervised anomaly detection identifies patterns that deviate statistically from "
                "the majority distribution without requiring target labels. Feature attributions indicate "
                "relative deviation magnitudes and model sensitivities, serving as decision support."
            ),
        }


# Singleton accessor
state = AppState()

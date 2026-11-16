"""
backend/services/detection.py

Detection orchestration: single-model scoring, consensus ensembles, and the overview payload.

Extracted verbatim from the former monolithic ``backend/state.py``.  The
methods still read and write the shared :class:`backend.state.AppState`
instance through ``self``; the split is organisational, and it lets a
reviewer read one concern at a time.
"""

from typing import Any

import numpy as np

from backend.services.base import ServiceState, serialized
from src.detection.ensemble import IntelligentEnsembleEngine
from src.detection.ranking import (
    build_results_dataframe,
)
from src.detection.scoring import ScoreNormalizer, ensemble_scores
from src.detection.thresholding import (
    apply_threshold,
    assign_severity,
    get_threshold_value,
    severity_legend,
)
from src.models.model_manager import (
    DEFAULT_PARAMS,
    row_limit_for,
)


class DetectionServiceMixin(ServiceState):
    """Detection orchestration: single-model scoring, consensus ensembles, and the overview payload."""

    @serialized
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
        self.normalizer = None
        self.holdout_report = None
        self.holdout_table = None
        self.severity_bands = None
        self.ensemble_report = None
        self.drift_report = None
        self.benchmark_report = None
        self.ensemble_models = []

    @property
    def display_model_name(self) -> str:
        """
        Human-readable name of the current result set.

        ``active_model`` must always stay a *registry* name: holdout,
        explainability, stability and persistence all resolve it through
        ``DETECTOR_REGISTRY``.  Writing a label like
        ``"Ensemble (2 models)"`` into it made every later call fail with
        ``Unknown model 'Ensemble (2 models)'`` (HTTP 500).  The display label
        therefore lives here, derived from the cached ensemble report.
        """
        if self.ensemble_models:
            return f"Intelligent Ensemble ({len(self.ensemble_models)} models)"
        return self.active_model

    @staticmethod
    def _ensemble_params(model_name: str, contamination: float) -> dict[str, Any]:
        """
        Translate one run-level contamination value into a detector's own
        hyperparameters.  Fitting the auxiliary models with their defaults
        silently mixed a 0.10 contamination run with 0.05 auxiliary models, so
        the ensemble compared detectors under different flag budgets.
        """
        params = dict(DEFAULT_PARAMS.get(model_name, {}))
        if model_name == "One-Class SVM":
            params["nu"] = float(contamination)
        elif "contamination" in params:
            params["contamination"] = float(contamination)
        return params

    @staticmethod
    def _validated_ensemble_weights(
        requested: dict[str, float] | None,
        available: dict[str, Any],
    ) -> dict[str, float]:
        """Return one non-negative weight per fitted model, validated."""
        names = list(available.keys())
        if not requested:
            defaults = {
                "Isolation Forest": 0.4,
                "Local Outlier Factor": 0.3,
                "One-Class SVM": 0.3,
            }
            return {n: float(defaults.get(n, 1.0)) for n in names}

        weights: dict[str, float] = {}
        for name in names:
            try:
                value = float(requested.get(name, 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Ensemble weight for '{name}' must be a number, got "
                    f"{requested.get(name)!r}."
                ) from exc
            if value < 0:
                raise ValueError(
                    f"Ensemble weight for '{name}' must be non-negative, got {value}."
                )
            weights[name] = value

        if sum(weights.values()) <= 0:
            raise ValueError(
                "Ensemble weights sum to zero; at least one fitted model must "
                "carry a positive weight."
            )
        return weights

    @serialized
    def run_detection(
        self,
        model_name: str = "Isolation Forest",
        params: dict[str, Any] | None = None,
        threshold_method: str = "contamination",
        contamination: float = 0.05,
        quantile_val: float = 0.95,
        fixed_val: float = 0.5,
        use_ensemble: bool = False,
        ensemble_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute unsupervised anomaly detection pipeline."""
        if self.X is None:
            raise ValueError("Feature matrix X is not ready. Preprocess dataset first.")

        # A shared memory bound is not a complexity bound.  LOF and k-NN build
        # pairwise neighbourhoods and cannot process a dataset that Isolation
        # Forest handles comfortably, so refuse early instead of appearing to
        # hang.
        limit = row_limit_for(model_name)
        if limit is not None and len(self.X) > limit:
            raise ValueError(
                f"'{model_name}' is limited to {limit:,} rows "
                f"(dataset has {len(self.X):,}). Use Isolation Forest, PCA "
                f"Reconstruction or HBOS for datasets this large."
            )

        self.active_model = model_name
        self.ensemble_models = []
        self.threshold_method = threshold_method
        self.contamination = contamination

        merged_params = {**DEFAULT_PARAMS.get(model_name, {})}
        if params:
            merged_params.update(params)

        fit_result = self.model_manager.fit(model_name, self.X, self.feature_names, merged_params)
        if not fit_result.get("success"):
            raise RuntimeError(f"Training failed: {fit_result.get('error')}")

        raw_scores = fit_result["scores"]

        # Fit the normaliser once and reuse it, so every downstream consumer
        # (thresholding, severity, SHAP, counterfactuals) shares one mapping.
        # Fitting a fresh normaliser inside each helper would let the units
        # drift between them.
        self.normalizer = ScoreNormalizer().fit(raw_scores)
        norm_scores = self.normalizer.transform(raw_scores)

        if use_ensemble:
            scores_dict = {model_name: norm_scores}
            for m in ("Isolation Forest", "Local Outlier Factor", "One-Class SVM"):
                if m == model_name:
                    continue
                # Each detector's raw scores live on their own scale.  Mapping
                # another model's scores through the active model's anchors puts
                # them on the wrong axis: before this fix, LOF scores were 100%
                # clipped to 1.0, One-Class SVM 100% to 0.0 and Elliptic
                # Envelope 100% to 1.0 — the "ensemble" was a weighted vote of
                # constants.  Fit one normaliser per detector on that detector's
                # own reference scores.
                r = self.model_manager.fit(
                    m,
                    self.X,
                    self.feature_names,
                    self._ensemble_params(m, contamination),
                )
                if not r.get("success"):
                    continue
                model_normalizer = ScoreNormalizer().fit(r["scores"])
                scores_dict[m] = model_normalizer.transform(r["scores"])

            weights = self._validated_ensemble_weights(
                ensemble_weights, scores_dict
            )
            norm_scores = ensemble_scores(scores_dict, weights)
            # Label the run as a consensus result while leaving active_model a
            # real registry entry.
            self.ensemble_models = list(scores_dict)

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
        # Tier severity by rank within the flagged tail, so the four bands stay
        # meaningful for any score distribution.
        severity = assign_severity(
            norm_scores, anomaly_mask=(predictions == -1)
        )

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

        # Cache the severity legend so the UI states what each band means.
        self.severity_bands = {
            "mode": "rank_within_flagged",
            "levels": severity_legend(
                {
                    label: int(((predictions == -1) & (severity == label)).sum())
                    for label in set(severity.tolist())
                }
            ),
            "note": (
                "Severity is the standing of a flagged row within this run's "
                "flagged set, not an absolute score. Bands are comparable "
                "across runs, not across datasets without recalibration."
            ),
        }

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

    @serialized
    def run_intelligent_ensemble(
        self,
        models: list[str],
        aggregation_method: str = "rank_borda",
        weights: dict[str, float] | None = None,
        min_consensus_votes: int | None = None,
    ) -> dict[str, Any]:
        """Train multiple models and compute consensus ensemble."""
        if self.X is None:
            raise ValueError("Feature matrix X is not ready. Preprocess dataset first.")
        if not models:
            raise ValueError("At least one model must be specified.")

        # Apply the same per-detector ceiling the single-model path enforces:
        # an ensemble must not smuggle a 200k-row matrix into LOF just because
        # the request named it alongside cheaper models.
        for name in models:
            limit = row_limit_for(name)
            if limit is not None and len(self.X) > limit:
                raise ValueError(
                    f"'{name}' is limited to {limit:,} rows "
                    f"(dataset has {len(self.X):,}). Remove it from the "
                    f"ensemble or use Isolation Forest, PCA Reconstruction or "
                    f"HBOS for datasets this large."
                )

        detector_results = {}
        for m in models:
            fit_res = self.model_manager.fit(m, self.X, self.feature_names)
            if fit_res.get("success"):
                detector_results[m] = {
                    "scores": fit_res["scores"],
                    "predictions": fit_res["predictions"],
                }

        if not detector_results:
            raise RuntimeError("All models in ensemble failed to train.")

        engine = IntelligentEnsembleEngine(
            aggregation_method=aggregation_method,
            min_consensus_votes=min_consensus_votes,
        )
        ensemble_res = engine.run_ensemble(detector_results, weights=weights)

        norm_scores = np.asarray(ensemble_res["ensemble_scores"], dtype=np.float64)
        predictions = np.asarray(ensemble_res["consensus_predictions"], dtype=int)
        severity = assign_severity(norm_scores, anomaly_mask=(predictions == -1))

        df_results = build_results_dataframe(
            self.df_raw,
            raw_scores=norm_scores,
            norm_scores=norm_scores,
            predictions=predictions,
            severity=severity,
            model_name="Intelligent Ensemble",
        )

        self.norm_scores = norm_scores
        self.predictions = predictions
        self.severity = severity
        self.df_results = df_results
        # Keep ``active_model`` a real registry entry (the first fitted member)
        # and record the full membership separately.  Overwriting it with
        # "Ensemble (n models)" broke every follow-up endpoint that resolves the
        # active model through the registry (holdout -> HTTP 400/500,
        # explainability -> unknown model).
        self.active_model = next(iter(detector_results))
        self.ensemble_models = list(detector_results)
        self.detection_run = True
        self.ensemble_report = ensemble_res
        self.threshold_val = float(
            get_threshold_value(norm_scores, method="contamination",
                                contamination=self.contamination)
        )
        self.threshold_method = "contamination"

        self.experiment_tracker.record_experiment(
            dataset_name=self.dataset_name,
            df=self.df_raw,
            model_name=self.display_model_name,
            parameters={"models": models, "method": aggregation_method, "weights": weights},
            results={
                "fit_time_sec": 0.0,
                "n_anomalies": int((predictions == -1).sum()),
                "anomaly_fraction": float((predictions == -1).mean()),
            },
        )

        return ensemble_res

    def get_detection_overview(self) -> dict[str, Any]:
        """Return high-level KPIs and distribution bins."""
        if (
            not self.detection_run
            or self.norm_scores is None
            or self.predictions is None
            or self.severity is None
        ):
            return {"ready": False}

        predictions = self.predictions
        severity = self.severity
        norm_scores = self.norm_scores

        n_total = len(predictions)
        n_anom = int((predictions == -1).sum())
        anom_mask = predictions == -1
        anom_sev = severity[anom_mask] if n_anom > 0 else np.array([])

        n_crit = int((anom_sev == "CRITICAL").sum())
        n_high = int((anom_sev == "HIGH").sum())
        n_med = int((anom_sev == "MEDIUM").sum())
        n_low = int((anom_sev == "LOW").sum())

        anom_scores = norm_scores[anom_mask] if n_anom > 0 else np.array([0.0])
        normal_scores = norm_scores[~anom_mask] if (n_total - n_anom) > 0 else np.array([0.0])

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
                "score": round(float(norm_scores[idx]), 4),
                "is_anomaly": bool(predictions[idx] == -1),
                "severity": str(severity[idx]),
            })

        return {
            "ready": True,
            "model_name": self.display_model_name,
            "n_total": n_total,
            "n_anomalies": n_anom,
            "anomaly_rate": round(float(n_anom / n_total), 4) if n_total > 0 else 0,
            "avg_score": round(float(np.mean(norm_scores)), 4),
            "median_anomaly_score": round(float(np.median(anom_scores)), 4),
            "max_score": round(float(np.max(norm_scores)), 4),
            "threshold_val": round(self.threshold_val, 4),
            "threshold_method": self.threshold_method,
            "severity_counts": {
                "CRITICAL": n_crit,
                "HIGH": n_high,
                "MEDIUM": n_med,
                "LOW": n_low,
            },
            "severity_bands": self.severity_bands,
            "score_normalization": (
                self.normalizer.describe() if self.normalizer else None
            ),
            "score_histogram": hist_data,
            "score_timeline": timeline,
        }

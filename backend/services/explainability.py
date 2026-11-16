"""
backend/services/explainability.py

Local and global explanations, counterfactuals, verification, anomaly queries and explanation stability.

Extracted verbatim from the former monolithic ``backend/state.py``.  The
methods still read and write the shared :class:`backend.state.AppState`
instance through ``self``; the split is organisational, and it lets a
reviewer read one concern at a time.
"""

from typing import Any

import numpy as np
import pandas as pd

from backend.config import settings
from backend.services.base import ServiceState, serialized
from backend.services.util import _accepts_random_state
from src.detection.ranking import (
    lookup_by_original_index,
)
from src.detection.scoring import ScoreNormalizer, normalize_scores
from src.explainability.counterfactuals import counterfactuals_for_anomalies
from src.explainability.explanation_generator import ExplanationGenerator
from src.explainability.feature_contributions import perturbation_importance
from src.explainability.validation import explanation_stability, faithfulness
from src.models.model_manager import (
    DEFAULT_PARAMS,
)


class ExplainabilityServiceMixin(ServiceState):
    """Local and global explanations, counterfactuals, verification, anomaly queries and explanation stability."""

    def get_anomalies(
        self,
        search: str | None = None,
        severity_filter: list[str] | None = None,
        min_score: float = 0.0,
        max_score: float = 1.0,
        anomalies_only: bool = True,
        sort_by: str = "rank",
        ascending: bool = True,
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        """Query and paginate anomaly records."""
        if not self.detection_run or self.df_results is None:
            return {"total": 0, "items": []}

        # Defence in depth: the router bounds page/page_size, but this method is
        # also called from run.py and tests, so clamp here too rather than
        # trusting every caller.
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), settings.MAX_PAGE_SIZE))

        df = self.df_results.copy()
        if anomalies_only:
            df = df[df["is_anomaly"]]

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

    @serialized
    def investigate_anomaly(self, row_id: int) -> dict[str, Any]:
        """Provide single-anomaly deep-dive and transparent explanation."""
        if (
            not self.detection_run
            or self.df_results is None
            or self.X is None
            or self.df_raw is None
            or self.norm_scores is None
            or self.predictions is None
            or self.severity is None
        ):
            raise ValueError("Detection must be completed first.")

        df_raw = self.df_raw
        if row_id < 0 or row_id >= len(df_raw):
            raise IndexError(f"Observation ID {row_id} out of bounds.")

        norm_score = float(self.norm_scores[row_id])
        sev = str(self.severity[row_id])
        is_anom = bool(self.predictions[row_id] == -1)

        # df_results is sorted (anomalies first), so positional lookup would
        # return a different record. Resolve by the original index instead.
        result_row = lookup_by_original_index(self.df_results, row_id)
        rank_val = result_row.get("rank") if result_row is not None else None
        rank = int(rank_val) if rank_val is not None and pd.notna(rank_val) else None

        orig_vals = df_raw.iloc[row_id].replace({np.nan: None}).to_dict()
        ref_stats = self.pipeline.get_reference_stats() if self.pipeline else {}

        # Look in cached explanations or compute on the fly
        exp = next((e for e in self.explanations if e.get("row_id") == row_id), None)
        if exp is None:
            exp = self._build_explanation(row_id, use_shap=False)

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
                    "robust_z_score": None,
                })

        # Counterfactual: what would have to change for this row to look normal.
        counterfactual = self.get_counterfactual(row_id)

        # Behavioural check that the named features actually drive the score.
        verification = self.get_explanation_verification(row_id)

        return {
            "row_id": row_id,
            "rank": rank,
            "model_name": self.display_model_name,
            "normalized_score": norm_score,
            "is_anomaly": is_anom,
            "severity": sev,
            "threshold": self.threshold_val,
            "explanation_text": exp.get("explanation_text", ""),
            "confidence_note": exp.get("confidence_note", ""),
            "top_features": exp.get("top_features", []),
            "feature_comparison": comparison_list,
            "counterfactual": counterfactual,
            "verification": verification,
            "raw_record": orig_vals,
        }

    def _make_score_fn(self):
        """
        Build a callable returning the normalised score the UI displays.

        Uses the normaliser fitted during detection so the mapping stays
        identical to the one behind the threshold and the severity tiers.
        """
        detector = self.model_manager._detectors.get(self.active_model)
        normalizer = self.normalizer

        def score_fn(obs):
            arr = np.atleast_2d(np.asarray(obs, dtype=np.float64))
            if detector is None:
                return np.zeros(arr.shape[0], dtype=np.float64)
            raw = detector.score_samples(arr)
            if normalizer is not None:
                return normalizer.transform(raw)
            return normalize_scores(raw)

        return score_fn, detector

    def _build_explanation(self, row_id: int, use_shap: bool = False) -> dict[str, Any]:
        """Compute a local explanation for one row on demand."""
        if (
            self.X is None
            or self.norm_scores is None
            or self.severity is None
            or self.df_raw is None
        ):
            raise ValueError("Detection must be completed before explaining a row.")

        X = self.X
        df_raw = self.df_raw
        score_fn, detector = self._make_score_fn()
        sklearn_model = getattr(detector, "_model", None) if detector else None

        explainer = ExplanationGenerator(
            model_name=self.active_model,
            sklearn_model=sklearn_model,
            X_train=X,
            feature_names=self.feature_names,
            reference_stats=self.pipeline.get_reference_stats() if self.pipeline else {},
            norm_score_fn=score_fn,
            use_shap=use_shap,
            top_k=8,
            denormalize=(
                self.pipeline.denormalize if self.pipeline is not None else None
            ),
            score_space=self.normalizer.describe() if self.normalizer else None,
        )
        return explainer.explain_observation(
            observation=X[row_id],
            norm_score=float(self.norm_scores[row_id]),
            severity=str(self.severity[row_id]),
            row_id=row_id,
            original_values=df_raw.iloc[row_id].replace({np.nan: None}).to_dict(),
        )

    def get_counterfactual(self, row_id: int, immutable_features: list[str] | None = None) -> dict[str, Any]:
        """
        Suggest what would have to change for a flagged row to look normal.

        Returns a structured "not found" payload rather than raising, because a
        missing recommendation is a legitimate outcome for outliers that sit
        far from every normal prototype.
        """
        if not self.detection_run or self.X is None or self.predictions is None:
            return {"found": False, "reason": "Run detection first.",
                    "changes": [], "text": ""}

        score_fn, _ = self._make_score_fn()
        normal_mask = self.predictions == 1

        results = counterfactuals_for_anomalies(
            anomaly_indices=[int(row_id)],
            X=self.X,
            normal_mask=normal_mask,
            score_fn=score_fn,
            feature_names=self.feature_names,
            original_df=self.df_raw,
            denormalize=(
                self.pipeline.denormalize if self.pipeline is not None else None
            ),
            score_threshold=self.threshold_val,
            immutable_features=immutable_features if immutable_features is not None else (self.pipeline.id_cols if self.pipeline else []),
        )
        return results[0] if results else {
            "found": False, "reason": "No result.", "changes": [], "text": "",
        }

    def get_explanation_verification(self, row_id: int) -> dict[str, Any]:
        """
        Check that the explanation for a row is behaviourally faithful.

        Masks features in the order the explanation ranks them and measures how
        fast the anomaly score collapses, comparing against random orderings on
        the same row.  A ranking that performs no better than random is
        reported as unverified rather than presented as a finding.
        """
        if not self.detection_run or self.X is None:
            return {"overall": "UNKNOWN", "checks_run": 0,
                    "summary": "Run detection first."}

        exp = self._build_explanation(row_id, use_shap=False)
        order = [
            self.feature_names.index(e["feature"])
            for e in exp.get("top_features", [])
            if e["feature"] in self.feature_names
        ]
        if not order:
            return {"overall": "UNKNOWN", "checks_run": 0,
                    "summary": "No ranked features to verify."}

        score_fn, _ = self._make_score_fn()
        def single(z):
            return float(np.asarray(score_fn(np.atleast_2d(z))).ravel()[0])

        faith = faithfulness(
            self.X[int(row_id)], single, order, n_random=3
        )
        ok = (
            faith.get("relative_improvement") is not None
            and faith["relative_improvement"] <= -0.10
        )
        return {
            "overall": "BEHAVIOURALLY FAITHFUL" if ok else "UNVERIFIED",
            "checks_run": 1,
            "summary": faith.get("verdict", ""),
            "faithfulness": faith,
        }

    @serialized
    def generate_all_explanations(self, use_shap: bool = False, top_k: int = 8) -> dict[str, Any]:
        """Compute global explainability across all flagged anomalies."""
        if (
            not self.detection_run
            or self.X is None
            or self.norm_scores is None
            or self.predictions is None
            or self.severity is None
        ):
            raise ValueError("Detection must be run first.")

        X = self.X
        score_fn, detector = self._make_score_fn()
        sklearn_model = getattr(detector, "_model", None) if detector else None

        explainer = ExplanationGenerator(
            model_name=self.active_model,
            sklearn_model=sklearn_model,
            X_train=X,
            feature_names=self.feature_names,
            reference_stats=self.pipeline.get_reference_stats() if self.pipeline else {},
            norm_score_fn=score_fn,
            use_shap=use_shap,
            top_k=top_k,
            denormalize=(
                self.pipeline.denormalize if self.pipeline is not None else None
            ),
            score_space=self.normalizer.describe() if self.normalizer else None,
        )

        explanations = explainer.explain_batch(
            X=X,
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

    @serialized
    def explanation_stability(
        self,
        seeds: list[int] | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Refit the model under several seeds and measure how much the feature
        ranking moves.

        Uses perturbation importance (model-agnostic, so it works for every
        detector) rather than SHAP, which keeps this affordable to run on
        demand.
        """
        if self.X is None or self.predictions is None:
            raise ValueError("Run detection first.")

        target = model_name or self.active_model
        seeds = list(seeds or [11, 23, 37, 53, 71])
        anom_idx = np.flatnonzero(self.predictions == -1)[:25]

        if anom_idx.size == 0:
            return {
                "n_fits": 0,
                "verdict": "No flagged anomalies to compare.",
                "mean_rank_correlation": None,
                "top_k_features": [],
                "per_fit_leader": [],
                "top_k_retention": None,
                "min_rank_correlation": None,
            }

        attribute_rows: list[np.ndarray] = []
        fit_errors: list[dict[str, str]] = []
        supports_seed = _accepts_random_state(target)
        for seed in seeds:
            params = dict(DEFAULT_PARAMS.get(target, {}))
            try:
                if supports_seed:
                    params["random_state"] = seed
                    fit_X = self.X
                else:
                    # Detectors such as LOF, k-NN, HBOS and One-Class SVM expose
                    # no random_state.  Injecting one raised TypeError inside
                    # create_detector, the loop swallowed it, and stability
                    # reported "need at least two fits" for those models
                    # forever.  A deterministic estimator has no seed to vary,
                    # so perturb the *training sample* instead: a bootstrap
                    # resample is a legitimate test of whether the explanation
                    # survives a different, equally valid fit.
                    rng = np.random.default_rng(seed)
                    boot = rng.integers(0, len(self.X), size=len(self.X))
                    fit_X = self.X[boot]

                detector = self.model_manager.create_detector(target, params)
                detector.fit(fit_X)
                local = ScoreNormalizer().fit(detector.score_samples(self.X))
                def fn(z, d=detector, n=local):
                    return n.transform(d.score_samples(np.atleast_2d(z)))
                acc = np.zeros(len(self.feature_names), dtype=np.float64)
                for idx in anom_idx:
                    for entry in perturbation_importance(
                        self.X[int(idx)], fn, self.feature_names
                    ):
                        acc[self.feature_names.index(entry["feature"])] += \
                            entry["importance"]
                attribute_rows.append(acc)
            except Exception as exc:  # recorded, never silently skipped
                fit_errors.append(
                    {"seed": str(seed), "error": f"{type(exc).__name__}: {exc}"}
                )

        refit_strategy = (
            "varied random_state" if supports_seed
            else "bootstrap resample (detector is deterministic)"
        )

        if len(attribute_rows) < 2:
            return {
                "n_fits": len(attribute_rows),
                "verdict": "Need at least two successful fits to measure stability.",
                "mean_rank_correlation": None,
                "top_k_features": [],
                "per_fit_leader": [],
                "top_k_retention": None,
                "min_rank_correlation": None,
                "refit_strategy": refit_strategy,
                "errors": fit_errors,
            }

        return {
            "model": target,
            "seeds": seeds,
            "refit_strategy": refit_strategy,
            "errors": fit_errors,
            "n_anomalies_probed": int(anom_idx.size),
            **explanation_stability(
                np.vstack(attribute_rows), self.feature_names, top_k=3
            ),
        }

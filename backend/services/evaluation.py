"""
backend/services/evaluation.py

Leakage-safe held-out evaluation across one or all detectors.

Extracted verbatim from the former monolithic ``backend/state.py``.  The
methods still read and write the shared :class:`backend.state.AppState`
instance through ``self``; the split is organisational, and it lets a
reviewer read one concern at a time.
"""

from typing import Any

import numpy as np
import pandas as pd

from backend.services.base import ServiceState, serialized
from backend.services.util import _round_or_none
from src.data.preprocessing import PreprocessingPipeline
from src.evaluation.holdout import (
    evaluate_holdout_split,
    holdout_split,
    split_raw_frame,
)
from src.models.model_manager import (
    DEFAULT_PARAMS,
    DETECTOR_REGISTRY,
)


class EvaluationServiceMixin(ServiceState):
    """Leakage-safe held-out evaluation across one or all detectors."""

    @serialized
    def evaluate_holdout_metrics(
        self,
        model_name: str | None = None,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Report in-sample vs held-out metrics for the active (or given) model.

        Fitting and scoring the same rows inflates every metric. This reports
        both, plus the optimism gap, so the headline number can be read with
        the right caveat attached.
        """
        if self.X is None:
            raise ValueError("Feature matrix is not ready.")

        target = model_name or self.active_model
        params = dict(DEFAULT_PARAMS.get(target, {}))
        if target == "Local Outlier Factor":
            # novelty=True is required to score rows the model never saw.
            params["novelty"] = True

        def factory():
            return self.model_manager.create_detector(target, params)

        inputs = self._holdout_inputs(test_size=test_size, random_state=random_state)
        report = evaluate_holdout_split(
            factory,
            inputs["X_fit"],
            inputs["X_test"],
            y_fit=inputs["y_fit"],
            y_test=inputs["y_test"],
            threshold_method=self.threshold_method,
            contamination=self.contamination,
        )
        report["model"] = target
        report["test_size"] = test_size
        report["random_state"] = random_state
        report["preprocessing"] = inputs["preprocessing"]
        if inputs["preprocessing_warning"]:
            report["preprocessing_warning"] = inputs["preprocessing_warning"]

        if report.get("available"):
            gap = report.get("optimism_gap", {}).get("roc_auc")
            if gap is None:
                summary = "Held-out metrics unavailable; see the reason field."
            elif gap < -0.05:
                # A negative gap is NOT "the two agree".  It means the in-sample
                # score was pessimistic, which happens when the density /
                # covariance estimate is distorted by the fit split (too few
                # rows, contamination inside the training set, EM local optima).
                # Reporting it as agreement would hide a broken estimator.
                summary = (
                    f"Held-out ROC-AUC is {abs(gap):.3f} HIGHER than in-sample. "
                    "A materially negative gap is not a good sign: the in-sample "
                    "figure is pessimistic, usually because the detector's "
                    "density estimate is distorted by the training rows. Quote "
                    "the held-out value and investigate the estimator rather "
                    "than treating the in-sample number as a ceiling."
                )
            elif gap <= 0.02:
                summary = (
                    "In-sample and held-out ROC-AUC agree closely, so the "
                    "headline figure is not an artefact of scoring the "
                    "training rows."
                )
            elif gap <= 0.06:
                summary = (
                    f"Held-out ROC-AUC is {gap:.3f} below the in-sample value. "
                    "Quote the held-out figure."
                )
            else:
                summary = (
                    f"Held-out ROC-AUC drops by {gap:.3f}. The in-sample number "
                    "is substantially optimistic and should not be reported "
                    "on its own."
                )
            report["summary"] = summary
        else:
            report["summary"] = report.get("reason") or "Unavailable."

        self.holdout_report = report
        return report

    @serialized
    def evaluate_all_models_holdout(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Out-of-sample comparison table across every registered detector."""
        if self.X is None:
            raise ValueError("Feature matrix is not ready.")

        inputs = self._holdout_inputs(test_size=test_size, random_state=random_state)
        factories: dict[str, Any] = {}
        for name in DETECTOR_REGISTRY:
            params = dict(DEFAULT_PARAMS.get(name, {}))
            if name == "Local Outlier Factor":
                params["novelty"] = True
            factories[name] = (
                lambda n=name, p=params: self.model_manager.create_detector(n, p)
            )

        rows: list[dict[str, Any]] = []
        for name, factory in factories.items():
            try:
                res = evaluate_holdout_split(
                    factory,
                    inputs["X_fit"],
                    inputs["X_test"],
                    y_fit=inputs["y_fit"],
                    y_test=inputs["y_test"],
                    threshold_method=self.threshold_method,
                    contamination=self.contamination,
                )
            except Exception as exc:  # one failing detector must not kill the table
                rows.append({"Model": name, "Error": f"{type(exc).__name__}: {exc}"})
                continue

            if not res.get("available"):
                rows.append(
                    {"Model": name, "Available": False, "Reason": res.get("reason")}
                )
                continue

            ins, oos = res["in_sample"], res["out_of_sample"]
            gaps = res.get("optimism_gap", {})
            rows.append(
                {
                    "Model": name,
                    "ROC-AUC (in-sample)": _round_or_none(ins.get("roc_auc")),
                    "ROC-AUC (out-of-sample)": _round_or_none(oos.get("roc_auc")),
                    "AUC gap": _round_or_none(gaps.get("roc_auc")),
                    "PR-AUC (in-sample)": _round_or_none(ins.get("pr_auc")),
                    "PR-AUC (out-of-sample)": _round_or_none(oos.get("pr_auc")),
                    "F1 (out-of-sample)": _round_or_none(oos.get("f1")),
                    "Recall (out-of-sample)": _round_or_none(oos.get("recall")),
                    "Precision (out-of-sample)": _round_or_none(oos.get("precision")),
                    "Anomalies (out-of-sample)": oos.get("n_detected_anomalies"),
                    "n_test": res.get("n_test"),
                }
            )

        table = pd.DataFrame(rows)
        self.holdout_table = table
        records = (
            table.replace({np.nan: None}).to_dict(orient="records")
            if table is not None and not table.empty
            else []
        )
        return {
            "test_size": test_size,
            "random_state": random_state,
            "table": records,
            "preprocessing": inputs["preprocessing"],
            "preprocessing_warning": inputs["preprocessing_warning"],
            "note": (
                "Rows were split BEFORE preprocessing, so the scaler, imputer "
                "and encoder were fitted on the training rows only. Every model "
                "here is fitted on that training split and scored on rows it "
                "never saw."
            ),
        }

    def _holdout_inputs(
        self,
        test_size: float,
        random_state: int,
    ) -> dict[str, Any]:
        """
        Build leakage-free fit/test feature matrices.

        The raw frame is split first and the preprocessing pipeline is fitted
        on the fit rows only.  Splitting ``self.X`` instead lets every held-out
        row influence the scaler's mean/std and the encoder's category set,
        which is the classic preprocessing leakage: the reported held-out
        metric is then partly in-sample no matter which detector is used.
        """
        if self.df_raw is None:
            raise ValueError(
                "A raw dataset is required for a leakage-free holdout split."
            )

        split = split_raw_frame(
            self.df_raw,
            label_col=self.label_col,
            test_size=test_size,
            random_state=random_state,
        )

        try:
            pipeline = PreprocessingPipeline(
                label_col=self.label_col,
                scaler_type=self.scaler_type,
                drop_duplicates=False,
                id_cols=self.id_cols,
            )
            X_fit, _ = pipeline.fit_transform(split["fit_df"])
            X_test = pipeline.transform(split["test_df"])
        except Exception as exc:
            # Unseen categories or dtype drift can make transform() fail on a
            # pipeline fitted on a subset.  Fall back to splitting the existing
            # matrix, but say so in the report: a silently leaky number is
            # worse than a labelled compromise.
            if self.X is None:
                raise ValueError(
                    "The fit-only preprocessing pipeline could not transform the "
                    f"held-out rows and no preprocessed matrix exists to fall "
                    f"back on: {type(exc).__name__}: {exc}"
                ) from exc
            fallback = holdout_split(
                self.X,
                y=self.y_true,
                test_size=test_size,
                random_state=random_state,
            )
            return {
                "X_fit": fallback["X_fit"],
                "X_test": fallback["X_test"],
                "y_fit": fallback["y_fit"],
                "y_test": fallback["y_test"],
                "preprocessing": (
                    "fallback: preprocessor was fitted on the full frame "
                    "because the fit-only pipeline could not transform the "
                    "held-out rows"
                ),
                "preprocessing_warning": f"{type(exc).__name__}: {exc}",
            }

        return {
            "X_fit": X_fit,
            "X_test": X_test,
            "y_fit": split["y_fit"],
            "y_test": split["y_test"],
            "preprocessing": "preprocessor fitted on the fit split only",
            "preprocessing_warning": None,
        }

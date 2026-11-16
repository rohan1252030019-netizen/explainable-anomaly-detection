"""
src/evaluation/synthetic_lab.py

Synthetic Anomaly Injection and Benchmark Laboratory for XAUD 2.0.

Provides controlled, traceable anomaly generation across distinct anomaly types:
1. Point / global anomalies: severe deviations along marginal distributions.
2. Multivariate correlation anomalies: marginal distributions stay nominal but
   the joint dependence structure is severed.
3. Cluster anomalies: dense micro-clusters in low-density regions.
4. Contextual anomalies: values that stay inside the marginal support but
   violate the conditional distribution given a context feature.

Evaluation protocol
-------------------
The benchmark is out-of-sample.  Injected rows are excluded from the training
split, the detectors are fitted on clean rows only, and every metric is computed
on held-out rows the detector never saw.  Fitting and scoring the same injected
rows (the previous behaviour) turned the benchmark into a memorisation test and
reported ROC-AUC near 1.0 for any detector.

Difficulty levels
-----------------
``easy``   multiplies the displacement by 1.5 (blatant outliers)
``medium`` leaves the requested severity unchanged
``hard``   multiplies it by 0.5 (subtle, near the noise floor)

Contamination propagation
-------------------------
``contamination`` is translated into each detector's own flag budget
(``contamination`` for most, ``nu`` for One-Class SVM) instead of silently
falling back to that detector's default.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.models.model_manager import DEFAULT_PARAMS, ModelManager

VALID_ANOMALY_TYPES = ("point", "multivariate", "cluster", "contextual")
VALID_DIFFICULTIES = ("easy", "medium", "hard")

_DIFFICULTY_SCALE = {"easy": 1.5, "medium": 1.0, "hard": 0.5}


class SyntheticInjectionLab:
    """
    Controlled laboratory for injecting specific anomaly mechanisms into
    unlabelled feature matrices and benchmarking detector sensitivity.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # ── Injection ──────────────────────────────────────────────────────────

    def inject_anomalies(
        self,
        X: np.ndarray,
        anomaly_type: str = "multivariate",
        contamination: float = 0.05,
        severity: float = 3.5,
        difficulty: str = "medium",
    ) -> dict[str, Any]:
        """
        Inject controlled synthetic anomalies into feature matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Base feature matrix.  Injected rows are *replaced* in a copy; the
            input array is never modified.
        anomaly_type : {"point", "multivariate", "cluster", "contextual"}
        contamination : float
            Fraction of rows to convert into anomalies.
        severity : float
            Base displacement magnitude, in units of the feature's spread.
        difficulty : {"easy", "medium", "hard"}
            Scales ``severity`` (see the module docstring).

        Returns
        -------
        dict with keys ``X_injected``, ``injected_mask``, ``n_injected``,
        ``anomaly_type``, ``difficulty``, ``severity_effective``,
        ``contamination_actual``.
        """
        if anomaly_type not in VALID_ANOMALY_TYPES:
            raise ValueError(
                f"Unknown anomaly_type '{anomaly_type}'. "
                f"Choose from {list(VALID_ANOMALY_TYPES)}."
            )
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Unknown difficulty '{difficulty}'. "
                f"Choose from {list(VALID_DIFFICULTIES)}."
            )
        if not 0.0 < contamination < 1.0:
            raise ValueError(
                f"contamination must be in (0, 1), got {contamination}."
            )
        if severity <= 0:
            raise ValueError(f"severity must be positive, got {severity}.")

        rng = np.random.default_rng(self.random_state)
        X_clean = np.asarray(X, dtype=np.float64)
        if X_clean.ndim != 2 or X_clean.shape[0] < 2:
            raise ValueError("X must be a 2-D matrix with at least two rows.")
        n_samples, n_features = X_clean.shape

        effective_severity = float(severity) * _DIFFICULTY_SCALE[difficulty]
        n_anomalies = max(1, int(round(n_samples * contamination)))
        n_anomalies = min(n_anomalies, n_samples)
        injected_indices = rng.choice(n_samples, size=n_anomalies, replace=False)
        injected_mask = np.zeros(n_samples, dtype=bool)
        injected_mask[injected_indices] = True

        X_mod = X_clean.copy()
        means = np.mean(X_clean, axis=0)
        stds = np.std(X_clean, axis=0)
        stds[stds == 0] = 1.0

        if anomaly_type == "point":
            # Shift a random subset of features by several standard deviations.
            for idx in injected_indices:
                n_feats_to_shift = int(
                    rng.integers(1, max(2, n_features // 2 + 1))
                )
                shift_cols = rng.choice(
                    n_features, size=n_feats_to_shift, replace=False
                )
                direction = rng.choice([-1.0, 1.0], size=n_feats_to_shift)
                X_mod[idx, shift_cols] += (
                    direction * effective_severity * stds[shift_cols]
                )

        elif anomaly_type == "multivariate":
            if n_features < 2:
                X_mod[injected_indices, 0] += effective_severity * stds[0]
            else:
                # Keep one anchor feature at its observed value and rebuild every
                # other column from a DIFFERENT normal row.  Marginals are
                # preserved exactly (each value is a real value from that
                # column), yet the joint pairing is destroyed, so a detector that
                # only looks at marginals sees nothing while a detector that
                # models dependence sees the row as atypical.  The previous
                # implementation permuted values only *within* the injected
                # block, which usually left much of the dependence intact.
                donor_pool = np.setdiff1d(
                    np.arange(n_samples), injected_indices, assume_unique=False
                )
                if donor_pool.size == 0:
                    donor_pool = np.arange(n_samples)
                for idx in injected_indices:
                    donors = rng.choice(
                        donor_pool, size=n_features, replace=True
                    )
                    for j in range(1, n_features):
                        X_mod[idx, j] = X_clean[donors[j], j]

        elif anomaly_type == "cluster":
            centroid = means + effective_severity * stds * rng.choice(
                [-1.0, 1.0], size=n_features
            )
            cluster_noise = rng.normal(
                0.0, 0.2, size=(n_anomalies, n_features)
            ) * stds
            X_mod[injected_indices] = centroid + cluster_noise

        else:  # contextual
            # A contextual anomaly is a value that is individually plausible but
            # wrong for its context.  Pick the most strongly correlated pair
            # (context, target), keep the context fixed, and move the target far
            # from its conditional mean in the direction it was already off --
            # then clip to the observed marginal range so the value never looks
            # extreme on its own.
            if n_features < 2:
                raise ValueError(
                    "contextual injection needs at least two features."
                )
            corr = np.corrcoef(X_clean, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 0.0)
            row, col = np.unravel_index(
                int(np.argmax(np.abs(corr))), corr.shape
            )
            context, target = int(row), int(col)
            if np.isclose(corr[context, target], 0.0):
                context, target = 0, 1

            slope, intercept = np.polyfit(
                X_clean[:, context], X_clean[:, target], 1
            )
            residual = X_clean[:, target] - (
                slope * X_clean[:, context] + intercept
            )
            residual_std = float(np.std(residual))
            if residual_std <= 0 or not np.isfinite(residual_std):
                residual_std = float(stds[target])

            lo = float(np.min(X_clean[:, target]))
            hi = float(np.max(X_clean[:, target]))
            for idx in injected_indices:
                conditional_mean = slope * X_clean[idx, context] + intercept
                sign = (
                    1.0
                    if X_clean[idx, target] >= conditional_mean
                    else -1.0
                )
                value = (
                    conditional_mean
                    + sign * effective_severity * residual_std
                )
                X_mod[idx, target] = float(np.clip(value, lo, hi))

        return {
            "X_injected": X_mod,
            "injected_mask": injected_mask,
            "injected_indices": injected_indices,
            "n_injected": int(n_anomalies),
            "anomaly_type": anomaly_type,
            "difficulty": difficulty,
            "severity_effective": round(effective_severity, 4),
            "contamination_actual": round(n_anomalies / n_samples, 4),
        }

    # ── Benchmark ──────────────────────────────────────────────────────────

    def benchmark_detectors(
        self,
        X: np.ndarray,
        anomaly_type: str = "multivariate",
        contamination: float = 0.05,
        models_to_test: list[str] | None = None,
        feature_names: list[str] | None = None,
        difficulty: str = "medium",
        severity: float = 3.5,
        train_fraction: float = 0.6,
    ) -> dict[str, Any]:
        """
        Benchmark detectors on injected anomalies under an out-of-sample split.

        Injected rows are held out entirely: the detectors are fitted on a clean
        subset of the untouched rows and every metric is computed on the held-out
        rows plus the injected anomalies.
        """
        if not 0.2 <= float(train_fraction) <= 0.9:
            raise ValueError(
                f"train_fraction must be in [0.2, 0.9], got {train_fraction}."
            )

        X_all = np.asarray(X, dtype=np.float64)
        n_samples = X_all.shape[0]
        if n_samples < 20:
            raise ValueError(
                "A disjoint fit/evaluation split needs at least 20 rows; "
                f"received {n_samples}."
            )

        rng = np.random.default_rng(self.random_state)
        injection = self.inject_anomalies(
            X_all,
            anomaly_type=anomaly_type,
            contamination=contamination,
            severity=severity,
            difficulty=difficulty,
        )
        injected_mask = injection["injected_mask"]

        # Fit only on rows that were NOT converted into anomalies, so the
        # injected outliers cannot shape the model they are scored against.
        clean_idx = np.flatnonzero(~injected_mask)
        eval_idx = np.flatnonzero(injected_mask)
        n_fit = int(round(clean_idx.size * float(train_fraction)))
        if n_fit < 2:
            raise ValueError(
                "Not enough clean rows to form a training split; lower "
                "contamination or provide more data."
            )
        shuffled = rng.permutation(clean_idx)
        fit_idx = np.sort(shuffled[:n_fit])
        remaining_clean = np.sort(shuffled[n_fit:])

        X_fit = X_all[fit_idx]
        eval_indices = np.concatenate([remaining_clean, eval_idx])
        X_eval = injection["X_injected"][eval_indices]
        y_true = injected_mask[eval_indices].astype(int)

        candidate_models = models_to_test or [
            "Isolation Forest",
            "Local Outlier Factor",
            "k-Nearest Neighbors",
            "Histogram-Based Outlier Score",
            "PCA Reconstruction",
        ]

        manager = ModelManager()
        results_table: list[dict[str, Any]] = []
        k = int(y_true.sum())

        for model_name in candidate_models:
            if model_name not in manager.get_available_models():
                continue

            try:
                params = self._detector_params(model_name, contamination)
                fit_res = manager.fit(
                    model_name, X_fit, feature_names=feature_names, params=params
                )
                if not fit_res.get("success"):
                    raise RuntimeError(
                        fit_res.get("error") or "detector failed to fit"
                    )

                detector = manager.get_detector(model_name)
                if detector is None:  # pragma: no cover - fit just succeeded
                    raise RuntimeError(f"'{model_name}' was not retained after fit.")
                raw_scores = np.asarray(
                    detector.score_samples(X_eval), dtype=np.float64
                )
                preds = np.asarray(detector.predict(X_eval), dtype=int)

                # Lower raw score is more anomalous in XAUD; invert for ROC/PR
                # where 1 is anomalous.
                inverted_scores = -raw_scores

                roc_auc = float(roc_auc_score(y_true, inverted_scores))
                pr_auc = float(average_precision_score(y_true, inverted_scores))

                binary_preds = (preds == -1).astype(int)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, binary_preds, average="binary", zero_division=0
                )

                top_k_indices = np.argsort(-inverted_scores)[:k]
                prec_at_k = float(np.mean(y_true[top_k_indices]))

                results_table.append({
                    "model_name": model_name,
                    "family": manager.get_model_capabilities(model_name).get(
                        "family", "Unknown"
                    ),
                    "roc_auc": round(roc_auc, 4),
                    "pr_auc": round(pr_auc, 4),
                    "precision_at_k": round(prec_at_k, 4),
                    "f1_score": round(float(f1), 4),
                    "recall": round(float(rec), 4),
                    "fit_time_sec": fit_res["fit_time_sec"],
                })

            except Exception as exc:  # recorded per model, never swallowed
                results_table.append({
                    "model_name": model_name,
                    "family": "Error",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        # Sort by PR-AUC (the informative metric under class imbalance).
        results_table.sort(
            key=lambda r: r.get("pr_auc", -1.0), reverse=True
        )

        return {
            "experiment_type": "Synthetic Anomaly Benchmark",
            "anomaly_type": anomaly_type,
            "difficulty": difficulty,
            "severity_effective": injection["severity_effective"],
            "n_samples": int(n_samples),
            "n_fit": int(X_fit.shape[0]),
            "n_evaluated": int(X_eval.shape[0]),
            "n_injected": int(injection["n_injected"]),
            "contamination_rate": injection["contamination_actual"],
            "evaluation_contamination_rate": round(
                float(k / max(len(y_true), 1)), 4
            ),
            "evaluation_protocol": (
                "out-of-sample: detectors fitted on clean rows only and scored "
                "on held-out rows plus injected anomalies"
            ),
            "results_table": results_table,
            "top_model": results_table[0]["model_name"] if results_table else None,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _detector_params(
        model_name: str, contamination: float
    ) -> dict[str, Any]:
        """Translate the benchmark's contamination into a detector's budget."""
        params = dict(DEFAULT_PARAMS.get(model_name, {}))
        if model_name == "One-Class SVM":
            params["nu"] = float(contamination)
        elif "contamination" in params:
            params["contamination"] = float(contamination)
        return params

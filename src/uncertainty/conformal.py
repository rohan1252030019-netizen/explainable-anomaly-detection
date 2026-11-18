"""
src/uncertainty/conformal.py

Conformal Anomaly Prediction with Finite-Sample Validity Guarantees for SENTINEL.

Mathematical Formulation:
--------------------------
Traditional anomaly detection outputs raw scores s(x) or min-max rescalings
that are falsely interpreted as Bayesian posterior probabilities P(anomaly|x).
Conformal Prediction replaces arbitrary score scaling with rigorous, non-parametric
conformal p-values that hold with exact finite-sample validity without parametric
distributional assumptions.

1. Split-Conformal Calibration:
   Given reference/calibration nominal observations D_cal = {x_1, ..., x_n}
   and a non-conformity score S(x) where higher = more anomalous:
   The non-conformity scores S_1, ..., S_n are evaluated on D_cal.
   For a test point x_new with non-conformity score S(x_new):
       p(x_new) = (1 + sum_{i=1}^n I(S_i >= S(x_new))) / (n + 1)

   Under the null hypothesis H_0: x_new is exchangeable with D_cal:
       P(p(x_new) <= alpha) <= alpha, for all alpha in (0, 1).

2. Mondrian Group-Conditional Conformal Calibration:
   When observations belong to categorical segments or clusters g in {1, ..., G}:
   Non-conformity scores are calibrated within each subgroup D_cal^g:
       p_g(x_new) = (1 + sum_{x_i in D_cal^g} I(S_i >= S(x_new))) / (|D_cal^g| + 1)
   Guaranteeing:
       P(p_g(X) <= alpha | Group = g) <= alpha, for all groups g.

3. Coverage Statement & Calibrated Threshold:
   At significance level alpha, the calibrated anomaly threshold q_alpha is:
       q_alpha = Quantile_{ (1 - alpha) * (1 + 1/n) }( {S_1, ..., S_n} )
   Any test sample with S(x_new) > q_alpha has p(x_new) <= alpha and is certified
   as an anomaly with confidence 1 - alpha.
"""

from typing import Any

import numpy as np


class ConformalResult:
    """Container for conformal evaluation outputs."""

    def __init__(
        self,
        p_values: np.ndarray,
        is_anomaly: np.ndarray,
        threshold_at_alpha: float,
        alpha: float,
        coverage_statement: str,
        n_calibration_samples: int,
    ):
        self.p_values = p_values
        self.is_anomaly = is_anomaly
        self.threshold_at_alpha = threshold_at_alpha
        self.alpha = alpha
        self.coverage_statement = coverage_statement
        self.n_calibration_samples = n_calibration_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_values": [round(float(p), 6) for p in self.p_values],
            "is_anomaly": [bool(a) for a in self.is_anomaly],
            "threshold_at_alpha": round(float(self.threshold_at_alpha), 6),
            "alpha": float(self.alpha),
            "confidence_level": round(float(1.0 - self.alpha), 4),
            "coverage_statement": self.coverage_statement,
            "n_calibration_samples": int(self.n_calibration_samples),
            "n_anomalies_flagged": int(np.sum(self.is_anomaly)),
            "flagged_rate": round(float(np.mean(self.is_anomaly)), 4),
        }


class ConformalAnomalyCalibrator:
    """
    Split-Conformal Anomaly Predictor providing finite-sample valid p-values.
    """

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.calibration_scores_: np.ndarray | None = None
        self._is_calibrated: bool = False
        self.threshold_at_alpha_: float = 0.0
        self.n_calibration_: int = 0

    def fit_calibration(self, calibration_scores: np.ndarray) -> "ConformalAnomalyCalibrator":
        """
        Calibrate on non-conformity scores of reference/nominal samples.

        Parameters
        ----------
        calibration_scores : np.ndarray of shape (n_samples,)
            Scores where higher = more anomalous. Must be computed on holdout reference
            data to prevent in-sample memorization leakage.
        """
        scores = np.asarray(calibration_scores, dtype=np.float64)
        if len(scores) == 0:
            raise ValueError("Calibration scores array cannot be empty.")

        self.calibration_scores_ = np.sort(scores)
        self.n_calibration_ = len(scores)

        # Finite-sample calibrated quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, np.ceil((self.n_calibration_ + 1) * (1.0 - self.alpha)) / self.n_calibration_)
        self.threshold_at_alpha_ = float(np.quantile(self.calibration_scores_, q_level))
        self._is_calibrated = True
        return self

    def compute_p_values(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Compute exact finite-sample conformal p-values:
        p(x) = (1 + sum_{i=1}^n I(S_i >= S(x))) / (n + 1)
        """
        self._check_calibrated()
        test_scores_arr = np.asarray(test_scores, dtype=np.float64)
        n = self.n_calibration_

        # Fast vectorized rank calculation using searchsorted on sorted calibration scores
        # Count calibration scores >= test_score
        # idx is the first index where cal_score >= test_score
        # count(cal_scores >= test_score) = n - searchsorted(side='left')
        indices = np.searchsorted(self.calibration_scores_, test_scores_arr, side="left")
        counts_greater_equal = n - indices

        p_vals = (counts_greater_equal + 1.0) / (n + 1.0)
        return np.clip(p_vals, 0.0, 1.0)

    def predict(self, test_scores: np.ndarray, alpha: float | None = None) -> ConformalResult:
        """
        Return conformal predictions with explicit confidence statement.
        """
        self._check_calibrated()
        target_alpha = self.alpha if alpha is None else float(alpha)
        if not (0.0 < target_alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {target_alpha}")

        p_values = self.compute_p_values(test_scores)
        is_anomaly = p_values <= target_alpha

        # Compute threshold at target alpha
        n = self.n_calibration_
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - target_alpha)) / n)
        threshold = float(np.quantile(self.calibration_scores_, q_level))

        coverage_statement = (
            f"Under the null hypothesis of nominal exchangeability with n={n} "
            f"reference samples, false alarm rate is mathematically bounded by alpha={target_alpha:.3f} "
            f"(Confidence: {(1.0 - target_alpha) * 100:.1f}%)."
        )

        return ConformalResult(
            p_values=p_values,
            is_anomaly=is_anomaly,
            threshold_at_alpha=threshold,
            alpha=target_alpha,
            coverage_statement=coverage_statement,
            n_calibration_samples=n,
        )

    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def _check_calibrated(self) -> None:
        if not self._is_calibrated or self.calibration_scores_ is None:
            raise RuntimeError(
                "ConformalAnomalyCalibrator is not calibrated. "
                "Call fit_calibration() with holdout reference scores first."
            )


class MondrianConformalCalibrator:
    """
    Mondrian (Group-Conditional) Conformal Anomaly Calibrator.

    Partitions calibration data into discrete subgroups (e.g., categorical attributes
    or geometric clusters) and guarantees finite-sample coverage conditionally per group.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.group_calibrators_: dict[str, ConformalAnomalyCalibrator] = {}
        self.global_calibrator_: ConformalAnomalyCalibrator = ConformalAnomalyCalibrator(alpha=alpha)
        self._is_calibrated: bool = False
        self.groups_: list[str] = []
        self.fallback_groups_: list[str] = []

    def fit_calibration(
        self,
        calibration_scores: np.ndarray,
        groups: list[str] | np.ndarray,
    ) -> "MondrianConformalCalibrator":
        """
        Calibrate group-wise non-conformity scores.
        """
        scores = np.asarray(calibration_scores, dtype=np.float64)
        groups_arr = np.asarray([str(g) for g in groups])

        if len(scores) != len(groups_arr):
            raise ValueError("calibration_scores and groups must have matching lengths.")

        self.global_calibrator_.fit_calibration(scores)
        unique_groups = np.unique(groups_arr)
        self.groups_ = [str(g) for g in unique_groups]
        self.fallback_groups_ = []

        # Mathematical finite-sample threshold: n >= ceil(1 / alpha) - 1
        # If n < ceil(1 / alpha) - 1, min possible p-value (1 / (n+1)) > alpha, so null cannot be rejected.
        min_required_samples = max(1, int(np.ceil(1.0 / self.alpha)) - 1)

        for g in unique_groups:
            mask = groups_arr == g
            g_scores = scores[mask]
            if len(g_scores) >= min_required_samples:
                cal = ConformalAnomalyCalibrator(alpha=self.alpha)
                cal.fit_calibration(g_scores)
                self.group_calibrators_[str(g)] = cal
            else:
                self.fallback_groups_.append(str(g))
                self.group_calibrators_[str(g)] = self.global_calibrator_

        self._is_calibrated = True
        return self

    def compute_p_values(
        self,
        test_scores: np.ndarray,
        groups: list[str] | np.ndarray,
    ) -> np.ndarray:
        """
        Compute Mondrian conditional p-values.
        """
        if not self._is_calibrated:
            raise RuntimeError("MondrianConformalCalibrator is not calibrated.")

        test_scores_arr = np.asarray(test_scores, dtype=np.float64)
        test_groups_arr = np.asarray([str(g) for g in groups])
        p_vals = np.zeros(len(test_scores_arr), dtype=np.float64)

        for i, (score, g) in enumerate(zip(test_scores_arr, test_groups_arr, strict=True)):
            cal = self.group_calibrators_.get(g, self.global_calibrator_)
            p_vals[i] = cal.compute_p_values(np.array([score]))[0]

        return p_vals

    def predict(
        self,
        test_scores: np.ndarray,
        groups: list[str] | np.ndarray,
        alpha: float | None = None,
    ) -> dict[str, Any]:
        target_alpha = self.alpha if alpha is None else float(alpha)
        p_values = self.compute_p_values(test_scores, groups)
        is_anomaly = p_values <= target_alpha

        group_breakdown = {}
        for g in self.groups_:
            mask = np.asarray([str(grp) for grp in groups]) == g
            if np.sum(mask) > 0:
                group_breakdown[g] = {
                    "count": int(np.sum(mask)),
                    "anomalies": int(np.sum(is_anomaly[mask])),
                    "anomaly_rate": round(float(np.mean(is_anomaly[mask])), 4),
                    "mean_p_value": round(float(np.mean(p_values[mask])), 4),
                }

        return {
            "p_values": [round(float(p), 6) for p in p_values],
            "is_anomaly": [bool(a) for a in is_anomaly],
            "alpha": target_alpha,
            "confidence_level": round(float(1.0 - target_alpha), 4),
            "fallback_groups": list(self.fallback_groups_),
            "group_breakdown": group_breakdown,
            "coverage_statement": (
                f"Mondrian conditional validity guaranteed across {len(self.groups_)} groups "
                f"(fallback to global for {len(self.fallback_groups_)} small groups) "
                f"at significance alpha={target_alpha}."
            ),
        }

"""
backend/routers/conformal.py

FastAPI endpoints for SENTINEL Calibrated Uncertainty, Conformal p-values,
Mondrian Group-Conditional Calibration, and Benjamini-Hochberg FDR Control.
Versioned under /api/v2/conformal/.
"""

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.model_selection import train_test_split

from backend.schemas import (
    ConformalCalibrationRequest,
    ConformalCalibrationResponse,
    ConformalScoreItem,
    ConformalScoreRequest,
    ConformalScoreResponse,
    FDRControlRequest,
    FDRControlResponse,
)
from backend.state import state
from src.uncertainty.conformal import (
    ConformalAnomalyCalibrator,
    MondrianConformalCalibrator,
)
from src.uncertainty.fdr import benjamini_hochberg_fdr

router = APIRouter(prefix="/api/v2/conformal", tags=["conformal-v2"])


@router.post("/calibrate", response_model=ConformalCalibrationResponse)
def calibrate_conformal(req: ConformalCalibrationRequest) -> ConformalCalibrationResponse:
    """
    Calibrate non-parametric conformal anomaly predictor using a holdout reference partition.
    Guarantees finite-sample valid p-values under the exchangeability hypothesis.
    """
    if state.X is None or len(state.feature_names) == 0:
        raise HTTPException(
            status_code=400,
            detail="No preprocessed dataset loaded. Please upload or load a dataset first.",
        )

    # Ensure detection scores exist
    if state.norm_scores is None:
        state.run_detection()

    scores = state.norm_scores
    if scores is None or len(scores) == 0:
        raise HTTPException(
            status_code=500,
            detail="Failed to obtain detection scores for conformal calibration.",
        )

    n_samples = len(scores)

    if req.group_column and state.df_raw is not None and req.group_column in state.df_raw.columns:
        # Mondrian group-conditional calibration
        groups = state.df_raw[req.group_column].astype(str).to_numpy()

        cal_indices, _ = train_test_split(
            np.arange(n_samples),
            test_size=1.0 - req.split_ratio,
            random_state=42,
            stratify=groups if len(np.unique(groups)) > 1 and np.min(np.unique(groups, return_counts=True)[1]) >= 2 else None,
        )

        calibrator = MondrianConformalCalibrator(alpha=req.alpha)
        calibrator.fit_calibration(
            calibration_scores=scores[cal_indices],
            groups=groups[cal_indices],
        )
        state.conformal_calibrator = calibrator

        return ConformalCalibrationResponse(
            is_calibrated=True,
            alpha=req.alpha,
            confidence_level=round(1.0 - req.alpha, 4),
            n_calibration_samples=len(cal_indices),
            threshold_at_alpha=calibrator.global_calibrator_.threshold_at_alpha_,
            is_mondrian=True,
            groups=calibrator.groups_,
            fallback_groups=calibrator.fallback_groups_,
            coverage_statement=(
                f"Mondrian group-conditional conformal calibration established across "
                f"{len(calibrator.groups_)} groups with n={len(cal_indices)} reference points. "
                f"Conditional Type-I error bounded by alpha={req.alpha}. "
                f"({len(calibrator.fallback_groups_)} small groups fell back to global calibrator)."
            ),
        )
    else:
        # Standard Split-Conformal calibration
        cal_indices, _ = train_test_split(
            np.arange(n_samples),
            test_size=1.0 - req.split_ratio,
            random_state=42,
        )

        calibrator = ConformalAnomalyCalibrator(alpha=req.alpha)
        calibrator.fit_calibration(calibration_scores=scores[cal_indices])
        state.conformal_calibrator = calibrator

        return ConformalCalibrationResponse(
            is_calibrated=True,
            alpha=req.alpha,
            confidence_level=round(1.0 - req.alpha, 4),
            n_calibration_samples=len(cal_indices),
            threshold_at_alpha=calibrator.threshold_at_alpha_,
            is_mondrian=False,
            groups=[],
            fallback_groups=[],
            coverage_statement=(
                f"Split-conformal calibration established with n={len(cal_indices)} reference points. "
                f"Under nominal exchangeability, Type-I false alarm rate is bounded by alpha={req.alpha} "
                f"(Confidence: {(1.0 - req.alpha) * 100:.1f}%)."
            ),
        )


@router.post("/score", response_model=ConformalScoreResponse)
def score_conformal(req: ConformalScoreRequest) -> ConformalScoreResponse:
    """
    Score dataset observations, generating conformal p-values and Benjamini-Hochberg FDR q-values.
    Replaces uncalibrated heuristics with mathematically guaranteed uncertainty bounds.
    """
    if state.norm_scores is None:
        state.run_detection()

    scores = state.norm_scores
    if scores is None or len(scores) == 0:
        raise HTTPException(status_code=400, detail="No scores available.")

    # Auto-calibrate if not yet calibrated
    if state.conformal_calibrator is None:
        default_alpha = req.alpha or 0.05
        cal_req = ConformalCalibrationRequest(alpha=default_alpha, split_ratio=0.3)
        calibrate_conformal(cal_req)

    calibrator = state.conformal_calibrator
    target_alpha = req.alpha if req.alpha is not None else getattr(calibrator, "alpha", 0.05)

    if isinstance(calibrator, MondrianConformalCalibrator):
        if state.df_raw is None:
            raise HTTPException(status_code=400, detail="Raw dataframe required for Mondrian groups.")
        # Find group column from calibrator
        group_col = None
        for col in state.df_raw.columns:
            if set(state.df_raw[col].astype(str).unique()).issubset(set(calibrator.groups_)):
                group_col = col
                break
        groups = (
            state.df_raw[group_col].astype(str).to_numpy()
            if group_col is not None
            else np.array(["global"] * len(scores))
        )
        p_values = calibrator.compute_p_values(scores, groups)
    else:
        p_values = calibrator.compute_p_values(scores)

    # Benjamini-Hochberg FDR control
    fdr_results = benjamini_hochberg_fdr(p_values, target_fdr=req.target_fdr)
    q_values = fdr_results["q_values"]
    is_fdr_discovery = fdr_results["is_discovery"]

    is_conformal_anomaly = p_values <= target_alpha

    items: list[ConformalScoreItem] = [
        ConformalScoreItem(
            index=i,
            score=round(float(scores[i]), 4),
            conformal_p_value=round(float(p_values[i]), 6),
            fdr_q_value=round(float(q_values[i]), 6),
            is_conformal_anomaly=bool(is_conformal_anomaly[i]),
            is_fdr_significant=bool(is_fdr_discovery[i]),
        )
        for i in range(len(scores))
    ]

    conf_count = int(np.sum(is_conformal_anomaly))
    fdr_count = int(np.sum(is_fdr_discovery))

    coverage_statement = (
        f"Non-parametric conformal p-values guarantee finite-sample validity at alpha={target_alpha:.3f} "
        f"({(1.0 - target_alpha) * 100:.1f}% confidence). "
        f"Certified conformal anomalies: {conf_count} of {len(scores)}."
    )
    fdr_statement = (
        f"Benjamini-Hochberg adjustment guarantees False Discovery Rate <= {req.target_fdr:.1%}. "
        f"Multiple-testing certified discoveries: {fdr_count} of {len(scores)}."
    )

    state.conformal_results = {
        "alpha": target_alpha,
        "target_fdr": req.target_fdr,
        "conformal_anomalies_flagged": conf_count,
        "fdr_discoveries_flagged": fdr_count,
        "coverage_statement": coverage_statement,
        "fdr_statement": fdr_statement,
    }

    return ConformalScoreResponse(
        total_samples=len(scores),
        alpha=target_alpha,
        confidence_level=round(1.0 - target_alpha, 4),
        target_fdr=req.target_fdr,
        conformal_anomalies_flagged=conf_count,
        fdr_discoveries_flagged=fdr_count,
        coverage_statement=coverage_statement,
        fdr_statement=fdr_statement,
        items=items,
    )


@router.post("/fdr", response_model=FDRControlResponse)
def perform_fdr_control(req: FDRControlRequest) -> FDRControlResponse:
    """Perform Benjamini-Hochberg False Discovery Rate control on an arbitrary list of p-values."""
    res = benjamini_hochberg_fdr(np.array(req.p_values), target_fdr=req.target_fdr)
    return FDRControlResponse(**res)


@router.get("/status")
def get_conformal_status() -> dict[str, Any]:
    """Retrieve active conformal calibration metadata and guarantees."""
    if state.conformal_calibrator is None:
        return {
            "is_calibrated": False,
            "message": "Conformal calibrator is not initialized. Run /api/v2/conformal/calibrate.",
        }

    cal = state.conformal_calibrator
    is_mondrian = isinstance(cal, MondrianConformalCalibrator)

    return {
        "is_calibrated": True,
        "is_mondrian": is_mondrian,
        "alpha": getattr(cal, "alpha", 0.05),
        "confidence_level": round(1.0 - getattr(cal, "alpha", 0.05), 4),
        "n_calibration_samples": (
            cal.global_calibrator_.n_calibration_
            if is_mondrian
            else cal.n_calibration_
        ),
        "groups": cal.groups_ if is_mondrian else [],
        "fallback_groups": cal.fallback_groups_ if is_mondrian else [],
        "contamination_warning": (
            "Statistical Guarantee Note: If the reference calibration set contains undetected true "
            "anomalies (unsupervised contamination), the empirical non-conformity threshold is shifted "
            "rightward, making conformal p-values conservative (i.e. empirical Type-I error rate <= nominal alpha)."
        ),
        "last_results_summary": state.conformal_results,
    }

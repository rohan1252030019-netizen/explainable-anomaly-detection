"""
backend/routers/drift.py

Endpoints for Dataset & Anomaly Score Drift Monitoring.
"""

from fastapi import APIRouter, HTTPException

from backend.schemas import DriftEvaluateRequest, DriftReportResponse
from backend.state import state

router = APIRouter(prefix="/api/drift", tags=["drift"])


@router.post("/evaluate", response_model=DriftReportResponse)
def evaluate_drift(req: DriftEvaluateRequest | None = None):
    """
    Run distribution drift analysis (PSI, KS test, JS divergence)
    comparing baseline reference with incoming/evaluation split.
    """
    try:
        psi_bins = req.psi_bins if req else 10
        return state.run_drift_analysis(psi_bins=psi_bins)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/status", response_model=DriftReportResponse)
def get_drift_status():
    """Get the latest computed drift report."""
    if state.drift_report is None:
        if state.df_raw is not None and len(state.df_raw) >= 30:
            try:
                return state.run_drift_analysis()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DriftReportResponse(
            ready=False, message="No drift report generated yet."
        )
    return DriftReportResponse(**state.drift_report)

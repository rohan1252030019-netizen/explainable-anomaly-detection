"""
backend/routers/reports.py

Audit and demonstration report generation endpoints.
"""

from fastapi import APIRouter, HTTPException

from backend.schemas import AuditReportResponse
from backend.state import state

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/generate", response_model=AuditReportResponse)
def generate_report():
    """Generate structured audit and demonstration report."""
    try:
        return state.generate_report()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

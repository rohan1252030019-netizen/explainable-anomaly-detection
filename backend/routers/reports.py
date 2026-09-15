"""
backend/routers/reports.py

Audit and demonstration report generation endpoints.
"""

from fastapi import APIRouter, HTTPException
from backend.state import state

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/generate")
def generate_report():
    """Generate structured audit and demonstration report."""
    try:
        return state.generate_report()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

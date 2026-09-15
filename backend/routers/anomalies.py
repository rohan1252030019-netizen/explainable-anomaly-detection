"""
backend/routers/anomalies.py

Anomalies table query and detailed single-observation investigation endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from backend.state import state

router = APIRouter(prefix="/api/anomalies", tags=["anomalies"])


@router.get("")
def get_anomalies(
    search: Optional[str] = None,
    severity: Optional[List[str]] = Query(None),
    min_score: float = 0.0,
    max_score: float = 1.0,
    anomalies_only: bool = True,
    sort_by: str = "rank",
    ascending: bool = True,
    page: int = 1,
    page_size: int = 25,
):
    """Query, filter, and paginate anomalies."""
    try:
        return state.get_anomalies(
            search=search,
            severity_filter=severity,
            min_score=min_score,
            max_score=max_score,
            anomalies_only=anomalies_only,
            sort_by=sort_by,
            ascending=ascending,
            page=page,
            page_size=page_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{row_id}/investigate")
def investigate_anomaly(row_id: int):
    """Get deep-dive explanation, feature contributions, and observed-vs-normal stats."""
    try:
        return state.investigate_anomaly(row_id=row_id)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

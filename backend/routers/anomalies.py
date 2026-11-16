"""
backend/routers/anomalies.py

Anomalies table query and detailed single-observation investigation endpoints.
"""

from fastapi import APIRouter, HTTPException, Query

from backend.config import settings
from backend.schemas import AnomalyListResponse, InvestigationResponse
from backend.state import state

router = APIRouter(prefix="/api/anomalies", tags=["anomalies"])


@router.get("", response_model=AnomalyListResponse)
def get_anomalies(
    search: str | None = None,
    severity: list[str] | None = Query(None),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    max_score: float = Query(1.0, ge=0.0, le=1.0),
    anomalies_only: bool = True,
    sort_by: str = "rank",
    ascending: bool = True,
    page: int = Query(1, ge=1),
    page_size: int = Query(
        settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE
    ),
):
    """
    Query, filter, and paginate anomalies.

    ``page_size`` is bounded: without a ceiling a single request can ask for
    every row of a large dataset, force a full serialisation, and pin a worker.
    """
    if min_score > max_score:
        raise HTTPException(
            status_code=400, detail="min_score cannot exceed max_score."
        )
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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{row_id}/investigate", response_model=InvestigationResponse)
def investigate_anomaly(row_id: int):
    """Get deep-dive explanation, feature contributions, and observed-vs-normal stats."""
    try:
        return state.investigate_anomaly(row_id=row_id)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

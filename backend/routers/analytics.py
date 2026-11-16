"""
backend/routers/analytics.py

Visual analytics endpoints: PCA 2D projections and feature correlation matrices.
"""

from fastapi import APIRouter, HTTPException, Query

from backend.schemas import CorrelationsResponse, PcaAnalyticsResponse
from backend.state import state

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/pca", response_model=PcaAnalyticsResponse)
def get_pca_projection():
    """Get 2D PCA projection coordinates and variance explained."""
    try:
        return state.compute_pca_analytics()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/correlations", response_model=CorrelationsResponse)
def get_correlations(max_features: int = Query(12, ge=2, le=30)):
    """Get correlation matrix for numerical features."""
    try:
        return state.compute_correlations(max_features=max_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

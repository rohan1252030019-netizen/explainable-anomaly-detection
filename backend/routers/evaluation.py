"""
backend/routers/evaluation.py

Evaluation endpoints: out-of-sample (held-out) detection metrics.

Every metric the dashboard shows elsewhere is computed by fitting a detector
and scoring the same rows, which inflates the result. These endpoints report the
held-out number next to the in-sample one so the gap is visible instead of
implied.
"""

from fastapi import APIRouter, HTTPException, Query

from backend.schemas import HoldoutComparisonResponse, HoldoutMetrics
from backend.state import state

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.get("/holdout", response_model=HoldoutMetrics)
def get_holdout_metrics(
    model_name: str | None = Query(
        None, description="Detector to evaluate; defaults to the active model."
    ),
    test_size: float = Query(0.3, ge=0.1, le=0.5),
    random_state: int = Query(42),
):
    """
    Fit on a training split, score the held-out rows, and report both.

    Returns the in-sample and out-of-sample metrics plus ``optimism_gap`` (the
    difference). A large gap means the headline in-sample figure should not be
    quoted on its own.
    """
    try:
        return state.evaluate_holdout_metrics(
            model_name=model_name,
            test_size=test_size,
            random_state=random_state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/holdout/all", response_model=HoldoutComparisonResponse)
def get_holdout_comparison(
    test_size: float = Query(0.3, ge=0.1, le=0.5),
    random_state: int = Query(42),
):
    """Out-of-sample comparison table across every registered detector."""
    try:
        return state.evaluate_all_models_holdout(
            test_size=test_size, random_state=random_state
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

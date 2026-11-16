"""
backend/routers/experiments.py

Endpoints for Experiment Tracking and Reproducibility.
"""

from fastapi import APIRouter, HTTPException

from backend.schemas import ExperimentRecord
from backend.state import state

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/history", response_model=list[ExperimentRecord])
def get_experiment_history():
    """List reproducible experiment runs with dataset hashes, parameters, and results."""
    return state.get_experiments_history()


@router.get("/{experiment_id}", response_model=ExperimentRecord)
def get_experiment_detail(experiment_id: str):
    """Retrieve full configuration and metrics for a specific experiment run."""
    exp = state.experiment_tracker.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(
            status_code=404, detail=f"Experiment '{experiment_id}' not found."
        )
    return exp

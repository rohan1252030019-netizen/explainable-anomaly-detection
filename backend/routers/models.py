"""
backend/routers/models.py

Model Lab endpoints: multi-detector benchmarking, consensus analysis,
capability registry, and pairwise Jaccard similarity.
"""

from fastapi import APIRouter, HTTPException

from backend.schemas import (
    AvailableModelsResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    ModelCapability,
)
from backend.state import state

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("/benchmark", response_model=BenchmarkResponse)
def benchmark_all(req: BenchmarkRequest):
    """Run detectors side-by-side and return comparative metrics and consensus."""
    try:
        return state.benchmark_models(contamination=req.contamination)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/capabilities", response_model=dict[str, ModelCapability])
def get_capabilities():
    """Retrieve Model Capability Registry detailing complexity, assumptions, and explanation methods."""
    return state.model_manager.get_model_capabilities()


@router.get("/available", response_model=AvailableModelsResponse)
def get_available_models():
    """
    List all available anomaly detector algorithms.

    ``capabilities`` is what the dashboard reads.  ``default_params`` is kept
    alongside it because it shipped first and existing clients use it; the two
    are derived from the same registry, so they cannot disagree.
    """
    names = state.model_manager.get_available_models()
    return AvailableModelsResponse(
        models=names,
        capabilities=state.model_manager.get_model_capabilities(),
        default_params={
            name: state.model_manager.get_default_params(name) for name in names
        },
    )

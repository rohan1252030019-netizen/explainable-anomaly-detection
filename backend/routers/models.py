"""
backend/routers/models.py

Model Lab endpoints: multi-detector benchmarking, consensus analysis,
and pairwise Jaccard similarity.
"""

from fastapi import APIRouter, HTTPException
from backend.state import state
from backend.schemas import BenchmarkRequest

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("/benchmark")
def benchmark_all(req: BenchmarkRequest):
    """Run all 4 detectors side-by-side and return comparative metrics and consensus."""
    try:
        return state.benchmark_models(contamination=req.contamination)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

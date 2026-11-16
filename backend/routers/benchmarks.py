"""
backend/routers/benchmarks.py

Endpoints for Synthetic Anomaly Injection & Benchmark Lab.
"""

from fastapi import APIRouter, HTTPException

from backend.schemas import SyntheticBenchmarkRequest, SyntheticBenchmarkResponse
from backend.state import state

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


@router.post("/run", response_model=SyntheticBenchmarkResponse)
def run_benchmark(req: SyntheticBenchmarkRequest):
    """
    Inject synthetic anomalies (point, multivariate, cluster, contextual) into
    current features and benchmark detector sensitivity with ground-truth
    ROC/PR metrics.
    """
    try:
        return state.run_synthetic_benchmark(
            anomaly_type=req.anomaly_type,
            contamination=req.contamination,
            models=req.models_to_test,
            difficulty=req.difficulty,
            severity=req.severity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/status", response_model=SyntheticBenchmarkResponse)
def get_benchmark_status():
    """Retrieve the latest benchmark report."""
    if state.benchmark_report is None:
        return SyntheticBenchmarkResponse(
            ready=False, message="No benchmark run yet."
        )
    return state.benchmark_report

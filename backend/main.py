"""
backend/main.py

FastAPI backend application for the Explainable Anomaly Detection Framework.
"""

import os
import sys

from fastapi import FastAPI

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.middleware.cors import CORSMiddleware

from backend.config import ALLOWED_HEADERS, ALLOWED_METHODS, settings
from backend.routers import (
    analytics,
    anomalies,
    benchmarks,
    dataset,
    detection,
    drift,
    evaluation,
    experiments,
    explainability,
    export,
    models,
    reports,
)
from backend.schemas import HealthResponse
from backend.security import api_key_middleware
from backend.state import state
from src import __version__

app = FastAPI(
    title="XAUD API — Autonomous Anomaly Intelligence Platform",
    description="Production REST API for unsupervised anomaly detection, validation, explainability, drift, and benchmarks.",
    version=__version__,
)

# CORS for the Next.js frontend.  Origins are an allow-list (never "*") and so
# are methods and headers.  Credentials are disabled: the API authenticates
# with a key header, not with cookies, and `allow_credentials=True` combined
# with a reflected Origin is the classic CORS misconfiguration.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
    expose_headers=["Content-Disposition"],
)

# Optional shared-secret guard; open by default so local development needs no
# setup.  Set XAUD_API_KEY to turn it on.
app.middleware("http")(api_key_middleware)

# Register routers
app.include_router(dataset.router)
app.include_router(detection.router)
app.include_router(anomalies.router)
app.include_router(explainability.router)
app.include_router(analytics.router)
app.include_router(models.router)
app.include_router(reports.router)
app.include_router(export.router)
app.include_router(evaluation.router)
app.include_router(drift.router)
app.include_router(benchmarks.router)
app.include_router(experiments.router)


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Healthcheck endpoint returning framework and pipeline status."""
    return HealthResponse(
        status="healthy",
        framework="XAUD Autonomous Intelligence",
        version=__version__,
        dataset_loaded=state.df_raw is not None,
        dataset_name=state.dataset_name or None,
        detection_ready=state.detection_run,
        active_model=state.display_model_name,
        anomalies_detected=(
            int((state.predictions == -1).sum())
            if state.predictions is not None
            else 0
        ),
        preprocessed_features=len(state.feature_names),
        auth_required=settings.auth_required,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.RELOAD)

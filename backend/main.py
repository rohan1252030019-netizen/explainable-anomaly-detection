"""
backend/main.py

FastAPI backend application for the Explainable Anomaly Detection Framework.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.routers import (
    dataset,
    detection,
    anomalies,
    explainability,
    analytics,
    models,
    reports,
    export,
)
from backend.state import state

app = FastAPI(
    title="XAUD API — Explainable Anomaly Detection Framework",
    description="Production REST API for unsupervised anomaly detection and explainability.",
    version="2.0.0",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(dataset.router)
app.include_router(detection.router)
app.include_router(anomalies.router)
app.include_router(explainability.router)
app.include_router(analytics.router)
app.include_router(models.router)
app.include_router(reports.router)
app.include_router(export.router)


@app.get("/api/health")
def health_check():
    """Healthcheck endpoint returning framework and pipeline status."""
    return {
        "status": "healthy",
        "framework": "XAUD Unsupervised Intelligence",
        "version": "2.0.0",
        "dataset_loaded": state.df_raw is not None,
        "dataset_name": state.dataset_name,
        "detection_ready": state.detection_run,
        "active_model": state.active_model,
        "anomalies_detected": int((state.predictions == -1).sum()) if state.predictions is not None else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

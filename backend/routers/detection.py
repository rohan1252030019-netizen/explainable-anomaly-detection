"""
backend/routers/detection.py

Anomaly detection workflow endpoints: model registry, pipeline execution,
and overview analytics.
"""

from fastapi import APIRouter, HTTPException
from backend.state import state
from backend.schemas import DetectionRequest
from src.models.model_manager import DEFAULT_PARAMS

router = APIRouter(prefix="/api/detection", tags=["detection"])

MODEL_METADATA = {
    "Isolation Forest": {
        "icon": "Trees",
        "short": "IF",
        "tagline": "Tree partitioning. Fast and scalable.",
        "description": "Isolates anomalies by randomly selecting a feature and split value. Anomalies require fewer recursive splits to isolate.",
        "best_for": "Global outliers, high-dimensional data, tabular datasets",
        "complexity": "O(n log n)",
        "default_params": DEFAULT_PARAMS.get("Isolation Forest", {}),
    },
    "Local Outlier Factor": {
        "icon": "Target",
        "short": "LOF",
        "tagline": "Density-based. Detects local sparse clusters.",
        "description": "Compares local density of an observation against its k nearest neighbours. Low-density regions are flagged.",
        "best_for": "Local outliers, clusters with varying densities",
        "complexity": "O(n²)",
        "default_params": DEFAULT_PARAMS.get("Local Outlier Factor", {}),
    },
    "One-Class SVM": {
        "icon": "Zap",
        "short": "OCSVM",
        "tagline": "Kernel boundary. Non-linear hypersphere.",
        "description": "Learns a max-margin decision boundary enclosing the bulk of normal training points in feature space.",
        "best_for": "Non-linear distributions, novelty boundary detection",
        "complexity": "O(n²–n³)",
        "default_params": DEFAULT_PARAMS.get("One-Class SVM", {}),
    },
    "Elliptic Envelope": {
        "icon": "CircleDot",
        "short": "EE",
        "tagline": "Gaussian assumption. Robust Mahalanobis distance.",
        "description": "Fits a robust covariance estimate to determine Mahalanobis distances from the central cluster.",
        "best_for": "Normally distributed features, unimodal clusters",
        "complexity": "O(n · p²)",
        "default_params": DEFAULT_PARAMS.get("Elliptic Envelope", {}),
    },
}


@router.get("/models")
def get_available_models():
    """List available unsupervised detection models with metadata and default parameters."""
    return {"models": MODEL_METADATA}


@router.post("/run")
def run_detection(req: DetectionRequest):
    """Execute detection model and score observations."""
    try:
        overview = state.run_detection(
            model_name=req.model_name,
            params=req.params,
            threshold_method=req.threshold_method,
            contamination=req.contamination,
            quantile_val=req.quantile_val,
            fixed_val=req.fixed_val,
            use_ensemble=req.use_ensemble,
            ensemble_weights=req.ensemble_weights,
        )
        return {"success": True, "overview": overview}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/results")
def get_results():
    """Get high-level detection results, severity breakdown, and histograms."""
    overview = state.get_detection_overview()
    if not overview.get("ready"):
        return {"ready": False, "message": "No detection has been run yet."}
    return overview


@router.get("/experiments")
def get_experiments():
    """Get history of all detection runs in this session."""
    return {"experiments": state.experiment_history}

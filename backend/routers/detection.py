"""
backend/routers/detection.py

Anomaly detection workflow endpoints: model registry, pipeline execution,
intelligent ensemble consensus, and overview analytics.
"""

from fastapi import APIRouter, HTTPException

from backend.schemas import (
    DetectionExperimentsResponse,
    DetectionModelsResponse,
    DetectionOverview,
    DetectionRequest,
    DetectionRunResponse,
    EnsembleRunRequest,
    EnsembleRunResponse,
)
from backend.state import state
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
        "complexity": "O(n^2)",
        "default_params": DEFAULT_PARAMS.get("Local Outlier Factor", {}),
    },
    "One-Class SVM": {
        "icon": "Zap",
        "short": "OCSVM",
        "tagline": "Kernel boundary. Non-linear hypersphere.",
        "description": "Learns a max-margin decision boundary enclosing the bulk of normal training points in feature space.",
        "best_for": "Non-linear distributions, novelty boundary detection",
        "complexity": "O(n^2) to O(n^3)",
        "default_params": DEFAULT_PARAMS.get("One-Class SVM", {}),
    },
    "Elliptic Envelope": {
        "icon": "CircleDot",
        "short": "EE",
        "tagline": "Gaussian assumption. Robust Mahalanobis distance.",
        "description": "Fits a robust covariance estimate to determine Mahalanobis distances from the central cluster.",
        "best_for": "Normally distributed features, unimodal clusters",
        "complexity": "O(n * p^2)",
        "default_params": DEFAULT_PARAMS.get("Elliptic Envelope", {}),
    },
    "k-Nearest Neighbors": {
        "icon": "Network",
        "short": "KNN",
        "tagline": "Proximity-based. Geometric nearest neighbor distance.",
        "description": "Measures Euclidean distance to k-th nearest neighbor. Transparent distance interpretation without assumptions.",
        "best_for": "Multi-cluster geometric anomalies, proximity validation",
        "complexity": "O(n^2 * d)",
        "default_params": DEFAULT_PARAMS.get("k-Nearest Neighbors", {}),
    },
    "PCA Reconstruction": {
        "icon": "Activity",
        "short": "PCA",
        "tagline": "Subspace projection. Reconstruction error.",
        "description": "Projects data onto top principal components. Anomalies failing to align with principal axes incur high reconstruction error.",
        "best_for": "Linear subspace violations, high-dimensional correlation shifts",
        "complexity": "O(n * d * k)",
        "default_params": DEFAULT_PARAMS.get("PCA Reconstruction", {}),
    },
    "Histogram-Based Outlier Score": {
        "icon": "BarChart2",
        "short": "HBOS",
        "tagline": "Ultra-fast density. Linear time complexity.",
        "description": "Constructs univariate histograms per feature and sums inverse log densities. Ideal for high-throughput baseline screening.",
        "best_for": "Very large datasets, real-time screening, independent features",
        "complexity": "O(n * d)",
        "default_params": DEFAULT_PARAMS.get("Histogram-Based Outlier Score", {}),
    },
    "Gaussian Mixture Model": {
        "icon": "Layers",
        "short": "GMM",
        "tagline": "Multimodal density. Exact log-likelihood.",
        "description": "Fits K Gaussian distributions to unlabelled data. Observations situated in low probability tails are flagged.",
        "best_for": "Multimodal cluster datasets with probabilistic requirements",
        "complexity": "O(iter * K * n * d^2)",
        "default_params": DEFAULT_PARAMS.get("Gaussian Mixture Model", {}),
    },
}


@router.get("/models", response_model=DetectionModelsResponse)
def get_available_models():
    """List available unsupervised detection models with metadata and default parameters."""
    return DetectionModelsResponse(models=MODEL_METADATA)


@router.post("/run", response_model=DetectionRunResponse)
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
    except ValueError as exc:
        # A rejected configuration or an unavailable precondition; the caller
        # can fix it.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Detector training failure is reported as a 400 because the cause is
        # the request's model/parameters, but the chain is preserved so the
        # traceback still reaches the logs.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DetectionRunResponse(success=True, overview=DetectionOverview(**overview))


@router.post("/ensemble", response_model=EnsembleRunResponse)
def run_ensemble(req: EnsembleRunRequest):
    """Execute intelligent multi-detector ensemble with consensus aggregation."""
    try:
        res = state.run_intelligent_ensemble(
            models=req.models,
            aggregation_method=req.aggregation_method,
            weights=req.weights,
            min_consensus_votes=req.min_consensus_votes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EnsembleRunResponse(
        success=True,
        ensemble=res,
        overview=DetectionOverview(**state.get_detection_overview()),
    )


@router.get("/results", response_model=DetectionOverview)
def get_results():
    """Get high-level detection results, severity breakdown, and histograms."""
    overview = state.get_detection_overview()
    if not overview.get("ready"):
        return DetectionOverview(
            ready=False, message="No detection has been run yet."
        )
    return DetectionOverview(**overview)


@router.get("/experiments", response_model=DetectionExperimentsResponse)
def get_experiments():
    """Get history of all detection runs in this session."""
    return DetectionExperimentsResponse(experiments=state.experiment_history)

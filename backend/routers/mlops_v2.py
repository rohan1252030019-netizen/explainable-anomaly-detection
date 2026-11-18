"""
backend/routers/mlops_v2.py

API v2 router for SENTINEL MLOps & Production Hardening:
- Auto-generated Model Cards (Mitchell et al., 2019)
- Versioned Model Registry & Promotion
- Prometheus Metrics Exporter (/metrics)
- Presentation-ready HTML Audit Report Exporter
"""

import hashlib
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse, PlainTextResponse

from backend.schemas import (
    ModelCardGenerateRequest,
    ModelCardGenerateResponse,
    RegistryPromoteRequest,
    RegistryPromoteResponse,
    RegistrySaveRequest,
    RegistrySaveResponse,
)
from backend.state import state
from src.mlops.html_report import generate_html_audit_report
from src.mlops.model_card import generate_model_card
from src.mlops.monitoring import metrics_tracker
from src.mlops.registry import ModelRegistry

router = APIRouter(tags=["mlops-and-hardening"])

_registry = ModelRegistry()


@router.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    """Expose application telemetry in Prometheus exposition format."""
    # Update current counters
    if state.detection_run and state.predictions is not None:
        n_anom = int((state.predictions == -1).sum())
        metrics_tracker.anomalies_detected_count = n_anom
    return metrics_tracker.generate_prometheus_metrics()


@router.post("/api/v2/mlops/model-card/generate", response_model=ModelCardGenerateResponse)
def generate_run_model_card(req: ModelCardGenerateRequest) -> ModelCardGenerateResponse:
    """Generate structured Model Card and Markdown documentation for the active model."""
    if not state.detection_run or state.X is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Run detection before generating a model card.",
        )

    # Compute dataset hash
    data_bytes = state.X.tobytes()
    data_hash = hashlib.sha256(data_bytes).hexdigest()

    # Quantitative metrics
    metrics: dict[str, Any] = {
        "anomalies_detected": int((state.predictions == -1).sum()) if state.predictions is not None else 0,
        "anomaly_rate": float(np.mean(state.predictions == -1)) if state.predictions is not None else 0.0,
        "decision_threshold": float(state.threshold_val),
    }
    if hasattr(state, "holdout_report") and state.holdout_report:
        metrics["holdout_roc_auc"] = state.holdout_report.get("roc_auc")
        metrics["holdout_pr_auc"] = state.holdout_report.get("pr_auc")

    card = generate_model_card(
        model_name=state.display_model_name,
        version=req.version,
        dataset_name=state.dataset_name or "In-Memory Dataset",
        dataset_hash=data_hash,
        hyperparameters={"threshold": state.threshold_val, "contamination": state.contamination},
        feature_names=state.feature_names,
        metrics=metrics,
        author=req.author,
    )

    return ModelCardGenerateResponse(
        model_name=state.display_model_name,
        version=req.version,
        sha256_hash=card["sha256_hash"],
        model_card_data=card["model_card_data"],
        markdown=card["markdown"],
    )


@router.post("/api/v2/mlops/registry/save", response_model=RegistrySaveResponse)
def save_model_to_registry(req: RegistrySaveRequest) -> RegistrySaveResponse:
    """Persist active detector, metadata, and model card into versioned registry."""
    if not state.detection_run or state.X is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Run detection before registering a model.",
        )

    m_name = req.name or state.active_model
    detector = state.model_manager._detectors.get(state.active_model)

    # Generate card for registry
    data_hash = hashlib.sha256(state.X.tobytes()).hexdigest()
    card = generate_model_card(
        model_name=m_name,
        version=req.version,
        dataset_name=state.dataset_name or "In-Memory",
        dataset_hash=data_hash,
        hyperparameters={"threshold": state.threshold_val},
        feature_names=state.feature_names,
        metrics={"threshold": state.threshold_val},
    )

    meta = _registry.register_model(
        name=m_name,
        version=req.version,
        model_obj=detector,
        model_card=card,
        stage=req.stage,
        tags=req.tags,
    )

    return RegistrySaveResponse(success=True, metadata=meta)


@router.get("/api/v2/mlops/registry/list")
def list_registry_models():
    """List all registered models and their lifecycle stages."""
    return {"models": _registry.list_models()}


@router.post("/api/v2/mlops/registry/promote", response_model=RegistryPromoteResponse)
def promote_model_lifecycle(req: RegistryPromoteRequest) -> RegistryPromoteResponse:
    """Promote a registered model version to staging or production."""
    try:
        meta = _registry.promote_stage(name=req.name, version=req.version, new_stage=req.stage)
        return RegistryPromoteResponse(success=True, metadata=meta)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/v2/reports/html", response_class=HTMLResponse)
def export_html_report():
    """Export standalone, presentation-ready HTML Audit Report."""
    if not state.detection_run:
        raise HTTPException(status_code=400, detail="Run detection before generating report.")
    rep_data = state.generate_report()
    html_doc = generate_html_audit_report(rep_data)
    return HTMLResponse(content=html_doc)

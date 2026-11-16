"""
backend/routers/explainability.py

Explainability endpoints: global feature importance, perturbation & SHAP
attribution, and feature distribution separation.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.schemas import (
    ExplainabilityGenerateResponse,
    ExplanationRequest,
    FeatureDistributionResponse,
    GlobalImportanceResponse,
    StabilityResponse,
)
from backend.state import state

router = APIRouter(prefix="/api/explainability", tags=["explainability"])


@router.post("/generate", response_model=ExplainabilityGenerateResponse)
def generate_explanations(req: ExplanationRequest):
    """Compute local explanations for all anomalies and global feature importance."""
    try:
        return state.generate_all_explanations(use_shap=req.use_shap, top_k=req.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/global", response_model=GlobalImportanceResponse)
def get_global_importance():
    """Retrieve global feature importance ranking."""
    if state.global_importance is None:
        return GlobalImportanceResponse(
            ready=False, message="Explanations have not been generated yet."
        )

    records = state.global_importance.replace({np.nan: None}).to_dict(orient="records")
    return GlobalImportanceResponse(
        ready=True,
        n_explained=len(state.explanations),
        top_driver=records[0]["feature"] if records else None,
        global_importance=records,
    )


@router.get("/stability", response_model=StabilityResponse)
def get_explanation_stability(
    model_name: str | None = Query(
        None, description="Detector to probe; defaults to the active model."
    ),
):
    """
    Measure how much the feature ranking moves when the model is refitted
    under different seeds.

    An explanation that is only true for one random seed should not be acted
    on, and this endpoint is how a user finds that out before they do.
    """
    try:
        return state.explanation_stability(model_name=model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/feature-distributions", response_model=FeatureDistributionResponse)
def get_feature_distributions(
    feature: str = Query(..., min_length=1, max_length=200)
):
    """Compute histogram data for a specific feature comparing normal vs anomalous records."""
    if state.df_raw is None or state.predictions is None:
        raise HTTPException(status_code=400, detail="Dataset and detection required.")

    if feature not in state.df_raw.columns:
        raise HTTPException(
            status_code=404, detail=f"Feature '{feature}' not found in dataset."
        )

    if not pd.api.types.is_numeric_dtype(state.df_raw[feature]):
        raise HTTPException(
            status_code=400, detail=f"Feature '{feature}' is not numeric."
        )

    normal_vals = state.df_raw.loc[state.predictions == 1, feature].dropna().values
    anomaly_vals = state.df_raw.loc[state.predictions == -1, feature].dropna().values

    min_val = float(state.df_raw[feature].min())
    max_val = float(state.df_raw[feature].max())
    if min_val == max_val:
        max_val += 1.0

    bins = np.linspace(min_val, max_val, 25)
    norm_counts, _ = np.histogram(normal_vals, bins=bins)
    anom_counts, _ = np.histogram(anomaly_vals, bins=bins)

    dist_data = []
    for i in range(len(bins) - 1):
        dist_data.append({
            "bin_start": round(float(bins[i]), 3),
            "bin_end": round(float(bins[i + 1]), 3),
            "normal_count": int(norm_counts[i]),
            "anomaly_count": int(anom_counts[i]),
        })

    norm_median = float(np.median(normal_vals)) if len(normal_vals) > 0 else 0.0
    anom_median = float(np.median(anomaly_vals)) if len(anomaly_vals) > 0 else 0.0

    return FeatureDistributionResponse(
        feature=feature,
        normal_median=round(norm_median, 3),
        anomaly_median=round(anom_median, 3),
        separation=round(abs(anom_median - norm_median), 3),
        distribution=dist_data,
    )

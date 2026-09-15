"""
backend/routers/explainability.py

Explainability endpoints: global feature importance, perturbation & SHAP attribution,
and feature distribution separation.
"""

from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
from backend.state import state
from backend.schemas import ExplanationRequest

router = APIRouter(prefix="/api/explainability", tags=["explainability"])


@router.post("/generate")
def generate_explanations(req: ExplanationRequest):
    """Compute local explanations for all anomalies and global feature importance."""
    try:
        return state.generate_all_explanations(use_shap=req.use_shap, top_k=req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/global")
def get_global_importance():
    """Retrieve global feature importance ranking."""
    if state.global_importance is None:
        return {"ready": False, "message": "Explanations have not been generated yet."}
    
    records = state.global_importance.replace({np.nan: None}).to_dict(orient="records")
    return {
        "ready": True,
        "n_explained": len(state.explanations),
        "top_driver": records[0]["feature"] if records else None,
        "global_importance": records,
    }


@router.get("/feature-distributions")
def get_feature_distributions(feature: str = Query(...)):
    """Compute histogram data for a specific feature comparing normal vs anomalous records."""
    if state.df_raw is None or state.predictions is None:
        raise HTTPException(status_code=400, detail="Dataset and detection required.")

    if feature not in state.df_raw.columns:
        raise HTTPException(status_code=404, detail=f"Feature '{feature}' not found in dataset.")

    if not pd.api.types.is_numeric_dtype(state.df_raw[feature]):
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' is not numeric.")

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

    return {
        "feature": feature,
        "normal_median": round(norm_median, 3),
        "anomaly_median": round(anom_median, 3),
        "separation": round(abs(anom_median - norm_median), 3),
        "distribution": dist_data,
    }

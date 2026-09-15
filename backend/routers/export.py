"""
backend/routers/export.py

Export endpoints for downloading CSV predictions, JSON explanations,
and Markdown reports.
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from backend.state import state

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/predictions")
def export_predictions():
    """Download full predictions CSV."""
    if state.df_results is None:
        raise HTTPException(status_code=400, detail="No detection results available.")
    
    csv_bytes = state.df_results.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=xaud_predictions.csv"},
    )


@router.get("/anomalies")
def export_anomalies():
    """Download anomalies-only CSV."""
    if state.df_results is None:
        raise HTTPException(status_code=400, detail="No detection results available.")

    anom_df = state.df_results[state.df_results["is_anomaly"]].sort_values("rank")
    csv_bytes = anom_df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=xaud_anomalies_only.csv"},
    )


@router.get("/explanations")
def export_explanations():
    """Download explanations JSON."""
    if not state.explanations:
        raise HTTPException(status_code=400, detail="No explanations generated.")

    clean_exps = []
    for exp in state.explanations:
        item = {
            "row_id": exp.get("row_id"),
            "model_name": exp.get("model_name"),
            "normalized_score": exp.get("norm_score"),
            "severity": exp.get("severity"),
            "explanation_text": exp.get("explanation_text"),
            "confidence_note": exp.get("confidence_note"),
            "top_features": exp.get("top_features", []),
        }
        clean_exps.append(item)

    json_bytes = json.dumps(clean_exps, indent=2).encode("utf-8")
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=xaud_explanations.json"},
    )

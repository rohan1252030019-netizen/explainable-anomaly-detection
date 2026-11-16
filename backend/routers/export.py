"""
backend/routers/export.py

Export endpoints for downloading CSV predictions, JSON explanations,
and Markdown reports.

CSV cells are sanitised before serialisation.  A spreadsheet treats a cell
whose text begins with ``=``, ``+``, ``-``, ``@``, a tab, or a carriage return
as a formula, so a dataset value such as ``=HYPERLINK("http://evil", ...)``
executes when the exported file is opened.  The export is the one place raw
user-controlled strings cross into another interpreter.
"""

import json

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.state import state

router = APIRouter(prefix="/api/export", tags=["export"])

# Leading characters Excel / LibreOffice / Sheets interpret as a formula.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r", "\n")


def _sanitize_cell(value):
    """
    Neutralise a spreadsheet formula without corrupting ordinary data.

    Numeric dtypes are never visited: pandas writes numbers unquoted, so
    ``-5.2`` reaches the file as a number, not as the string ``"-5.2"``.  Any
    *string* that begins with a trigger character is, by definition, text the
    spreadsheet would try to evaluate.
    """
    if not isinstance(value, str):
        return value
    if not value:
        return value
    if value.startswith(_FORMULA_TRIGGERS):
        # A leading apostrophe forces text interpretation; the visible value is
        # unchanged in the spreadsheet.
        return "'" + value
    return value


def sanitize_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with formula-like string cells neutralised."""
    out = df.copy()
    for column in out.columns:
        series = out[column]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            out[column] = series.map(_sanitize_cell)
    return out


@router.get("/predictions")
def export_predictions():
    """Download full predictions CSV."""
    if state.df_results is None:
        raise HTTPException(status_code=400, detail="No detection results available.")

    csv_bytes = sanitize_csv_frame(state.df_results).to_csv(index=False).encode("utf-8")
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
    csv_bytes = sanitize_csv_frame(anom_df).to_csv(index=False).encode("utf-8")
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

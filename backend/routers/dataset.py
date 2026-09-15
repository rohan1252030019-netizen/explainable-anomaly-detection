"""
backend/routers/dataset.py

Dataset management endpoints: sample dataset loading, CSV upload,
schema profiling, and preprocessing configuration.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.state import state
from backend.schemas import PreprocessRequest

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/sample")
def load_sample():
    """Load or generate the benchmark sample dataset."""
    try:
        summary = state.load_sample_dataset()
        return {"success": True, "dataset": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a custom CSV dataset."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        summary = state.load_user_csv(contents, file.filename)
        return {"success": True, "dataset": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {exc}")


@router.get("/summary")
def get_summary():
    """Get current dataset profiling and column information."""
    return state.get_dataset_summary()


@router.post("/preprocess")
def preprocess(req: PreprocessRequest):
    """Reconfigure and run preprocessing pipeline."""
    try:
        result = state.preprocess_dataset(
            scaler_type=req.scaler_type,
            label_col=req.label_col,
            id_cols=req.id_cols,
        )
        return {"success": True, "preprocessing": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/reset")
def reset_dataset():
    """Reset dataset and all analysis results."""
    state.reset_all()
    return {"success": True, "message": "State reset successfully."}

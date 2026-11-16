"""
backend/routers/dataset.py

Dataset management endpoints: sample dataset loading, CSV upload,
schema profiling, and preprocessing configuration.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.config import settings
from backend.schemas import (
    DatasetLoadedResponse,
    DatasetProfileResponse,
    DatasetSummary,
    PreprocessRequest,
    PreprocessResponse,
    SimpleSuccessResponse,
)
from backend.state import state

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/sample", response_model=DatasetLoadedResponse)
def load_sample():
    """Load or generate the benchmark sample dataset."""
    try:
        summary = state.load_sample_dataset()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DatasetLoadedResponse(success=True, dataset=summary)


@router.post("/upload", response_model=DatasetLoadedResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a custom CSV dataset.

    Size and row limits are enforced before parsing. Reading an unbounded
    upload into memory is the cheapest way to take the process down, so the
    byte budget is checked first and the row budget afterwards.

    Parsing and preprocessing are CPU-bound.  This handler is ``async`` so the
    body can be streamed, which means a direct call to ``state.load_user_csv``
    would block the event loop for the whole parse and freeze every other
    request.  It is dispatched to the threadpool instead.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    limit = settings.MAX_UPLOAD_BYTES
    declared = file.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File is {int(declared) / 1e6:.1f} MB, above the "
                        f"{limit / 1e6:.0f} MB limit."
                    ),
                )
        except ValueError:
            pass

    contents = await file.read(limit + 1)
    if len(contents) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {limit / 1e6:.0f} MB limit.",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        summary = await run_in_threadpool(
            state.load_user_csv, contents, file.filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MemoryError:
        raise HTTPException(
            status_code=413, detail="File too large to parse."
        ) from None

    if summary.get("n_rows", 0) > settings.MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Dataset has {summary['n_rows']} rows, above the "
                f"{settings.MAX_ROWS} row limit."
            ),
        )

    return DatasetLoadedResponse(success=True, dataset=DatasetSummary(**summary))


@router.get("/summary", response_model=DatasetSummary)
def get_summary():
    """Get current dataset profiling and column information."""
    return DatasetSummary(**state.get_dataset_summary())


@router.get("/profile", response_model=DatasetProfileResponse)
def get_profile():
    """Get deep data intelligence profiling and health score."""
    try:
        return DatasetProfileResponse(**state.get_profiler_report())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess(req: PreprocessRequest):
    """Reconfigure and run preprocessing pipeline."""
    try:
        result = state.preprocess_dataset(
            scaler_type=req.scaler_type,
            label_col=req.label_col,
            id_cols=req.id_cols,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PreprocessResponse(success=True, preprocessing=result)


@router.post("/reset", response_model=SimpleSuccessResponse)
def reset_dataset():
    """Reset dataset and all analysis results."""
    state.reset_all()
    return SimpleSuccessResponse(success=True, message="State reset successfully.")

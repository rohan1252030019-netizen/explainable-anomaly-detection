"""
backend/schemas.py

Pydantic schemas for request and response validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PreprocessRequest(BaseModel):
    scaler_type: str = Field(default="standard", description="standard or robust")
    label_col: Optional[str] = Field(default=None, description="Optional ground truth column for evaluation only")
    id_cols: Optional[List[str]] = Field(default=[], description="Columns to exclude from training")


class DetectionRequest(BaseModel):
    model_name: str = Field(default="Isolation Forest")
    params: Optional[Dict[str, Any]] = Field(default={})
    threshold_method: str = Field(default="contamination", description="contamination, quantile, or fixed")
    contamination: float = Field(default=0.05, ge=0.001, le=0.5)
    quantile_val: float = Field(default=0.95, ge=0.5, le=0.999)
    fixed_val: float = Field(default=0.5, ge=0.0, le=1.0)
    use_ensemble: bool = Field(default=False)
    ensemble_weights: Optional[Dict[str, float]] = Field(default=None)


class ExplanationRequest(BaseModel):
    use_shap: bool = Field(default=False)
    top_k: int = Field(default=8, ge=3, le=20)


class BenchmarkRequest(BaseModel):
    contamination: float = Field(default=0.05, ge=0.001, le=0.5)

"""
src/models/supervised/__init__.py

Supervised, surrogate, and stacking anomaly detection models for SENTINEL.
"""

from src.models.supervised.base import BaseSupervisedDetector
from src.models.supervised.classifier import (
    SupervisedClassifierDetector,
    compute_expected_calibration_error,
    optimize_threshold_cost_matrix,
)
from src.models.supervised.stacking import HybridStackingDetector
from src.models.supervised.surrogate import FidelityReport, SurrogateDetector
from src.models.supervised.tuning import tune_xgboost_hyperparameters

__all__ = [
    "BaseSupervisedDetector",
    "SurrogateDetector",
    "SupervisedClassifierDetector",
    "HybridStackingDetector",
    "FidelityReport",
    "tune_xgboost_hyperparameters",
    "compute_expected_calibration_error",
    "optimize_threshold_cost_matrix",
]

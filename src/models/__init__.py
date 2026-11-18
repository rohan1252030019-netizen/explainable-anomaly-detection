"""
src/models/__init__.py

Detector implementations and model management for SENTINEL.
"""

from src.models.base_detector import BaseDetector
from src.models.elliptic_envelope import EllipticEnvelopeDetector
from src.models.gmm_detector import GMMDetector
from src.models.hbos_detector import HBOSDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.knn_detector import KNNDetector
from src.models.local_outlier_factor import LOFDetector
from src.models.model_manager import (
    DEFAULT_PARAMS,
    DETECTOR_REGISTRY,
    MODEL_CAPABILITIES,
    ModelManager,
)
from src.models.one_class_svm import OneClassSVMDetector
from src.models.pca_detector import PCADetector

__all__ = [
    "BaseDetector",
    "IsolationForestDetector",
    "LOFDetector",
    "OneClassSVMDetector",
    "EllipticEnvelopeDetector",
    "KNNDetector",
    "PCADetector",
    "HBOSDetector",
    "GMMDetector",
    "ModelManager",
    "DETECTOR_REGISTRY",
    "DEFAULT_PARAMS",
    "MODEL_CAPABILITIES",
]

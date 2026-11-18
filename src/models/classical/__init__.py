"""
src/models/classical/__init__.py

Advanced Classical Anomaly Detectors for SENTINEL.
"""

from src.models.classical.copod_detector import COPODDetector
from src.models.classical.dbscan_detector import DBSCANDetector
from src.models.classical.ecod_detector import ECODDetector
from src.models.classical.extended_isolation_forest import (
    ExtendedIsolationForestDetector,
)
from src.models.classical.mahalanobis_detector import MahalanobisDetector
from src.models.classical.suod_ensemble import SUODDetector

__all__ = [
    "ECODDetector",
    "COPODDetector",
    "MahalanobisDetector",
    "DBSCANDetector",
    "ExtendedIsolationForestDetector",
    "SUODDetector",
]

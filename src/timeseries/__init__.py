"""
src/timeseries package

SENTINEL Time-Series & Streaming Anomaly Intelligence:
- STL Seasonal-Trend Decomposition Residual Anomaly Detection
- Strictly Causal Rolling Window Feature Engineering (Zero Lookahead Leakage)
- Change-Point Detection (PELT & BOCPD)
- Streaming Half-Space Trees (HS-Trees) Incremental Detector
- Sliding-Window Population Stability Index (PSI) Drift Monitor & Recalibration Policy
"""

from src.timeseries.changepoint import detect_changepoints_bocpd, detect_changepoints_pelt
from src.timeseries.rolling_features import (
    generate_causal_rolling_features,
    verify_temporal_causality,
)
from src.timeseries.stl_detector import STLAnomalyDetector
from src.timeseries.streaming_detector import (
    StreamingDriftMonitor,
    StreamingHalfSpaceTrees,
)

__all__ = [
    "STLAnomalyDetector",
    "generate_causal_rolling_features",
    "verify_temporal_causality",
    "detect_changepoints_pelt",
    "detect_changepoints_bocpd",
    "StreamingHalfSpaceTrees",
    "StreamingDriftMonitor",
]

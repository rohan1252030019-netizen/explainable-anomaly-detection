"""
src/mlops package

SENTINEL MLOps & Production Hardening Engine:
- Model Cards Generator (Mitchell et al., 2019)
- Versioned Model Registry & Lifecycle Promotion
- Prometheus Telemetry & Metrics Exporter
- Token Bucket Rate Limiting
- Standalone HTML Audit Report Generator
"""

from src.mlops.html_report import generate_html_audit_report
from src.mlops.model_card import generate_model_card
from src.mlops.monitoring import (
    MetricsTracker,
    RateLimiter,
    metrics_tracker,
    rate_limiter,
)
from src.mlops.registry import ModelRegistry

__all__ = [
    "generate_model_card",
    "ModelRegistry",
    "MetricsTracker",
    "RateLimiter",
    "metrics_tracker",
    "rate_limiter",
    "generate_html_audit_report",
]

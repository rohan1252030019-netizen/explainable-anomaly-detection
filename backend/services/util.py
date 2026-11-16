"""
backend/services/util.py

Shared helpers for the state service mixins.
"""

import inspect
import os
from typing import Any

from src.models.model_manager import DETECTOR_REGISTRY

# Repository root.  Kept here because several services resolve on-disk
# artefacts (the bundled sample dataset, the model and experiment folders)
# relative to it.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _round_or_none(value: Any, digits: int = 4) -> Any:
    """Round a numeric metric, passing through None / non-numerics."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), digits)
    return None


def _accepts_random_state(model_name: str) -> bool:
    """Whether a registry detector's constructor accepts ``random_state``."""
    cls = DETECTOR_REGISTRY.get(model_name)
    if cls is None:
        return False
    try:
        return "random_state" in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return False


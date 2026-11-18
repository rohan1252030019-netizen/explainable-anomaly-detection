"""SENTINEL — Autonomous Anomaly Intelligence Platform.

Compatibility shim exposing SENTINEL package namespace.
"""

import sys
import src

__version__ = src.__version__
__all__ = ["__version__"]

# Expose subpackages for direct imports e.g. sentinel.models
sys.modules["sentinel"] = sys.modules[__name__]
sys.modules["sentinel.models"] = src.models
sys.modules["sentinel.data"] = src.data
sys.modules["sentinel.detection"] = src.detection
sys.modules["sentinel.evaluation"] = src.evaluation
sys.modules["sentinel.explainability"] = src.explainability

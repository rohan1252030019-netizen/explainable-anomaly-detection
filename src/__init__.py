"""SENTINEL — Autonomous Anomaly Intelligence Platform.

Scientific, Explainable, Next-gen Threat & Irregularity Neutralization and Evaluation Layer.
"Detect. Explain. Verify. Act."

The version is defined here only. ``pyproject.toml`` reads it through
``[tool.setuptools.dynamic] version = { attr = "src.__version__" }`` and
``backend/main.py`` imports it, so package metadata, wheel filename,
the ``/api/health`` payload and the FastAPI ``version`` field remain unified.
"""

__version__ = "3.0.0"

__all__ = ["__version__"]

"""Explainable Anomaly Detection Framework (XAUD).

The version is defined here only.  ``pyproject.toml`` reads it through
``[tool.setuptools.dynamic] version = { attr = "src.__version__" }`` and
``backend/main.py`` imports it, so the package metadata, the wheel filename, the
``/api/health`` payload and the FastAPI ``version`` field cannot drift apart.
"""

__version__ = "2.1.0"

__all__ = ["__version__"]

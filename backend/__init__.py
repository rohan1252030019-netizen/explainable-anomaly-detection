"""
backend — FastAPI service layer for the Explainable Anomaly Detection framework.

This package is intentionally a regular (non-namespace) package: without this
file, ``backend/`` is a PEP 420 namespace package, so mypy and setuptools see
``backend/state.py`` under two different module names ("state" and
"backend.state").  That produced

    Source file found twice under different module names

and caused the wheel to ship almost none of the application.
"""

__all__ = ["config", "experiments", "main", "schemas", "state"]

"""
backend/config.py

Environment-driven configuration for the API.

Every value is read from an environment variable with a safe local default, so
the project runs with no setup while remaining configurable for deployment. No
secrets are stored here; see .env.example for the full list.
"""

import os


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_origins(raw: str, default: list[str]) -> list[str]:
    if not raw or not raw.strip():
        return default
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or default


DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3100",
    "http://127.0.0.1:3100",
    "http://localhost:3111",
    "http://127.0.0.1:3111",
]

# CORS is an allow-list in both directions.  ``allow_headers=["*"]`` reflects
# whatever the browser asks for, which turns any typo'd header into an accepted
# one; these are the only headers this API actually reads.
ALLOWED_METHODS = ["GET", "POST", "DELETE", "OPTIONS"]
ALLOWED_HEADERS = ["Accept", "Content-Type", "X-API-Key", "Authorization"]


class Settings:
    """
    Environment-driven settings, read when the object is constructed.

    These were class attributes evaluated once at import time, which made the
    class impossible to exercise without reloading the whole module and meant a
    test could not ask "what would the server do with XAUD_PORT=9123?".  Reading
    in ``__init__`` gives each instance its own snapshot; the module-level
    ``settings`` is still created once, so runtime behaviour is unchanged.
    """

    def __init__(self) -> None:
        self.HOST: str = os.environ.get("XAUD_HOST", "127.0.0.1")
        self.PORT: int = _get_int("XAUD_PORT", 8001)
        self.RELOAD: bool = _get_bool("XAUD_RELOAD", True)

        # Shared-secret API key.  Empty means "open", which is the local default.
        # Set XAUD_API_KEY to require X-API-Key (or a Bearer token) on every
        # /api route except the health probe.
        self.API_KEY: str = os.environ.get("XAUD_API_KEY", "").strip()

        # Comma-separated allow-list. Defaults to the local Next.js dev ports.
        self.CORS_ORIGINS: list[str] = _get_origins(
            os.environ.get("XAUD_CORS_ORIGINS", ""), DEFAULT_ORIGINS
        )

        # Upload guard rails. An unbounded CSV read is a cheap way to exhaust RAM.
        self.MAX_UPLOAD_BYTES: int = _get_int(
            "XAUD_MAX_UPLOAD_BYTES", 25 * 1024 * 1024
        )
        self.MAX_ROWS: int = _get_int("XAUD_MAX_ROWS", 200_000)

        # Pagination guard rail. An unbounded page_size lets one request
        # enumerate (and serialise) every row of a large dataset.
        self.DEFAULT_PAGE_SIZE: int = _get_int("XAUD_DEFAULT_PAGE_SIZE", 25)
        self.MAX_PAGE_SIZE: int = _get_int("XAUD_MAX_PAGE_SIZE", 200)

        # Explainer budgets. SHAP's model-agnostic explainers grow with the
        # background size and the row count, so both are capped.
        self.SHAP_MAX_BACKGROUND: int = _get_int("XAUD_SHAP_MAX_BACKGROUND", 100)
        self.MAX_EXPLANATIONS: int = _get_int("XAUD_MAX_EXPLANATIONS", 200)

    @property
    def auth_required(self) -> bool:
        return bool(self.API_KEY)


settings = Settings()

"""
backend/config.py

Environment-driven configuration for the SENTINEL API.

Every value is read from an environment variable with a safe local default, so
the project runs with no setup while remaining configurable for deployment. No
secrets are stored here; see .env.example for the full list.

SENTINEL_* is the primary environment variable namespace.
XAUD_* variables are supported as backward-compatible fallbacks with deprecation warnings.
"""

import os
import warnings


def _get_env(name: str, fallback_name: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is not None and val.strip():
        return val.strip()
    if fallback_name:
        old_val = os.environ.get(fallback_name)
        if old_val is not None and old_val.strip():
            warnings.warn(
                f"Environment variable '{fallback_name}' is deprecated and will be removed "
                f"in a future release. Use '{name}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return old_val.strip()
    return None


def _get_int(name: str, default: int, fallback_name: str | None = None) -> int:
    raw = _get_env(name, fallback_name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool, fallback_name: str | None = None) -> bool:
    raw = _get_env(name, fallback_name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_origins(raw: str | None, default: list[str]) -> list[str]:
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

    Reading in ``__init__`` gives each instance its own snapshot; the module-level
    ``settings`` is created once, so runtime behaviour is unchanged.
    """

    def __init__(self) -> None:
        self.HOST: str = _get_env("SENTINEL_HOST", "XAUD_HOST") or "127.0.0.1"
        self.PORT: int = _get_int("SENTINEL_PORT", 8000, "XAUD_PORT")
        self.RELOAD: bool = _get_bool("SENTINEL_RELOAD", True, "XAUD_RELOAD")

        # Shared-secret API key. Empty means "open", which is the local default.
        # Set SENTINEL_API_KEY (or legacy XAUD_API_KEY) to require authentication.
        self.API_KEY: str = _get_env("SENTINEL_API_KEY", "XAUD_API_KEY") or ""

        # Comma-separated allow-list. Defaults to the local Next.js dev ports.
        self.CORS_ORIGINS: list[str] = _get_origins(
            _get_env("SENTINEL_CORS_ORIGINS", "XAUD_CORS_ORIGINS"), DEFAULT_ORIGINS
        )

        # Upload guard rails.
        self.MAX_UPLOAD_BYTES: int = _get_int(
            "SENTINEL_MAX_UPLOAD_BYTES", 25 * 1024 * 1024, "XAUD_MAX_UPLOAD_BYTES"
        )
        self.MAX_ROWS: int = _get_int("SENTINEL_MAX_ROWS", 200_000, "XAUD_MAX_ROWS")

        # Pagination guard rails.
        self.DEFAULT_PAGE_SIZE: int = _get_int(
            "SENTINEL_DEFAULT_PAGE_SIZE", 25, "XAUD_DEFAULT_PAGE_SIZE"
        )
        self.MAX_PAGE_SIZE: int = _get_int(
            "SENTINEL_MAX_PAGE_SIZE", 200, "XAUD_MAX_PAGE_SIZE"
        )

        # Explainer budgets.
        self.SHAP_MAX_BACKGROUND: int = _get_int(
            "SENTINEL_SHAP_MAX_BACKGROUND", 100, "XAUD_SHAP_MAX_BACKGROUND"
        )
        self.MAX_EXPLANATIONS: int = _get_int(
            "SENTINEL_MAX_EXPLANATIONS", 200, "XAUD_MAX_EXPLANATIONS"
        )

    @property
    def auth_required(self) -> bool:
        return bool(self.API_KEY)


settings = Settings()

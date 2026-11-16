"""
backend/security.py

Optional API-key authentication for the XAUD API.

The audit found every endpoint — including destructive ones such as
``/api/dataset/reset`` and expensive ones such as ``/api/models/benchmark`` —
open to any caller that could reach the port.  Adding authentication without
breaking existing local deployments is the constraint, so:

* If ``XAUD_API_KEY`` is unset (the default), the API stays open and
  ``/api/health`` reports ``auth_required: false``.  Local development and the
  bundled frontend keep working with no setup.
* If it is set, every ``/api`` route except the health probe requires the key in
  ``X-API-Key`` (or ``Authorization: Bearer <key>``).  Comparison is constant
  time.

This is deliberately a shared-secret guard, not a user system: it stops an
exposed port from being a free compute and data-exfiltration service.  A
multi-tenant deployment should put a real identity provider in front of it.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.config import settings

# Liveness/readiness and the interactive docs stay public: a probe cannot carry
# a secret, and the docs page is useless without the key for the API calls it
# documents.
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/api/health",
        "/docs",
        "/docs/oauth2-redirect",
        "/openapi.json",
        "/redoc",
    }
)


def extract_key(request: Request) -> str | None:
    """Read the presented key from ``X-API-Key`` or a Bearer token."""
    header = request.headers.get("x-api-key")
    if header:
        return header.strip()
    authorization = request.headers.get("authorization", "")
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() == "bearer" and credential:
        return credential.strip()
    return None


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or not path.startswith("/api")


def key_is_valid(presented: str | None) -> bool:
    expected = settings.API_KEY
    if not expected:
        return True
    if not presented:
        return False
    return hmac.compare_digest(presented, expected)


async def api_key_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[object]]
):
    """
    Reject unauthenticated ``/api`` requests when a key is configured.

    Implemented as middleware rather than a per-router dependency so a route
    added later cannot forget to opt in.
    """
    if not is_public(request.url.path) and not key_is_valid(extract_key(request)):
        return JSONResponse(
            status_code=401,
            content={"detail": "A valid API key is required (X-API-Key header)."},
            headers={"WWW-Authenticate": 'Bearer realm="xaud"'},
        )
    return await call_next(request)

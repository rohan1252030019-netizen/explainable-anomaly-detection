"""
backend/session.py

Opt-in session isolation for the XAUD API.

The API serves one analysis session per process by default: ``backend.state.state``
is shared by every request, which is exactly what the bundled single-user
dashboard expects.  That is a deployment limitation, not an accidental one, so it
is stated rather than hidden.

``SessionRegistry`` provides the isolation primitive a multi-user or multi-tenant
deployment needs: each session id gets its own fully independent ``AppState``
(dataset, preprocessing, fitted models, results, explanation cache).  Sessions are
created lazily, bounded in number, and evicted least-recently-used so a caller
cannot exhaust memory by inventing session ids.

Nothing in the routers selects a session yet; wiring a header to ``get()`` is a
deliberate deployment decision, not something a library should do silently.  The
registry is therefore additive: existing code paths are unchanged.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

from backend.state import AppState


class SessionRegistry:
    """Bounded, thread-safe map of ``session_id -> AppState``."""

    DEFAULT_SESSION = "default"

    def __init__(self, max_sessions: int = 8) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1.")
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[str, AppState] = OrderedDict()
        self._lock = threading.RLock()
        self._evicted = 0

    def get(self, session_id: str = DEFAULT_SESSION) -> AppState:
        """
        Return the session's state, creating it on first use.

        The most recently used entry moves to the end of the order; when the
        registry is full the least recently used session is dropped.
        """
        if not session_id:
            raise ValueError("session_id must be a non-empty string.")
        with self._lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                return self._sessions[session_id]

            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
                self._evicted += 1

            session = AppState()
            self._sessions[session_id] = session
            return session

    def create(self) -> str:
        """Allocate a new session id and materialise its (empty) state."""
        session_id = uuid.uuid4().hex
        self.get(session_id)
        return session_id

    def drop(self, session_id: str) -> bool:
        """Remove a session, returning whether it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    @property
    def evicted(self) -> int:
        """How many sessions have been dropped by the LRU bound so far."""
        return self._evicted


sessions = SessionRegistry()

"""
Regression tests for the Phase 7 state, session, persistence and config work.

Each test pins a defect that existed before the refactor:

* ``AppState`` was a hard singleton, so every test and every tenant shared one
  object and a second ``AppState()`` silently returned the first.
* Concurrent synchronous requests could interleave writes to that object.
* ``save_model`` wrote a bare joblib blob with no record of features,
  hyperparameters or a digest, so a corrupted or swapped artifact loaded
  happily and scored silently wrong.
* Configuration, version and licence values were duplicated across files.
"""

import json
import threading
import time

import numpy as np
import pytest

from backend.config import Settings
from backend.session import SessionRegistry
from backend.state import AppState
from src.models.model_manager import ModelManager

# ── no more singleton ────────────────────────────────────────────────────────


def test_two_app_states_are_independent():
    """
    ``AppState()`` used to return the same instance through ``__new__``.

    Mutating one session must not be visible from another.
    """
    first = AppState()
    second = AppState()

    assert first is not second

    first.dataset_name = "session-one.csv"
    first.threshold_val = 0.123
    assert second.dataset_name != "session-one.csv"
    assert second.threshold_val != 0.123


def test_the_default_session_is_still_exported():
    """``state`` remains the process-wide default, so existing routers work."""
    import backend.state as state_module

    assert isinstance(state_module.state, AppState)
    # And it is not rebuilt on every import.
    again = __import__("backend.state", fromlist=["state"]).state
    assert again is state_module.state


# ── session isolation ────────────────────────────────────────────────────────


def test_registry_returns_one_state_per_session():
    registry = SessionRegistry(max_sessions=4)

    alpha = registry.get("alpha")
    beta = registry.get("beta")

    assert alpha is not beta
    assert registry.get("alpha") is alpha
    assert registry.get("beta") is beta

    alpha.dataset_name = "alpha.csv"
    assert beta.dataset_name == ""


def test_registry_evicts_the_least_recently_used_session():
    registry = SessionRegistry(max_sessions=2)

    first = registry.get("first")
    registry.get("second")
    # Touch "first" so "second" becomes the least recently used.
    assert registry.get("first") is first

    registry.get("third")

    assert registry.ids() == ["first", "third"]
    assert registry.evicted == 1


def test_registry_create_allocates_a_fresh_id():
    registry = SessionRegistry(max_sessions=3)

    session_id = registry.create()

    assert session_id in registry.ids()
    # A brand-new session carries no dataset.
    assert registry.get(session_id).dataset_name == ""


def test_registry_drop_and_invalid_input():
    registry = SessionRegistry(max_sessions=2)
    registry.get("doomed")

    assert registry.drop("doomed") is True
    assert registry.drop("doomed") is False

    with pytest.raises(ValueError):
        registry.get("")
    with pytest.raises(ValueError):
        SessionRegistry(max_sessions=0)


# ── real concurrency protection ──────────────────────────────────────────────


def test_serialized_mutations_do_not_overlap():
    """
    Two threads entering ``run_detection`` at once used to interleave writes to
    ``self.predictions`` / ``self.active_model``.

    The spy below counts how many threads are inside ``model_manager.fit``: with
    the mutation lock in place the maximum must be exactly one.
    """
    import pandas as pd

    state = AppState()
    rng = np.random.default_rng(7)
    state.X = rng.normal(size=(80, 3))
    state.feature_names = ["a", "b", "c"]
    # ``build_results_dataframe`` joins the scores back onto the raw frame.
    state.df_raw = pd.DataFrame(state.X, columns=state.feature_names)

    real_fit = state.model_manager.fit
    guard = threading.Lock()
    active = 0
    peak = 0

    def slow_fit(*args, **kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.15)
            return real_fit(*args, **kwargs)
        finally:
            with guard:
                active -= 1

    state.model_manager.fit = slow_fit  # type: ignore[method-assign]

    errors: list[BaseException] = []

    def worker():
        try:
            state.run_detection("Isolation Forest")
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors, errors
    assert peak == 1, f"{peak} concurrent fits: the mutation lock was bypassed"


# ── reproducible model persistence ───────────────────────────────────────────


def _fit_and_save(tmp_path):
    manager = ModelManager(models_dir=str(tmp_path))
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 4))
    manager.fit("Isolation Forest", X, ["a", "b", "c", "d"])
    path = manager.save_model("Isolation Forest")
    return manager, path


def test_save_writes_a_manifest_next_to_the_artifact(tmp_path):
    manager, path = _fit_and_save(tmp_path)
    manifest = manager.read_manifest("Isolation Forest")

    assert manifest is not None
    assert manifest["model_name"] == "Isolation Forest"
    assert manifest["feature_names"] == ["a", "b", "c", "d"]
    assert len(manifest["artifact_sha256"]) == 64
    assert manifest["environment"]["python"]
    assert manifest["environment"]["sklearn"]
    # The sidecar is a real file on disk, not an in-memory convenience.
    on_disk = json.loads(open(path + ".meta.json", encoding="utf-8").read())
    assert on_disk["artifact"] == "isolation_forest.joblib"


def test_load_model_accepts_the_artifact_it_saved(tmp_path):
    manager, _ = _fit_and_save(tmp_path)

    reloaded = manager.load_model("Isolation Forest")

    assert reloaded.is_fitted()


def test_load_model_refuses_a_tampered_artifact(tmp_path):
    """
    A bare ``joblib.load`` cannot tell a truncated or swapped file from the one
    that was validated.  The digest makes that an error.
    """
    manager, path = _fit_and_save(tmp_path)

    with open(path, "ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(RuntimeError, match="manifest digest"):
        manager.load_model("Isolation Forest")


def test_load_model_refuses_a_manifest_for_a_different_model(tmp_path):
    manager, _ = _fit_and_save(tmp_path)

    with pytest.raises(RuntimeError, match="Manifest describes"):
        manager.load_model("HBOS", path=f"{tmp_path}/isolation_forest.joblib")


# ── centralised configuration and versioning ─────────────────────────────────


def test_settings_parse_environment(monkeypatch):
    monkeypatch.setenv("XAUD_PORT", "9123")
    monkeypatch.setenv("XAUD_API_KEY", "  key  ")
    monkeypatch.setenv("XAUD_CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("XAUD_MAX_PAGE_SIZE", "50")

    settings = Settings()

    assert settings.PORT == 9123
    assert settings.API_KEY == "key"
    assert settings.auth_required is True
    assert settings.CORS_ORIGINS == ["https://a.example", "https://b.example"]
    assert settings.MAX_PAGE_SIZE == 50


def test_settings_fall_back_on_invalid_values(monkeypatch):
    """A typo in an env var must not crash the server at import time."""
    monkeypatch.setenv("XAUD_PORT", "not-a-port")
    monkeypatch.setenv("XAUD_MAX_ROWS", "")

    settings = Settings()

    assert settings.PORT == 8000
    assert settings.MAX_ROWS == 200_000


def test_sentinel_env_precedence_and_deprecation(monkeypatch):
    """SENTINEL_* env vars take precedence; XAUD_* issues a DeprecationWarning."""
    import pytest

    # SENTINEL overrides XAUD
    monkeypatch.setenv("SENTINEL_PORT", "8888")
    monkeypatch.setenv("XAUD_PORT", "7777")
    s = Settings()
    assert s.PORT == 8888

    # XAUD fallback triggers deprecation warning
    monkeypatch.delenv("SENTINEL_PORT", raising=False)
    monkeypatch.setenv("XAUD_PORT", "9999")
    with pytest.deprecated_call():
        s_old = Settings()
        assert s_old.PORT == 9999


def test_version_has_one_source_of_truth():
    import src
    from backend.main import app

    assert app.version == src.__version__

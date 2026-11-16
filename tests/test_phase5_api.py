"""
tests/test_phase5_api.py

Regression tests for the Phase 5 API-hardening defects.

Each test pins one defect the audit found:

* no endpoint declared a response model (OpenAPI said ``{}`` for everything);
* the frontend contract had drifted from the payload (fingerprints, anomaly
  counts, drift report shape);
* the API had no authentication;
* CORS reflected any header;
* pagination was unbounded;
* hyperparameters were forwarded to sklearn without validation;
* quadratic detectors accepted datasets they cannot finish;
* CSV exports were a formula-injection vector;
* unexpected exceptions were reported as client errors.
"""

from __future__ import annotations

import inspect
import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.state as state_module
from backend.config import settings
from backend.main import app
from backend.routers import dataset as dataset_router
from backend.routers.export import sanitize_csv_frame
from backend.schemas import ExperimentRecord
from src.models import model_manager


@pytest.fixture
def client():
    state_module.state.reset_all()
    with TestClient(app) as c:
        yield c
    state_module.state.reset_all()


@pytest.fixture
def loaded_client(client):
    res = client.post("/api/dataset/sample")
    assert res.status_code == 200, res.text
    return client


# ── Response models ──────────────────────────────────────────────────────────


def test_every_json_endpoint_declares_a_response_schema():
    """
    OpenAPI must document a response body for every JSON route.

    Endpoints that stream a ``Response`` (the CSV/JSON exports) are excluded:
    they have a media type, not a modelled body.
    """
    paths = app.openapi()["paths"]
    undocumented = []
    for path, operations in paths.items():
        if path.startswith("/api/export"):
            continue
        for method, operation in operations.items():
            schema = (
                operation.get("responses", {})
                .get("200", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            if schema is None:
                undocumented.append(f"{method.upper()} {path}")
    assert not undocumented, f"endpoints without a response model: {undocumented}"


def test_health_contract_includes_dataset_and_auth_posture(loaded_client):
    body = loaded_client.get("/api/health").json()
    assert body["dataset_name"] == "sample_anomaly_data.csv"
    assert body["preprocessed_features"] > 0
    assert body["auth_required"] is False
    assert body["version"] == app.version


# ── Frontend contract ────────────────────────────────────────────────────────


def test_available_models_returns_capabilities_and_defaults(client):
    body = client.get("/api/models/available").json()
    assert isinstance(body["models"], list) and body["models"]
    assert "Isolation Forest" in body["capabilities"]
    assert "n_estimators" in body["default_params"]["Isolation Forest"]


def test_experiment_records_carry_full_fingerprint_and_nested_results(loaded_client):
    loaded_client.post(
        "/api/detection/run",
        json={"model_name": "Isolation Forest", "params": {"n_estimators": 20}},
    )
    history = loaded_client.get("/api/experiments/history").json()
    assert history
    record = history[0]
    assert len(record["dataset_fingerprint"]) == 64
    assert record["dataset_hash"] == record["dataset_fingerprint"][:16]
    assert record["results"]["n_anomalies"] == record["n_anomalies_flagged"]
    assert record["results"]["anomaly_fraction"] == record["anomaly_rate"]


def test_legacy_experiment_records_remain_readable():
    """Historic entries on disk lack the fields added in Phase 5."""
    legacy = {
        "experiment_id": "exp_old",
        "timestamp": "2024-01-01T00:00:00",
        "dataset_name": "old.csv",
        "dataset_hash": "deadbeefdeadbeef",
        "model_name": "Isolation Forest",
        "parameters": {},
        "fit_time_sec": 1.5,
        "n_anomalies_flagged": 7,
        "anomaly_rate": 0.07,
    }
    record = ExperimentRecord.model_validate(legacy)
    assert record.dataset_fingerprint == "deadbeefdeadbeef"
    assert record.results == {
        "fit_time_sec": 1.5,
        "n_anomalies": 7,
        "anomaly_fraction": 0.07,
    }


# ── Authentication ───────────────────────────────────────────────────────────


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "s3cret")

    assert client.get("/api/dataset/summary").status_code == 401
    assert client.get("/api/health").status_code == 200

    bad = client.get("/api/dataset/summary", headers={"X-API-Key": "wrong"})
    assert bad.status_code == 401

    good = client.get("/api/dataset/summary", headers={"X-API-Key": "s3cret"})
    assert good.status_code == 200

    bearer = client.get(
        "/api/dataset/summary", headers={"Authorization": "Bearer s3cret"}
    )
    assert bearer.status_code == 200


def test_api_is_open_when_no_key_is_configured(client):
    assert settings.API_KEY == ""
    assert client.get("/api/dataset/summary").status_code == 200


# ── CORS ─────────────────────────────────────────────────────────────────────


def test_cors_rejects_unlisted_origins_and_reflects_listed_ones(client):
    preflight = {
        "Origin": "http://evil.example",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-anything",
    }
    response = client.options("/api/dataset/summary", headers=preflight)
    assert response.headers.get("access-control-allow-origin") is None

    allowed = dict(preflight, Origin="http://localhost:3000")
    response = client.options("/api/dataset/summary", headers=allowed)
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    # Only the configured headers are echoed.
    assert "*" not in response.headers.get("access-control-allow-headers", "")


# ── Pagination ───────────────────────────────────────────────────────────────


def test_page_size_is_bounded(loaded_client):
    assert loaded_client.get("/api/anomalies?page_size=0").status_code == 422
    assert (
        loaded_client.get(f"/api/anomalies?page_size={settings.MAX_PAGE_SIZE + 1}")
        .status_code
        == 422
    )
    ok = loaded_client.get(f"/api/anomalies?page_size={settings.MAX_PAGE_SIZE}")
    assert ok.status_code == 200


def test_page_number_must_be_positive(loaded_client):
    assert loaded_client.get("/api/anomalies?page=0").status_code == 422


def test_min_score_cannot_exceed_max_score(loaded_client):
    response = loaded_client.get("/api/anomalies?min_score=0.9&max_score=0.1")
    assert response.status_code == 400


# ── Hyperparameter allow-list ────────────────────────────────────────────────


def test_unknown_hyperparameter_is_rejected(loaded_client):
    response = loaded_client.post(
        "/api/detection/run",
        json={"model_name": "Isolation Forest", "params": {"n_estimatrs": 10}},
    )
    assert response.status_code == 422
    assert "n_estimatrs" in response.text


def test_unknown_model_is_rejected(client):
    response = client.post(
        "/api/detection/run", json={"model_name": "Magic Detector"}
    )
    assert response.status_code == 422


def test_ensemble_weights_for_absent_model_are_rejected(loaded_client):
    response = loaded_client.post(
        "/api/detection/ensemble",
        json={
            "models": ["Isolation Forest"],
            "weights": {"Local Outlier Factor": 0.5},
        },
    )
    assert response.status_code == 422


def test_negative_ensemble_weight_is_rejected(loaded_client):
    response = loaded_client.post(
        "/api/detection/ensemble",
        json={"models": ["Isolation Forest"], "weights": {"Isolation Forest": -1.0}},
    )
    assert response.status_code == 422


# ── Per-model row limits ─────────────────────────────────────────────────────


def test_quadratic_detector_is_refused_above_its_row_limit(loaded_client, monkeypatch):
    monkeypatch.setitem(
        model_manager.MODEL_ROW_LIMITS, "Local Outlier Factor", 5
    )
    response = loaded_client.post(
        "/api/detection/run", json={"model_name": "Local Outlier Factor"}
    )
    assert response.status_code == 400
    assert "limited to 5 rows" in response.text


def test_ensemble_cannot_smuggle_rows_past_the_limit(loaded_client, monkeypatch):
    monkeypatch.setitem(
        model_manager.MODEL_ROW_LIMITS, "Local Outlier Factor", 5
    )
    response = loaded_client.post(
        "/api/detection/ensemble",
        json={"models": ["Isolation Forest", "Local Outlier Factor"]},
    )
    assert response.status_code == 400
    assert "Local Outlier Factor" in response.text


# ── CSV export sanitisation ──────────────────────────────────────────────────


def test_csv_formula_injection_is_neutralised():
    frame = pd.DataFrame(
        {
            "label": ["=cmd|' /C calc'!A0", "@SUM(A1)", "+1", "-1", "normal"],
            "amount": [-5.2, 3, 0.0, 1e6, -0.5],
        }
    )
    clean = sanitize_csv_frame(frame)
    assert clean["label"].tolist() == [
        "'=cmd|' /C calc'!A0",
        "'@SUM(A1)",
        "'+1",
        "'-1",
        "normal",
    ]
    # Numeric columns must be untouched, including negative numbers.
    assert clean["amount"].tolist() == frame["amount"].tolist()


def test_predictions_export_sanitises_strings(loaded_client):
    loaded_client.post(
        "/api/detection/run",
        json={"model_name": "Isolation Forest", "params": {"n_estimators": 20}},
    )
    csv_text = loaded_client.get("/api/export/predictions").text
    assert "=cmd" not in csv_text
    assert "transaction_id" in csv_text


# ── Non-blocking upload ──────────────────────────────────────────────────────


def test_upload_dispatches_parsing_to_the_threadpool():
    """
    The parse must not run on the event loop.

    ``upload_csv`` is ``async`` so it can stream the body; a synchronous call
    to the loader inside it blocks every other request for the whole parse.
    """
    assert inspect.iscoroutinefunction(dataset_router.upload_csv)
    source = inspect.getsource(dataset_router)
    assert "run_in_threadpool" in source
    assert "load_user_csv" in source


def test_upload_still_works(client):
    rows = "\n".join(f"{i},{i * 2}" for i in range(30))
    csv_bytes = f"a,b\n{rows}\n".encode()
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("tiny.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["dataset"]["loaded"] is True


def test_too_few_rows_is_a_client_error_not_a_server_error(client):
    """
    A rejected dataset is the caller's input, so it must be a 4xx.

    ``ValidationError`` used to derive from ``Exception``, which no router
    catches, so a 12-row CSV produced an unhandled 500 and a traceback.
    """
    csv_bytes = b"a,b\n" + b"\n".join(b"1,2" for _ in range(12)) + b"\n"
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("tiny.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 400
    assert "Minimum required" in response.json()["detail"]


# ── Error propagation ────────────────────────────────────────────────────────


def test_unexpected_backend_failure_is_a_500_not_a_400(client, monkeypatch):
    """
    A bug in the backend is not the caller's fault.

    The routers used to wrap ``except Exception`` into HTTP 400, which hid
    tracebacks and told clients to fix a request that was perfectly valid.
    """
    def explode():
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(state_module.state, "get_profiler_report", explode)
    client.post("/api/dataset/sample")

    with TestClient(app, raise_server_exceptions=False) as raw_client:
        response = raw_client.get("/api/dataset/profile")
    assert response.status_code == 500


def test_documented_client_error_still_returns_400(client):
    """No dataset: a precondition the caller can fix."""
    response = client.get("/api/dataset/profile")
    assert response.status_code == 400


# ── Timing guard ─────────────────────────────────────────────────────────────


def test_page_bounds_keep_a_single_query_fast(loaded_client):
    start = time.perf_counter()
    response = loaded_client.get(f"/api/anomalies?page_size={settings.MAX_PAGE_SIZE}")
    elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert elapsed < 10.0

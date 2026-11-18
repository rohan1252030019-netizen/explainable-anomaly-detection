"""
tests/test_phase7_mlops.py

Tests for SENTINEL MLOps & Production Hardening (Phase 7):
1. Automated Model Card Generation (Mitchell et al., 2019).
2. Versioned Model Registry & Lifecycle Stage Promotion.
3. Prometheus Metrics Exposition & Rate Limiting.
4. Presentation-ready Standalone HTML Audit Report Generation.
5. API v2 MLOps & Hardening Endpoints.
"""


import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.mlops.html_report import generate_html_audit_report
from src.mlops.model_card import generate_model_card
from src.mlops.monitoring import MetricsTracker, RateLimiter
from src.mlops.registry import ModelRegistry


@pytest.fixture
def temp_registry(tmp_path):
    reg_dir = tmp_path / "model_registry"
    return ModelRegistry(registry_dir=str(reg_dir))


def test_model_card_generator():
    card = generate_model_card(
        model_name="Isolation Forest",
        version="3.0.0",
        dataset_name="CreditCard Benchmark",
        dataset_hash="abcd1234ef5678",
        hyperparameters={"contamination": 0.05, "n_estimators": 100},
        feature_names=["amount", "distance", "velocity"],
        metrics={"roc_auc": 0.942, "pr_auc": 0.881},
    )

    assert "model_card_data" in card
    assert "markdown" in card
    assert len(card["sha256_hash"]) == 64
    assert "# Model Card: Isolation Forest" in card["markdown"]
    assert "CreditCard Benchmark" in card["markdown"]
    assert "0.942" in card["markdown"]


def test_model_registry_lifecycle(temp_registry):
    dummy_model = {"weights": [0.1, 0.2, 0.3], "name": "dummy_detector"}
    card_info = {"markdown": "# Model Card Test", "model_card_data": {"v": "1.0"}}

    # 1. Register model
    meta = temp_registry.register_model(
        name="Isolation Forest",
        version="1.0.0",
        model_obj=dummy_model,
        model_card=card_info,
        stage="staging",
        tags=["production-candidate"],
    )
    assert meta["stage"] == "staging"
    assert meta["version"] == "1.0.0"

    # 2. List models
    models = temp_registry.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "Isolation Forest"

    # 3. Retrieve model
    loaded = temp_registry.get_model(name="Isolation Forest", version="1.0.0")
    assert loaded["metadata"]["stage"] == "staging"
    assert loaded["model_obj"] == dummy_model

    # 4. Promote stage
    promoted = temp_registry.promote_stage(name="Isolation Forest", version="1.0.0", new_stage="production")
    assert promoted["stage"] == "production"

    # Verify updated
    reloaded = temp_registry.get_model(name="Isolation Forest", version="1.0.0")
    assert reloaded["metadata"]["stage"] == "production"


def test_metrics_and_rate_limiter():
    tracker = MetricsTracker()
    tracker.record_request(endpoint="/api/health", method="GET", status_code=200)
    tracker.record_anomalies(15)
    tracker.set_psi(0.042)

    prom_text = tracker.generate_prometheus_metrics()
    assert "sentinel_http_requests_total" in prom_text
    assert 'endpoint="/api/health"' in prom_text
    assert "sentinel_anomalies_detected_total 15" in prom_text
    assert "sentinel_streaming_drift_psi 0.0420" in prom_text

    # Rate Limiter
    limiter = RateLimiter(max_requests_per_minute=3)
    ip = "192.168.1.100"
    assert limiter.is_allowed(ip) is True
    assert limiter.is_allowed(ip) is True
    assert limiter.is_allowed(ip) is True
    # 4th request within 1 min should be denied
    assert limiter.is_allowed(ip) is False


def test_html_audit_report_generation():
    report_data = {
        "title": "SENTINEL Anomaly Audit Report",
        "framework": "SENTINEL 3.0",
        "generated_at": "November 18, 2026",
        "dataset": {"name": "Test Data", "rows": 1000, "features_preprocessed": 10},
        "detection": {"model": "Isolation Forest", "anomalies_detected": 50, "anomaly_rate": 0.05, "threshold_value": 0.55},
        "top_drivers": [{"feature": "amount", "importance": 0.42}],
        "sample_anomalies": [{"row_id": 12, "score": 0.89, "severity": "Critical"}],
        "methodology_note": "Unsupervised statistical ranking.",
    }

    html_out = generate_html_audit_report(report_data)
    assert "<!DOCTYPE html>" in html_out
    assert "SENTINEL Anomaly Audit Report" in html_out
    assert "badge-critical" in html_out
    assert "1,000" in html_out
    assert "amount" in html_out


def test_api_v2_mlops_endpoints():
    client = TestClient(app)

    # 1. Prometheus /metrics endpoint
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "sentinel_http_requests_total" in metrics_resp.text

    # 2. Load dataset and run detection
    client.post("/api/dataset/sample")
    client.post("/api/detection/run", json={"model_name": "Isolation Forest"})

    # 3. Model Card generation endpoint
    card_resp = client.post("/api/v2/mlops/model-card/generate", json={"version": "3.0.0"})
    assert card_resp.status_code == 200
    card_data = card_resp.json()
    assert "markdown" in card_data
    assert "sha256_hash" in card_data

    # 4. Registry save endpoint
    save_resp = client.post(
        "/api/v2/mlops/registry/save",
        json={"name": "Isolation Forest", "version": "1.0.0", "stage": "staging"},
    )
    assert save_resp.status_code == 200
    assert save_resp.json()["success"] is True

    # 5. Registry list endpoint
    list_resp = client.get("/api/v2/mlops/registry/list")
    assert list_resp.status_code == 200
    assert len(list_resp.json()["models"]) >= 1

    # 6. Registry promote endpoint
    prom_resp = client.post(
        "/api/v2/mlops/registry/promote",
        json={"name": "Isolation Forest", "version": "1.0.0", "stage": "production"},
    )
    assert prom_resp.status_code == 200
    assert prom_resp.json()["metadata"]["stage"] == "production"

    # 7. HTML Report export endpoint
    html_resp = client.get("/api/v2/reports/html")
    assert html_resp.status_code == 200
    assert "<!DOCTYPE html>" in html_resp.text
    assert "SENTINEL" in html_resp.text

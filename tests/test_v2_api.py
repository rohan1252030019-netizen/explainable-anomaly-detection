"""
tests/test_v2_api.py

Integration tests for XAUD 2.0 API endpoints:
- /api/dataset/profile
- /api/models/capabilities
- /api/detection/ensemble
- /api/drift/evaluate & /api/drift/status
- /api/benchmarks/run & /api/benchmarks/status
- /api/experiments/history
"""

import pytest
from fastapi.testclient import TestClient

import backend.state as state_module
from backend.main import app


@pytest.fixture
def client():
    state_module.state.reset_all()
    with TestClient(app) as c:
        yield c
    state_module.state.reset_all()


@pytest.fixture
def loaded_client(client):
    res = client.post("/api/dataset/sample")
    assert res.status_code == 200
    return client


def test_dataset_profile_endpoint(loaded_client):
    res = loaded_client.get("/api/dataset/profile")
    assert res.status_code == 200
    body = res.json()
    assert "health" in body
    assert "schema" in body
    assert "quality" in body
    assert "collinearity" in body
    assert body["health"]["score"] > 0


def test_models_capabilities_endpoint(client):
    res = client.get("/api/models/capabilities")
    assert res.status_code == 200
    body = res.json()
    assert "Isolation Forest" in body
    assert "k-Nearest Neighbors" in body
    assert "PCA Reconstruction" in body
    assert "Histogram-Based Outlier Score" in body
    assert "Gaussian Mixture Model" in body


def test_detection_ensemble_endpoint(loaded_client):
    req_body = {
        "models": ["Isolation Forest", "k-Nearest Neighbors", "Histogram-Based Outlier Score"],
        "aggregation_method": "rank_borda",
    }
    res = loaded_client.post("/api/detection/ensemble", json=req_body)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert "ensemble" in body
    assert len(body["ensemble"]["ensemble_scores"]) > 0
    assert "disagreement" in body["ensemble"]


def test_drift_endpoints(loaded_client):
    res = loaded_client.post("/api/drift/evaluate", json={"psi_bins": 10})
    assert res.status_code == 200
    body = res.json()
    assert "overall_severity" in body
    assert "feature_reports" in body

    status_res = loaded_client.get("/api/drift/status")
    assert status_res.status_code == 200
    assert status_res.json()["overall_severity"] == body["overall_severity"]


def test_benchmarks_endpoints(loaded_client):
    req = {
        "anomaly_type": "multivariate",
        "contamination": 0.05,
        "models_to_test": ["Isolation Forest", "Histogram-Based Outlier Score"],
    }
    res = loaded_client.post("/api/benchmarks/run", json=req)
    assert res.status_code == 200
    body = res.json()
    assert body["experiment_type"] == "Synthetic Anomaly Benchmark"
    assert len(body["results_table"]) >= 1

    status_res = loaded_client.get("/api/benchmarks/status")
    assert status_res.status_code == 200
    assert status_res.json()["anomaly_type"] == "multivariate"


def test_experiments_endpoints(loaded_client):
    # Trigger a detection run to generate an experiment entry
    loaded_client.post("/api/detection/run", json={"model_name": "Isolation Forest"})

    res = loaded_client.get("/api/experiments/history")
    assert res.status_code == 200
    history = res.json()
    assert len(history) >= 1
    exp_id = history[0]["experiment_id"]

    detail_res = loaded_client.get(f"/api/experiments/{exp_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["experiment_id"] == exp_id

"""
Integration tests for the FastAPI backend.

These exercise the real request path end to end against a freshly reset state,
so a regression in a router, a schema, or the state wiring fails here rather
than in the browser.
"""

import io

import pytest
from fastapi.testclient import TestClient

from backend import state as state_module
from backend.config import settings


@pytest.fixture
def client():
    """A TestClient whose state is reset before each test."""
    state_module.state.reset_all()
    with TestClient(state_module.state and __import__(
        "backend.main", fromlist=["app"]
    ).app) as c:
        yield c
    state_module.state.reset_all()


@pytest.fixture
def loaded_client(client):
    """A client with the sample dataset loaded and detection already run."""
    assert client.post("/api/dataset/sample").status_code == 200
    response = client.post(
        "/api/detection/run",
        json={"model_name": "Isolation Forest", "contamination": 0.05},
    )
    assert response.status_code == 200
    return client


class TestHealth:
    def test_health_reports_initial_state(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "healthy"
        assert body["dataset_loaded"] is False

    def test_health_tracks_the_pipeline(self, loaded_client):
        body = loaded_client.get("/api/health").json()
        assert body["dataset_loaded"] is True
        assert body["detection_ready"] is True
        assert body["anomalies_detected"] > 0


class TestDatasetRouter:
    def test_summary_before_load(self, client):
        body = client.get("/api/dataset/summary").json()
        assert body["loaded"] is False

    def test_load_sample(self, client):
        response = client.post("/api/dataset/sample")
        assert response.status_code == 200
        dataset = response.json()["dataset"]
        assert dataset["loaded"] is True
        assert dataset["n_rows"] > 0

    def test_preprocessing_runs_automatically(self, client):
        client.post("/api/dataset/sample")
        summary = client.get("/api/dataset/summary").json()
        assert summary["feature_matrix_ready"] is True
        assert summary["n_preprocessed_features"] > 0

    def test_rejects_non_csv_upload(self, client):
        response = client.post(
            "/api/dataset/upload",
            files={"file": ("data.txt", io.BytesIO(b"a,b\n1,2"), "text/plain")},
        )
        assert response.status_code == 400

    def test_rejects_empty_upload(self, client):
        response = client.post(
            "/api/dataset/upload",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        assert response.status_code == 400

    def test_rejects_oversized_upload(self, client, monkeypatch):
        """An unbounded read is the cheapest way to exhaust memory."""
        monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 16)
        response = client.post(
            "/api/dataset/upload",
            files={"file": ("big.csv", io.BytesIO(b"x" * 1024), "text/csv")},
        )
        assert response.status_code == 413

    def test_reset_clears_state(self, client):
        client.post("/api/dataset/sample")
        assert client.post("/api/dataset/reset").status_code == 200
        assert client.get("/api/dataset/summary").json()["loaded"] is False


class TestDetectionRouter:
    def test_results_before_running_detection(self, client):
        client.post("/api/dataset/sample")
        assert client.get("/api/detection/results").json()["ready"] is False

    def test_run_detection(self, loaded_client):
        body = loaded_client.get("/api/detection/results").json()
        assert body["ready"] is True
        assert body["n_anomalies"] > 0
        assert 0.0 <= body["anomaly_rate"] <= 1.0

    def test_severity_bands_are_all_populated(self, loaded_client):
        """
        Regression guard: fixed score bands collapsed the severity donut into a
        single colour, because heavy-tailed scores put every flagged row below
        the first band edge.
        """
        body = loaded_client.get("/api/detection/results").json()
        counts = body["severity_counts"]
        assert all(v > 0 for v in counts.values()), counts

    def test_severity_legend_explains_the_bands(self, loaded_client):
        body = loaded_client.get("/api/detection/results").json()
        bands = body["severity_bands"]
        assert bands["mode"] == "rank_within_flagged"
        assert len(bands["levels"]) == 4
        assert all(row["definition"] for row in bands["levels"])

    def test_reports_the_normalisation_used(self, loaded_client):
        body = loaded_client.get("/api/detection/results").json()
        norm = body["score_normalization"]
        assert norm["method"] == "quantile"
        assert norm["scale"] > 0

    def test_ensemble_mode_runs(self, client):
        client.post("/api/dataset/sample")
        response = client.post(
            "/api/detection/run",
            json={"model_name": "Isolation Forest", "use_ensemble": True},
        )
        assert response.status_code == 200
        assert response.json()["overview"]["ready"] is True

    def test_quantile_threshold_mode(self, loaded_client):
        response = loaded_client.post(
            "/api/detection/run",
            json={
                "model_name": "Isolation Forest",
                "threshold_method": "quantile",
                "quantile_val": 0.98,
            },
        )
        assert response.status_code == 200
        body = response.json()["overview"]
        assert body["threshold_method"] == "quantile"

    def test_models_endpoint_lists_the_registry(self, loaded_client):
        models = loaded_client.get("/api/detection/models").json()["models"]
        assert "Isolation Forest" in models

    def test_experiments_are_recorded(self, loaded_client):
        loaded_client.post(
            "/api/detection/run", json={"model_name": "One-Class SVM"}
        )
        experiments = loaded_client.get("/api/detection/experiments").json()
        assert len(experiments["experiments"]) >= 2

    def test_invalid_model_is_rejected_cleanly(self, loaded_client):
        response = loaded_client.post(
            "/api/detection/run", json={"model_name": "Not A Model"}
        )
        assert response.status_code >= 400


class TestAnomaliesRouter:
    def test_lists_only_anomalies_by_default(self, loaded_client):
        body = loaded_client.get("/api/anomalies?anomalies_only=true").json()
        assert body["total"] > 0
        assert all(item["is_anomaly"] for item in body["items"])

    def test_pagination_is_consistent(self, loaded_client):
        first = loaded_client.get("/api/anomalies?page=1&page_size=5").json()
        second = loaded_client.get("/api/anomalies?page=2&page_size=5").json()
        assert len(first["items"]) == 5
        assert first["total"] == second["total"]
        assert first["total_pages"] == pytest.approx(
            first["total"] / 5, abs=1
        )

    def test_severity_filter(self, loaded_client):
        body = loaded_client.get(
            "/api/anomalies?severity=CRITICAL"
        ).json()
        assert all(item["severity"] == "CRITICAL" for item in body["items"])

    def test_results_carry_the_original_index(self, loaded_client):
        """Needed so the UI can navigate from a row back to its investigation."""
        body = loaded_client.get("/api/anomalies?page=1&page_size=3").json()
        assert all("original_index" in item for item in body["items"])


class TestInvestigation:
    @pytest.fixture
    def top_row_id(self, loaded_client):
        body = loaded_client.get(
            "/api/anomalies?page=1&page_size=1&sort_by=rank&ascending=true"
        ).json()
        return int(body["items"][0]["original_index"])

    def test_returns_a_coherent_record(self, loaded_client, top_row_id):
        body = loaded_client.get(
            f"/api/anomalies/{top_row_id}/investigate"
        ).json()
        assert body["row_id"] == top_row_id
        assert body["rank"] == 1
        assert body["feature_comparison"]

    def test_rank_matches_the_anomaly_list(self, loaded_client, top_row_id):
        """Regression guard for positional lookup in a sorted frame."""
        listed = loaded_client.get(
            "/api/anomalies?page=1&page_size=1&sort_by=rank&ascending=true"
        ).json()["items"][0]
        investigated = loaded_client.get(
            f"/api/anomalies/{top_row_id}/investigate"
        ).json()
        assert investigated["rank"] == listed["rank"]
        assert investigated["normalized_score"] == pytest.approx(
            listed["normalized_score"], abs=1e-4
        )

    def test_includes_a_counterfactual(self, loaded_client, top_row_id):
        body = loaded_client.get(
            f"/api/anomalies/{top_row_id}/investigate"
        ).json()
        cf = body["counterfactual"]
        assert "found" in cf
        if cf["found"]:
            assert cf["text"]
            assert cf["quality"]["label"] in {
                "STRONG", "PARTIAL", "WEAK"
            }
            assert cf["caveat"]

    def test_includes_an_explanation_verification(self, loaded_client,
                                                  top_row_id):
        body = loaded_client.get(
            f"/api/anomalies/{top_row_id}/investigate"
        ).json()
        v = body["verification"]
        assert v["overall"] in {
            "BEHAVIOURALLY FAITHFUL", "UNVERIFIED", "UNKNOWN"
        }
        assert v["faithfulness"]["deletion_auc_explained"] is not None

    def test_out_of_range_row_is_an_error(self, loaded_client):
        response = loaded_client.get("/api/anomalies/99999/investigate")
        assert response.status_code >= 400

    def test_requires_detection_first(self, client):
        client.post("/api/dataset/sample")
        response = client.get("/api/anomalies/0/investigate")
        assert response.status_code >= 400


class TestEvaluationRouter:
    def test_holdout_reports_both_splits(self, loaded_client):
        body = loaded_client.get("/api/evaluation/holdout").json()
        assert body["available"] is True
        assert body["in_sample"]["roc_auc"] is not None
        assert body["out_of_sample"]["roc_auc"] is not None
        assert body["summary"]

    def test_holdout_comparison_table(self, loaded_client):
        body = loaded_client.get("/api/evaluation/holdout/all").json()
        assert body["table"]
        assert "ROC-AUC (out-of-sample)" in body["table"][0]

    def test_rejects_out_of_range_test_size(self, loaded_client):
        response = loaded_client.get("/api/evaluation/holdout?test_size=5")
        assert response.status_code == 422


class TestExplainabilityRouter:
    def test_global_importance_before_generation(self, loaded_client):
        assert loaded_client.get(
            "/api/explainability/global"
        ).json()["ready"] is False

    def test_generate_explanations(self, loaded_client):
        response = loaded_client.post(
            "/api/explainability/generate", json={"use_shap": False, "top_k": 5}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["n_explained"] > 0
        assert body["global_importance"]

    def test_stability_endpoint(self, loaded_client):
        body = loaded_client.get("/api/explainability/stability").json()
        assert body["n_fits"] >= 2
        assert body["mean_rank_correlation"] is not None
        assert body["verdict"]

    def test_feature_distribution_separation(self, loaded_client):
        body = loaded_client.get(
            "/api/explainability/feature-distributions?feature=transaction_amount"
        ).json()
        assert body["distribution"]
        assert body["separation"] >= 0

    def test_unknown_feature_is_404(self, loaded_client):
        response = loaded_client.get(
            "/api/explainability/feature-distributions?feature=nope"
        )
        assert response.status_code == 404


class TestAnalyticsAndModels:
    def test_pca_projection(self, loaded_client):
        body = loaded_client.get("/api/analytics/pca").json()
        assert body["points"]
        assert len(body["variance_explained"]) == 2

    def test_correlations(self, loaded_client):
        body = loaded_client.get("/api/analytics/correlations").json()
        assert body["features"]
        assert len(body["matrix"]) == len(body["features"])

    def test_model_benchmark(self, loaded_client):
        body = loaded_client.post(
            "/api/models/benchmark", json={"contamination": 0.05}
        ).json()
        assert body["comparison_table"]
        assert body["model_anomaly_counts"]

    def test_report_generation(self, loaded_client):
        body = loaded_client.get("/api/reports/generate").json()
        assert body["dataset"]["name"]
        assert body["detection"]["model"]
        assert body["methodology_note"]


class TestOpenAPI:
    def test_schema_is_valid(self, client):
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"]
        assert "/api/health" in schema["paths"]

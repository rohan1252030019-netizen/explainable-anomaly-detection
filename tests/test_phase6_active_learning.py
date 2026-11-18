"""
tests/test_phase6_active_learning.py

Tests for SENTINEL Human-in-the-Loop, Active Learning & Root-Cause Auditing (Phase 6):
1. Analyst Feedback Store & Annotation Precision Metrics.
2. Active Learning Query Strategies (Uncertainty, Disagreement, Hybrid).
3. Semi-Supervised Model Updating with Zero-Leakage Held-Out Evaluation.
4. Root-Cause Attribution Clustering (HDBSCAN on SHAP).
5. Subgroup Fairness Equity & Noise Perturbation Robustness Auditing.
6. API v2 Active Learning & Audit Endpoints.
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from src.audit.fairness_robustness import (
    evaluate_noise_robustness,
    evaluate_subgroup_fairness,
)
from src.audit.root_cause import cluster_root_causes
from src.learning.feedback_store import AnalystFeedbackStore
from src.learning.query_strategies import query_active_learning_batch
from src.learning.semi_supervised import SemiSupervisedAnomalyUpdater


def test_analyst_feedback_store():
    store = AnalystFeedbackStore(session_id="test_session")
    store.add_feedback(record_id=10, user_label=1, analyst_id="analyst_alice", notes="Confirmed wire fraud")
    store.add_feedback(record_id=15, user_label=0, analyst_id="analyst_bob", notes="Legitimate bulk transfer")
    store.add_feedback(record_id=22, user_label=1, analyst_id="analyst_alice")

    summary = store.get_summary_metrics()
    assert summary["total_reviewed"] == 3
    assert summary["confirmed_anomalies"] == 2
    assert summary["false_positives"] == 1
    assert round(summary["alert_precision"], 2) == 0.67
    assert summary["is_ready_for_semi_supervised"] is False

    # Add more to reach threshold
    for i in range(3):
        store.add_feedback(record_id=30 + i, user_label=0)
    summary_ready = store.get_summary_metrics()
    assert summary_ready["total_reviewed"] == 6
    assert summary_ready["is_ready_for_semi_supervised"] is True


def test_active_learning_query_strategies():
    scores = np.array([0.05, 0.49, 0.51, 0.95, 0.70])
    threshold = 0.50

    detector_scores = {
        "m1": np.array([0.05, 0.45, 0.55, 0.95, 0.70]),
        "m2": np.array([0.05, 0.85, 0.15, 0.95, 0.70]),
    }

    # 1. Uncertainty: rows 1 and 2 are closest to 0.50
    candidates_unc = query_active_learning_batch(
        norm_scores=scores,
        threshold=threshold,
        strategy="uncertainty",
        batch_size=2,
    )
    assert len(candidates_unc) == 2
    top_indices = [c["index"] for c in candidates_unc]
    assert 1 in top_indices or 2 in top_indices

    # 2. Disagreement: row 1 (0.45 vs 0.85) and row 2 (0.55 vs 0.15) have maximum conflict
    candidates_dis = query_active_learning_batch(
        norm_scores=scores,
        threshold=threshold,
        all_detector_scores=detector_scores,
        strategy="disagreement",
        batch_size=2,
    )
    assert len(candidates_dis) == 2
    assert candidates_dis[0]["disagreement_score"] > 0.0


def test_semi_supervised_updating_leakage_safety():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(100, 4))
    # Synthetic scores correlated with feature 0
    scores = 1.0 / (1.0 + np.exp(-X[:, 0]))

    labeled_indices = [0, 1, 2, 3, 4, 5]
    labels = [1, 1, 1, 0, 0, 0]

    updater = SemiSupervisedAnomalyUpdater()
    res = updater.update_with_feedback(
        X=X,
        unsupervised_scores=scores,
        labeled_indices=labeled_indices,
        labels=labels,
        held_out_ratio=0.3,
    )

    assert res["is_fitted"] is True
    assert len(res["refined_scores"]) == 100
    assert "metrics" in res
    assert "held_out_score_correlation" in res["metrics"]
    assert "zero-leakage" in res["evaluation_policy"].lower()


def test_root_cause_clustering():
    # Create 2 distinct attribution archetypes:
    # Archetype A: high attribution on feature 0
    # Archetype B: high attribution on feature 2
    attrs_a = np.zeros((10, 4))
    attrs_a[:, 0] = 0.8
    attrs_a[:, 1] = 0.1

    attrs_b = np.zeros((10, 4))
    attrs_b[:, 2] = 0.9
    attrs_b[:, 3] = 0.1

    all_attrs = np.vstack([attrs_a, attrs_b])
    anomaly_indices = list(range(20))
    feature_names = ["amount", "age", "frequency", "score"]

    res = cluster_root_causes(
        attribution_matrix=all_attrs,
        anomaly_indices=anomaly_indices,
        feature_names=feature_names,
        min_cluster_size=3,
    )

    assert res["n_clusters"] >= 2
    cluster_names = [c["cluster_name"] for c in res["clusters"]]
    assert any("amount" in name for name in cluster_names)
    assert any("frequency" in name for name in cluster_names)


def test_subgroup_fairness_and_noise_robustness():
    df = pd.DataFrame({
        "region": ["North"] * 50 + ["South"] * 50,
        "f1": np.random.default_rng(42).normal(0, 1, size=100),
    })
    preds = np.array([-1] * 10 + [1] * 40 + [-1] * 12 + [1] * 38)
    scores = np.random.default_rng(42).uniform(0, 1, size=100)

    fairness = evaluate_subgroup_fairness(
        df=df,
        group_column="region",
        predictions=preds,
        norm_scores=scores,
    )
    assert fairness["n_subgroups"] == 2
    assert "disparate_impact_ratio" in fairness
    assert fairness["disparate_impact_ratio"] > 0.0

    # Noise robustness
    X = np.random.default_rng(42).normal(0, 1, size=(50, 4))

    def dummy_score_fn(mat):
        return np.mean(mat ** 2, axis=1)

    robustness = evaluate_noise_robustness(
        X=X,
        score_fn=dummy_score_fn,
        noise_stds=[0.02, 0.05],
    )
    assert "robustness_verdict" in robustness
    assert len(robustness["perturbation_levels"]) == 2


def test_api_v2_active_learning_and_audit_endpoints():
    client = TestClient(app)

    # 1. Load benchmark & run detection
    load_resp = client.post("/api/dataset/sample")
    assert load_resp.status_code == 200

    det_resp = client.post("/api/detection/run", json={"model_name": "Isolation Forest"})
    assert det_resp.status_code == 200

    # 2. Submit feedback
    fb1 = client.post("/api/v2/feedback/submit", json={"record_id": 0, "user_label": 1, "notes": "True anomaly"})
    assert fb1.status_code == 200
    fb2 = client.post("/api/v2/feedback/submit", json={"record_id": 1, "user_label": 0, "notes": "False positive"})
    assert fb2.status_code == 200

    # 3. Get feedback summary
    summary_resp = client.get("/api/v2/feedback/summary")
    assert summary_resp.status_code == 200
    assert summary_resp.json()["total_reviewed"] >= 2

    # 4. Get active learning queue
    queue_resp = client.post("/api/v2/active-learning/queue", json={"batch_size": 5, "strategy": "hybrid"})
    assert queue_resp.status_code == 200
    q_data = queue_resp.json()
    assert len(q_data["candidates"]) <= 5

    # 5. Semi-supervised retrain (add 2 more to satisfy minimum 4)
    client.post("/api/v2/feedback/submit", json={"record_id": 2, "user_label": 1})
    client.post("/api/v2/feedback/submit", json={"record_id": 3, "user_label": 0})

    retrain_resp = client.post("/api/v2/active-learning/retrain-semi-supervised", json={"held_out_ratio": 0.25})
    assert retrain_resp.status_code == 200
    ret_data = retrain_resp.json()
    assert ret_data["success"] is True
    assert "metrics" in ret_data

    # 6. Root-cause clustering
    clus_resp = client.post("/api/v2/root-cause/cluster", json={"min_cluster_size": 3})
    assert clus_resp.status_code == 200
    clus_data = clus_resp.json()
    assert "clusters" in clus_data

    # 7. Robustness audit
    rob_resp = client.post("/api/v2/robustness/audit", json={"group_column": "region", "test_noise": True})
    assert rob_resp.status_code == 200
    rob_data = rob_resp.json()
    assert rob_data["fairness"] is not None
    assert rob_data["noise_robustness"] is not None

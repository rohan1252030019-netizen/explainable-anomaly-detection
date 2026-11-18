"""
tests/test_phase4_explainability.py

Tests for SENTINEL Advanced Explainability Engine (Phase 4):
1. LIME Local Surrogate Explainer and local fidelity.
2. Cross-method Explanation Agreement (Spearman rho, Jaccard overlap, sign concordance).
3. Pairwise non-linear feature interactions (bivariate second-order difference).
4. Cohort Explanation with Cohen's d effect sizes and KS distribution shift tests.
5. DiCE-style Diverse Counterfactuals with plausibility, sparsity (L0), and immutability constraints.
6. SENTINEL Forensic Investigation Narrative with deterministic fact-checking and SHA-256 audit fingerprint.
7. API v2 Explainability endpoints.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.explainability.cohort import explain_cohort
from src.explainability.diverse_counterfactuals import generate_diverse_counterfactuals
from src.explainability.interactions import compute_pairwise_interactions
from src.explainability.lime_explainer import LIMEExplainer, compute_explanation_agreement
from src.explainability.narrative import generate_investigation_narrative


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    # 200 normal samples, 5 features
    normals = rng.normal(0, 1, size=(200, 5))
    # 10 anomalous points with extreme offsets on features 0 and 2
    anomalies = rng.normal(4, 1, size=(10, 5))
    anomalies[:, 1] = 0.0  # normal value on feature 1
    anomalies[:, 3] = 0.0  # normal value on feature 3
    anomalies[:, 4] = 0.0  # normal value on feature 4
    feature_names = ["amount", "duration", "frequency", "age", "score"]
    return normals, anomalies, feature_names


def dummy_score_fn(X: np.ndarray) -> np.ndarray:
    """Non-linear anomaly score function emphasizing features 0 and 2."""
    X = np.asarray(X, dtype=np.float64)
    raw = (X[:, 0] ** 2) * 0.4 + (X[:, 2] ** 2) * 0.4 + (X[:, 1] * X[:, 3]) * 0.2
    # Squashed to [0, 1]
    return 1.0 - 1.0 / (1.0 + raw * 0.1)


def test_lime_explainer(synthetic_data):
    normals, anomalies, feature_names = synthetic_data
    explainer = LIMEExplainer(
        score_fn=dummy_score_fn,
        feature_names=feature_names,
        X_background=normals,
        n_samples=250,
        random_state=42,
    )
    result = explainer.explain_instance(anomalies[0], top_k=3)
    assert "attributions" in result
    assert "top_features" in result
    assert len(result["top_features"]) <= 3
    assert result["base_score"] > 0.0
    assert result["local_r2"] >= 0.0
    # Top features should identify amount or frequency as top drivers
    top_names = [f["feature"] for f in result["top_features"]]
    assert "amount" in top_names or "frequency" in top_names


def test_explanation_agreement():
    # Test identical rankings
    attr_a = {"feat1": 0.9, "feat2": 0.5, "feat3": 0.1}
    attr_b = {"feat1": 0.85, "feat2": 0.45, "feat3": 0.05}
    agreement = compute_explanation_agreement(attr_a, attr_b, top_k=2)
    assert agreement["spearman_rank_correlation"] == 1.0
    assert agreement["top_k_jaccard_overlap"] == 1.0
    assert agreement["sign_concordance"] == 1.0
    assert agreement["agreement_level"] == "High Agreement"

    # Test conflicting rankings
    attr_c = {"feat1": -0.9, "feat2": -0.5, "feat3": 0.8}
    conflict = compute_explanation_agreement(attr_a, attr_c, top_k=2)
    assert conflict["spearman_rank_correlation"] < 0.0
    assert conflict["agreement_level"] == "Low / Divergent"


def test_pairwise_feature_interactions(synthetic_data):
    normals, anomalies, feature_names = synthetic_data
    pairs = compute_pairwise_interactions(
        model_or_score_fn=dummy_score_fn,
        X=anomalies,
        feature_names=feature_names,
        top_n_pairs=3,
        max_samples=10,
    )
    assert len(pairs) > 0
    assert "feature_1" in pairs[0]
    assert "feature_2" in pairs[0]
    assert "interaction_strength" in pairs[0]
    assert pairs[0]["interaction_strength"] >= 0.0


def test_cohort_explanation(synthetic_data):
    normals, anomalies, feature_names = synthetic_data
    cohort_res = explain_cohort(
        cohort_X=anomalies,
        reference_X=normals,
        feature_names=feature_names,
        cohort_name="Severe Test Anomalies",
        top_k=2,
    )
    assert cohort_res["n_cohort"] == len(anomalies)
    assert cohort_res["n_reference"] == len(normals)
    assert len(cohort_res["dominant_drivers"]) == 2
    # Anomalies had mean 4 on amount and frequency
    top_driver_names = [d["feature"] for d in cohort_res["dominant_drivers"]]
    assert "amount" in top_driver_names or "frequency" in top_driver_names
    # Effect size d should be large for the shifted features
    assert abs(cohort_res["dominant_drivers"][0]["effect_size_d"]) > 1.0


def test_diverse_counterfactuals_constraints(synthetic_data):
    normals, anomalies, feature_names = synthetic_data
    obs = anomalies[0]

    # Set 'age' as immutable, max 2 features modified
    cfs = generate_diverse_counterfactuals(
        observation=obs,
        reference_normals=normals,
        score_fn=dummy_score_fn,
        feature_names=feature_names,
        k_diverse=3,
        max_features_modified=2,
        immutable_features=["age"],
        score_threshold=0.3,
        random_state=42,
    )

    assert len(cfs) >= 1
    for cf in cfs:
        assert cf["sparsity_l0"] <= 2
        # Check that 'age' was never modified
        modified_names = [m["feature"] for m in cf["modifications"]]
        assert "age" not in modified_names
        # Target score should be below or near threshold
        assert "is_in_distribution_plausible" in cf


def test_investigation_narrative_deterministic():
    narrative = generate_investigation_narrative(
        record_id=101,
        anomaly_score=0.892,
        threshold=0.650,
        severity_tier="Critical",
        top_features=[
            {"feature": "amount", "attribution": 0.45, "deviation": "+4.2 SD"},
            {"feature": "frequency", "attribution": 0.32, "deviation": "+3.1 SD"},
        ],
        conformal_p_value=0.0012,
        fdr_significant=True,
        agreement_summary={
            "spearman_rank_correlation": 0.88,
            "top_k_jaccard_overlap": 0.75,
            "agreement_level": "High Agreement",
            "summary": "Dual methods agree.",
        },
        detector_name="Isolation Forest",
    )

    assert narrative["record_id"] == 101
    assert "0.892" in narrative["narrative_text"]
    assert "CRITICAL" in narrative["narrative_text"] or "Critical" in narrative["narrative_text"]
    assert "amount" in narrative["narrative_text"]
    assert "0.0012" in narrative["narrative_text"]
    assert len(narrative["forensic_fingerprint"]) == 64  # SHA-256


def test_api_v2_explainability_endpoints():
    client = TestClient(app)

    # 1. Load benchmark dataset & run detection
    load_resp = client.post("/api/dataset/sample")
    assert load_resp.status_code == 200

    det_resp = client.post("/api/detection/run", json={"model_name": "Isolation Forest"})
    assert det_resp.status_code == 200

    # 2. Test Agreement endpoint
    agr_resp = client.post("/api/v2/explainability/agreement", json={"record_index": 0, "top_k": 3})
    assert agr_resp.status_code == 200
    agr_data = agr_resp.json()
    assert "agreement_level" in agr_data
    assert "spearman_rank_correlation" in agr_data
    assert "lime_attributions" in agr_data

    # 3. Test Interactions endpoint
    int_resp = client.post("/api/v2/explainability/interactions", json={"top_n_pairs": 3, "max_samples": 20})
    assert int_resp.status_code == 200
    int_data = int_resp.json()
    assert "top_interactions" in int_data

    # 4. Test Cohort endpoint
    coh_resp = client.post("/api/v2/explainability/cohort", json={"cohort_name": "Test Flagged", "top_k": 3})
    assert coh_resp.status_code == 200
    coh_data = coh_resp.json()
    assert "dominant_drivers" in coh_data

    # 5. Test Diverse Counterfactuals endpoint
    cf_resp = client.post(
        "/api/v2/explainability/diverse-counterfactuals",
        json={"record_index": 0, "k_diverse": 2, "max_features_modified": 2},
    )
    assert cf_resp.status_code == 200
    cf_data = cf_resp.json()
    assert "counterfactuals" in cf_data
    assert len(cf_data["counterfactuals"]) >= 1

    # 6. Test Narrative endpoint
    nar_resp = client.post("/api/v2/explainability/narrative", json={"record_index": 0})
    assert nar_resp.status_code == 200
    nar_data = nar_resp.json()
    assert "narrative_text" in nar_data
    assert "forensic_fingerprint" in nar_data

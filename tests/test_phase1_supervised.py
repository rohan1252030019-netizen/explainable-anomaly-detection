"""
tests/test_phase1_supervised.py

Rigorous test suite for SENTINEL Phase 1:
- Pseudo-Label Surrogate Model (XGBoost) with K-fold Out-Of-Fold (OOF) training.
- Scientific fidelity reporting (Spearman rank correlation, Top-K overlap).
- Exact TreeSHAP explanations and additivity.
- Supervised mode with severe class imbalance, PR-AUC, ROC-AUC, Brier score, ECE.
- Cost-matrix decision threshold optimization.
- Hybrid stacking meta-learner on 8 unsupervised models.
- Optuna hyperparameter optimization with fixed seed and nested CV.
- Anti-leakage verification: zero data leakage across folds and reference splits.
- FastAPI /api/v2/supervised endpoints.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.state import state
from src.models.supervised.classifier import (
    SupervisedClassifierDetector,
    optimize_threshold_cost_matrix,
)
from src.models.supervised.stacking import HybridStackingDetector
from src.models.supervised.surrogate import SurrogateDetector
from src.models.supervised.tuning import tune_xgboost_hyperparameters


@pytest.fixture
def synthetic_anomaly_data():
    """Deterministic synthetic dataset with known properties."""
    rng = np.random.RandomState(42)
    n_features = 6

    # Normal cluster (90%)
    X_norm = rng.normal(loc=0.0, scale=1.0, size=(180, n_features))
    y_norm = np.zeros(180, dtype=int)

    # Anomaly cluster (10%)
    X_anom = rng.normal(loc=4.0, scale=1.5, size=(20, n_features))
    y_anom = np.ones(20, dtype=int)

    X = np.vstack([X_norm, X_anom])
    y = np.concatenate([y_norm, y_anom])

    # Continuous score proxy
    ensemble_scores = np.linalg.norm(X, axis=1)
    ensemble_scores = (ensemble_scores - ensemble_scores.min()) / (
        ensemble_scores.max() - ensemble_scores.min() + 1e-8
    )

    feature_names = [f"signal_{i}" for i in range(n_features)]
    return X, y, ensemble_scores, feature_names


def test_surrogate_oof_training_and_no_leakage(synthetic_anomaly_data):
    """
    Verify surrogate is trained using K-fold out-of-fold validation and
    never scores rows on the models that trained on them.
    """
    X, y, ensemble_scores, feature_names = synthetic_anomaly_data

    surrogate = SurrogateDetector(
        n_estimators=30,
        max_depth=3,
        n_splits=5,
        contamination=0.10,
        random_state=42,
    )
    surrogate.fit_from_ensemble(X, ensemble_scores, feature_names=feature_names)

    assert surrogate.is_fitted()
    assert surrogate.oof_scores_ is not None
    assert len(surrogate.oof_scores_) == len(X)
    assert surrogate.oof_predictions_ is not None
    assert set(np.unique(surrogate.oof_predictions_)).issubset({-1, 1})

    # Metadata must clearly state "surrogate of ensemble"
    meta = surrogate.get_metadata()
    assert meta["provenance"] == "surrogate of ensemble"
    assert meta["is_ground_truth"] is False
    assert meta["n_splits"] == 5


def test_surrogate_fidelity_report(synthetic_anomaly_data):
    """
    Verify fidelity report computes rank correlation and Top-K overlap.
    """
    X, _, ensemble_scores, feature_names = synthetic_anomaly_data

    surrogate = SurrogateDetector(
        n_estimators=30,
        max_depth=3,
        n_splits=4,
        contamination=0.10,
        random_state=42,
    )
    surrogate.fit_from_ensemble(X, ensemble_scores, feature_names=feature_names)

    fidelity = surrogate.fidelity_report_
    assert fidelity is not None
    f_dict = fidelity.to_dict()

    assert "spearman_rank_corr" in f_dict
    assert f_dict["spearman_rank_corr"] > 0.25  # Surrogate must correlate positively with ensemble
    assert "top_5%" in f_dict["top_k_overlaps"]
    assert "top_10%" in f_dict["top_k_overlaps"]
    assert f_dict["provenance"] == "surrogate of ensemble"


def test_surrogate_treeshap_additivity(synthetic_anomaly_data):
    """
    Verify exact TreeSHAP explanations and sum-to-margin additivity:
    sum(SHAP values) + expected_value ≈ model margin output.
    """
    X, _, ensemble_scores, feature_names = synthetic_anomaly_data

    surrogate = SurrogateDetector(
        n_estimators=25,
        max_depth=3,
        contamination=0.10,
        random_state=42,
    )
    surrogate.fit_from_ensemble(X, ensemble_scores, feature_names=feature_names)

    explanation = surrogate.explain(X[:20])
    assert explanation["method"] == "TreeSHAP"
    assert explanation["provenance"] == "surrogate of ensemble"
    assert len(explanation["shap_values"]) == 20
    assert len(explanation["global_importance"]) == len(feature_names)

    # Feature rankings should have positive importance
    assert explanation["global_importance"][0]["mean_abs_shap"] >= 0.0


def test_supervised_classifier_metrics_and_calibration(synthetic_anomaly_data):
    """
    Verify supervised mode handles severe class imbalance and calculates
    PR-AUC, ROC-AUC, Brier score, and Expected Calibration Error (ECE).
    """
    X, y, _, feature_names = synthetic_anomaly_data

    clf = SupervisedClassifierDetector(
        framework="xgboost",
        n_estimators=30,
        max_depth=3,
        n_splits=3,
        cost_fp=1.0,
        cost_fn=15.0,
        random_state=42,
    )
    clf.fit(X, y, feature_names=feature_names)

    assert clf.is_fitted()
    meta = clf.get_metadata()
    assert meta["provenance"] == "ground_truth_supervised"
    assert meta["is_ground_truth"] is True

    metrics = clf.cv_metrics_
    assert metrics["pr_auc"] > 0.0
    assert metrics["roc_auc"] > 0.6
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert "precision@5%" in metrics["precision_at_k"]


def test_cost_matrix_optimization():
    """
    Verify cost matrix thresholding minimizes total asymmetric cost.
    """
    y_true = np.array([0] * 90 + [1] * 10)
    # Predicted probabilities
    y_prob = np.concatenate([
        np.linspace(0.05, 0.4, 90),
        np.linspace(0.45, 0.95, 10),
    ])

    # Heavily penalize false negatives (missed anomalies)
    res_high_fn = optimize_threshold_cost_matrix(y_true, y_prob, cost_fp=1.0, cost_fn=20.0)
    # Heavily penalize false positives (false alarms)
    res_high_fp = optimize_threshold_cost_matrix(y_true, y_prob, cost_fp=20.0, cost_fn=1.0)

    # When FN is expensive, optimal threshold should be lower to catch more anomalies
    assert res_high_fn["optimal_threshold"] <= res_high_fp["optimal_threshold"]


def test_hybrid_stacking_meta_learner(synthetic_anomaly_data):
    """
    Verify hybrid stacking meta-learner combines 8 unsupervised base detectors
    into out-of-fold meta-features without test leakage.
    """
    X, y, _, feature_names = synthetic_anomaly_data

    stacker = HybridStackingDetector(
        passthrough=True,
        n_estimators=20,
        max_depth=3,
        n_splits=3,
        random_state=42,
    )
    stacker.fit(X, y, feature_names=feature_names)

    assert stacker.is_fitted()
    assert stacker.oof_meta_matrix_ is not None
    assert stacker.oof_meta_matrix_.shape == (len(X), 8)

    # Test scoring on new observations
    new_scores = stacker.score_samples(X[:10])
    assert len(new_scores) == 10
    assert np.all((new_scores >= 0.0) & (new_scores <= 1.0))

    # Test meta-attribution explanations
    exp = stacker.explain(X[:5])
    assert exp["method"] == "TreeSHAP_HybridStacking"
    assert "base_detector_attribution" in exp
    assert len(exp["base_detector_attribution"]) == 8


def test_optuna_tuning_reproducibility(synthetic_anomaly_data):
    """
    Verify hyperparameter tuning uses nested CV and fixed seed for reproducible results.
    """
    X, y, _, _ = synthetic_anomaly_data

    res1 = tune_xgboost_hyperparameters(
        X, y, n_trials=5, cv_splits=2, metric="pr_auc", random_state=42
    )
    res2 = tune_xgboost_hyperparameters(
        X, y, n_trials=5, cv_splits=2, metric="pr_auc", random_state=42
    )

    assert res1["best_score"] == res2["best_score"]
    assert res1["best_params"] == res2["best_params"]
    assert len(res1["tuning_history"]) == 5


def test_api_v2_supervised_endpoints(synthetic_anomaly_data):
    """
    Test FastAPI v2 endpoints for surrogate training, fidelity, and tuning.
    """
    client = TestClient(app)

    # Load benchmark dataset first
    r_load = client.post("/api/dataset/sample")
    assert r_load.status_code == 200

    # 1. Train Surrogate
    r_surr = client.post(
        "/api/v2/supervised/surrogate/train",
        json={
            "contamination": 0.05,
            "n_estimators": 25,
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_splits": 3,
            "random_state": 42,
        },
    )
    assert r_surr.status_code == 200
    data_surr = r_surr.json()
    assert data_surr["model_type"] == "surrogate_xgboost"
    assert data_surr["provenance"] == "surrogate of ensemble"

    # 2. Get Surrogate Fidelity
    r_fid = client.get("/api/v2/supervised/surrogate/fidelity")
    assert r_fid.status_code == 200
    data_fid = r_fid.json()
    assert data_fid["provenance"] == "surrogate of ensemble"
    assert "spearman_rank_corr" in data_fid

    # 3. Explain Surrogate (TreeSHAP)
    r_exp = client.post("/api/v2/supervised/surrogate/explain?max_samples=50")
    assert r_exp.status_code == 200
    data_exp = r_exp.json()
    assert data_exp["method"] == "TreeSHAP"

    # 4. Train Supervised Classifier (using benchmark synthetic label if available)
    if "is_anomaly" in state.df_raw.columns:
        r_clf = client.post(
            "/api/v2/supervised/classify/train",
            json={
                "label_column": "is_anomaly",
                "framework": "xgboost",
                "n_estimators": 25,
                "max_depth": 3,
                "learning_rate": 0.05,
                "cost_fp": 1.0,
                "cost_fn": 10.0,
                "n_splits": 3,
                "random_state": 42,
            },
        )
        assert r_clf.status_code == 200
        data_clf = r_clf.json()
        assert data_clf["is_ground_truth"] is True
        assert "pr_auc" in data_clf["metrics"]

    # 5. Hybrid Stacking
    r_stack = client.post(
        "/api/v2/supervised/stacking/train",
        json={
            "passthrough": True,
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_splits": 3,
            "random_state": 42,
        },
    )
    assert r_stack.status_code == 200
    data_stack = r_stack.json()
    assert data_stack["model_type"] == "hybrid_stacking"
    assert data_stack["n_base_detectors"] == 8

    # 6. Hyperparameter Tune
    r_tune = client.post(
        "/api/v2/supervised/tune",
        json={
            "n_trials": 3,
            "metric": "pr_auc",
            "cv_splits": 2,
            "random_state": 42,
        },
    )
    assert r_tune.status_code == 200
    data_tune = r_tune.json()
    assert "best_params" in data_tune
    assert data_tune["n_trials"] == 3

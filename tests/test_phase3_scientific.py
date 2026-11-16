"""
Regression tests for the Phase 3 scientific-validity fixes.

Every test encodes a specific measurement that failed before the fix, so a
regression is caught as a wrong *number*, not just an exception.
"""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.evaluation.synthetic_lab as synthetic_lab
from backend import state as state_module
from src.detection.ensemble import IntelligentEnsembleEngine
from src.evaluation.synthetic_lab import SyntheticInjectionLab
from src.models.gmm_detector import GMMDetector
from src.models.hbos_detector import HBOSDetector
from src.models.knn_detector import KNNDetector
from src.models.model_manager import ModelManager


@pytest.fixture
def client():
    from backend.main import app

    state_module.state.reset_all()
    with TestClient(app) as c:
        yield c
    state_module.state.reset_all()


@pytest.fixture
def correlated_data():
    rng = np.random.default_rng(42)
    z = rng.normal(0, 1, size=(300, 2))
    X = np.zeros((300, 4))
    X[:, :2] = z
    X[:, 2] = z[:, 0] * 1.5 + z[:, 1] + rng.normal(0, 0.02, 300)
    X[:, 3] = z[:, 0] - z[:, 1] * 2.0 + rng.normal(0, 0.02, 300)
    return X


# ── Out-of-sample benchmark ──────────────────────────────────────────────────

def test_benchmark_holds_injected_rows_out_of_training(correlated_data):
    """
    Detectors used to be fitted on ``X_bench`` and scored on the *same* rows,
    which is why every detector reported ROC-AUC near 1.0.  The fit split must
    now be disjoint from the evaluated rows, and no injected row may train the
    model that is judged on it.
    """
    lab = SyntheticInjectionLab(random_state=42)
    report = lab.benchmark_detectors(
        correlated_data,
        anomaly_type="point",
        contamination=0.08,
        models_to_test=["Isolation Forest"],
    )

    assert report["evaluation_protocol"].startswith("out-of-sample")
    assert report["n_fit"] + report["n_evaluated"] == report["n_samples"]
    assert report["n_evaluated"] >= report["n_injected"]

    # The fit split is a fraction of the *clean* rows, so it must be strictly
    # smaller than the clean count when injected rows are removed.
    clean_count = report["n_samples"] - report["n_injected"]
    assert report["n_fit"] < clean_count
    assert report["n_fit"] == round(clean_count * 0.6)


def test_benchmark_records_the_evaluated_class_balance(correlated_data):
    """PR-AUC is only interpretable with the prevalence it was measured at."""
    lab = SyntheticInjectionLab(random_state=42)
    report = lab.benchmark_detectors(
        correlated_data, anomaly_type="point", contamination=0.08,
        models_to_test=["Isolation Forest"],
    )
    assert 0.0 < report["evaluation_contamination_rate"] < 1.0
    assert report["contamination_rate"] == pytest.approx(0.08, abs=1e-6)


# ── Injection quality ────────────────────────────────────────────────────────

def test_multivariate_injection_breaks_dependence_not_marginals(correlated_data):
    """
    The old code permuted values only *within* the injected block, leaving much
    of the correlation structure intact (a weak anomaly).  The injected rows
    must lose the dependence while every value stays inside its column's
    observed range.
    """
    lab = SyntheticInjectionLab(random_state=42)
    out = lab.inject_anomalies(
        correlated_data, anomaly_type="multivariate", contamination=0.10
    )
    mask = out["injected_mask"]
    X_inj = out["X_injected"]

    clean_corr = abs(np.corrcoef(
        correlated_data[~mask, 0], correlated_data[~mask, 2]
    )[0, 1])
    injected_corr = abs(np.corrcoef(X_inj[mask, 0], X_inj[mask, 2])[0, 1])

    assert clean_corr > 0.8
    assert injected_corr < 0.35

    for j in range(correlated_data.shape[1]):
        lo, hi = correlated_data[:, j].min(), correlated_data[:, j].max()
        assert X_inj[mask, j].min() >= lo - 1e-9
        assert X_inj[mask, j].max() <= hi + 1e-9


def test_contextual_injection_exists_and_violates_the_conditional(correlated_data):
    """
    ``anomaly_type="contextual"`` used to raise ValueError ("unknown type"),
    even though the module docstring advertised the mechanism.
    """
    lab = SyntheticInjectionLab(random_state=42)
    out = lab.inject_anomalies(
        correlated_data, anomaly_type="contextual", contamination=0.10
    )
    mask = out["injected_mask"]
    X_inj = out["X_injected"]

    assert out["n_injected"] == 30

    # Values stay inside the marginal support ...
    for j in range(correlated_data.shape[1]):
        assert X_inj[mask, j].min() >= correlated_data[:, j].min() - 1e-9
        assert X_inj[mask, j].max() <= correlated_data[:, j].max() + 1e-9

    # ... but break the fitted conditional relation on at least one feature.
    broken = 0
    for j in range(correlated_data.shape[1]):
        if np.allclose(X_inj[mask, j], correlated_data[mask, j]):
            continue
        broken += 1
    assert broken >= 1


@pytest.mark.parametrize("difficulty, factor", [("easy", 1.5), ("medium", 1.0), ("hard", 0.5)])
def test_difficulty_levels_scale_the_displacement(difficulty, factor):
    lab = SyntheticInjectionLab(random_state=7)
    X = np.random.default_rng(0).normal(size=(200, 3))
    out = lab.inject_anomalies(
        X, anomaly_type="point", contamination=0.1, severity=4.0,
        difficulty=difficulty,
    )
    assert out["severity_effective"] == pytest.approx(4.0 * factor)


def test_unknown_difficulty_is_rejected():
    lab = SyntheticInjectionLab()
    with pytest.raises(ValueError, match="difficulty"):
        lab.inject_anomalies(np.zeros((50, 2)), difficulty="impossible")


# ── Contamination propagation ───────────────────────────────────────────────

def test_benchmark_propagates_contamination_to_each_detector(correlated_data, monkeypatch):
    """``contamination`` was never passed to ``manager.fit``: every detector
    silently ran with its own default 0.05 budget."""
    captured: dict[str, dict] = {}

    class SpyManager(ModelManager):
        def fit(self, model_name, X, feature_names=None, params=None):  # noqa: N803
            captured[model_name] = dict(params or {})
            return super().fit(model_name, X, feature_names=feature_names, params=params)

    monkeypatch.setattr(synthetic_lab, "ModelManager", SpyManager)

    lab = SyntheticInjectionLab(random_state=42)
    lab.benchmark_detectors(
        correlated_data,
        anomaly_type="point",
        contamination=0.15,
        models_to_test=["Isolation Forest", "One-Class SVM"],
    )

    assert captured["Isolation Forest"]["contamination"] == pytest.approx(0.15)
    assert captured["One-Class SVM"]["nu"] == pytest.approx(0.15)


# ── HBOS out-of-range scoring ───────────────────────────────────────────────

def test_hbos_scores_values_beyond_the_fitted_range_as_extreme():
    """
    ``np.digitize`` folded out-of-range values into the first/last bin, so a
    point 50 standard deviations away scored like a typical edge-bin row and
    was never flagged.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    detector = HBOSDetector(n_bins=10, contamination=0.05).fit(X)

    inside = np.zeros((1, 2))
    far_outside = np.full((1, 2), 50.0)

    score_inside = detector.score_samples(inside)[0]
    score_outside = detector.score_samples(far_outside)[0]

    assert score_outside < score_inside
    assert score_outside < detector.score_samples(X).min()


# ── Borda on clipped scores ─────────────────────────────────────────────────

def test_borda_ranks_use_raw_scores_not_clipped_values():
    """
    Rank aggregation was computed on quantile-normalised scores, which clip at
    the 1/99 percentiles: the ten most extreme rows all became 1.0 and received
    identical Borda credit.  Ranks must come from the raw ordering.
    """
    base = np.linspace(0.0, 1.0, 100)
    tails = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    scores = np.concatenate([base, tails])
    predictions = np.where(scores > 1.0, -1, 1)

    engine = IntelligentEnsembleEngine(aggregation_method="rank_borda")
    result = engine.run_ensemble(
        {"Detector": {"scores": scores, "predictions": predictions}}
    )

    ensemble = np.asarray(result["ensemble_scores"])
    tail = ensemble[-10:]
    assert np.all(np.diff(tail) > 0), tail


# ── k-NN self-neighbour ─────────────────────────────────────────────────────

def test_knn_does_not_use_the_point_as_its_own_neighbour():
    """
    For training rows the nearest neighbour is the row itself at distance 0, so
    every distance was one rank too small and the training scores used for the
    contamination threshold were systematically too normal.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    detector = KNNDetector(n_neighbors=1, contamination=0.2).fit(X)
    scores = detector.score_samples(X)

    # With the self-match removed the outlier is the only distinct score.
    assert np.ptp(scores) > 0.0
    assert scores[-1] == scores.min()
    assert scores[0] == pytest.approx(-1.0)


def test_knn_duplicate_row_is_not_scored_as_an_outlier():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    detector = KNNDetector(n_neighbors=10, contamination=0.05).fit(X)

    scores = detector.score_samples(X)
    duplicate_scores = detector.score_samples(np.vstack([X, X[0]]))

    # The appended copy of X[0] is closer to the training set than X[0] itself
    # (its own copy sits at distance 0), so it must not be scored as more
    # anomalous.  With the old self-inclusive search both values collapsed to
    # the same self-distance and this ordering carried no information.
    assert duplicate_scores[-1] >= scores[0] - 1e-9


# ── GMM minimum sample size ─────────────────────────────────────────────────

def test_gmm_rejects_a_sample_too_small_for_full_covariance():
    """
    ``actual_k = min(n_components, n_samples)`` let a 5-component full-covariance
    GMM be fitted to 8 rows in 10 dimensions and report success.  The estimate is
    singular and its "log-likelihood" is numerical noise.
    """
    X = np.random.default_rng(0).normal(size=(8, 10))
    detector = GMMDetector(n_components=5, covariance_type="full")

    with pytest.raises(ValueError, match="at least"):
        detector.fit(X)
    assert detector.is_fitted() is False


def test_gmm_still_fits_when_there_are_enough_rows():
    X = np.random.default_rng(0).normal(size=(200, 3))
    detector = GMMDetector(n_components=3, covariance_type="full").fit(X)
    assert detector.is_fitted()
    assert detector.score_samples(X).shape == (200,)


# ── Negative optimism gap ───────────────────────────────────────────────────

def test_negative_optimism_gap_is_reported_as_pessimistic_not_agreement(client, monkeypatch):
    """
    A gap of -0.4370 (GMM) used to be summarised as "in-sample and held-out
    ROC-AUC agree closely", because only ``gap <= 0.02`` was tested.
    """
    client.post("/api/dataset/sample")
    client.post("/api/detection/run", json={"model_name": "Isolation Forest"})

    fake = {
        "available": True,
        "in_sample": {"roc_auc": 0.4849},
        "out_of_sample": {"roc_auc": 0.9219},
        "optimism_gap": {"roc_auc": -0.4370},
        "reason": None,
    }
    monkeypatch.setattr(
        importlib.import_module("backend.services.evaluation"),
        "evaluate_holdout_split",
        lambda *a, **k: dict(fake),
    )

    report = state_module.state.evaluate_holdout_metrics()
    assert "pessimistic" in report["summary"].lower()
    assert "agree" not in report["summary"].lower()


# ── Stability for deterministic detectors ───────────────────────────────────

def test_stability_runs_for_a_detector_without_random_state(client):
    """
    ``params["random_state"] = seed`` raised TypeError for LOF / k-NN / HBOS /
    One-Class SVM; the loop swallowed it and stability reported
    "need at least two successful fits" forever.
    """
    client.post("/api/dataset/sample")
    client.post("/api/detection/run", json={"model_name": "Isolation Forest"})

    result = state_module.state.explanation_stability(
        seeds=[11, 23], model_name="k-Nearest Neighbors"
    )

    assert result["n_fits"] >= 2, result.get("errors")
    assert result["errors"] == []
    assert "bootstrap" in result["refit_strategy"]
    assert result["mean_rank_correlation"] is not None

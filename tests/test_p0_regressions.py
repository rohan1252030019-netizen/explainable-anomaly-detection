"""
Regression tests for the P0 defects fixed in the recovery pass.

Each test targets one specific failure mode that was observed on the repository
before the fix, not a generic "does it run" smoke check.  The comment above each
test names the defect and the measurement that exposed it.
"""

import importlib
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import state as state_module
from src.data.labels import to_binary_labels
from src.evaluation.holdout import evaluate_holdout_split
from src.explainability.counterfactuals import nearest_normal_counterfactual
from src.explainability.validation import faithfulness

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """A TestClient over a freshly reset singleton state."""
    from backend.main import app

    state_module.state.reset_all()
    with TestClient(app) as c:
        yield c
    state_module.state.reset_all()


@pytest.fixture
def loaded_client(client):
    assert client.post("/api/dataset/sample").status_code == 200
    assert client.post(
        "/api/detection/run",
        json={"model_name": "Isolation Forest", "contamination": 0.05},
    ).status_code == 200
    return client


def _csv_bytes(n_rows: int = 24) -> bytes:
    rng = np.random.default_rng(7)
    lines = ["f1,f2,f3,label"]
    for i in range(n_rows):
        a, b, c = rng.normal(size=3).round(4)
        lines.append(f"{a},{b},{c},{int(i % 7 == 0)}")
    return ("\n".join(lines) + "\n").encode()


# ── P0-1: upload used a method that does not exist ────────────────────────────

def test_csv_upload_is_processed(client):
    """
    ``POST /api/dataset/upload`` returned HTTP 500
    "Failed to process CSV: 'DataLoader' object has no attribute 'load_bytes'"
    on every valid upload, because the real method is ``load_from_bytes``.
    """
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("upload.csv", io.BytesIO(_csv_bytes()), "text/csv")},
    )
    assert response.status_code == 200, response.text
    dataset = response.json()["dataset"]
    assert dataset["loaded"] is True
    assert dataset["n_rows"] == 24
    assert dataset["feature_matrix_ready"] is True


# ── P0-2: every ensemble member was mapped through the active model's anchors ─

def test_ensemble_normalises_each_detector_with_its_own_anchors(client, monkeypatch):
    """
    Before the fix, LOF scores were 100% clipped to 1.0, One-Class SVM 100% to
    0.0 and Elliptic Envelope 100% to 1.0, because the active model's
    quantile anchors were applied to raw scores from a different scale.  The
    "ensemble" was therefore a weighted vote of constants.

    The spy captures the per-detector vectors actually handed to the combiner:
    none of them may be degenerate or pinned to an edge.
    """
    client.post("/api/dataset/sample")

    captured: dict[str, np.ndarray] = {}
    # ``ensemble_scores`` is resolved at call time from the module that owns the
    # detection service, which is where this symbol lives after the state split.
    detection_service = importlib.import_module("backend.services.detection")
    real_combine = detection_service.ensemble_scores

    def spy(score_dict, weights=None):
        for name, arr in score_dict.items():
            captured[name] = np.asarray(arr, dtype=float)
        return real_combine(score_dict, weights)

    monkeypatch.setattr(detection_service, "ensemble_scores", spy)

    response = client.post(
        "/api/detection/run",
        json={"model_name": "Isolation Forest", "use_ensemble": True},
    )
    assert response.status_code == 200, response.text
    assert len(captured) >= 2, captured.keys()

    for name, scores in captured.items():
        assert np.ptp(scores) > 0.05, f"{name} scores collapsed to a constant"
        assert scores.min() < 0.5 < scores.max(), (
            f"{name} scores collapsed onto an edge: "
            f"min={scores.min():.3f} max={scores.max():.3f}"
        )


# ── P0-6: normaliser and threshold were re-fitted on the held-out batch ───────

class _FirstFeatureDetector:
    """Deterministic detector: lower score = more anomalous (sklearn sign)."""

    def fit(self, X):  # noqa: N803 - mirrors the sklearn API
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def score_samples(self, X):  # noqa: N803
        return -np.asarray(X, dtype=float)[:, 0]

    def predict(self, X):  # noqa: N803
        return np.ones(np.asarray(X).shape[0], dtype=int)


def test_holdout_normaliser_and_threshold_come_from_the_fit_rows():
    """
    ``norm_fit`` and ``norm_test`` were each normalised with their own batch
    quantiles, and the threshold was recomputed per batch, so the held-out flag
    rate was pinned to ``contamination`` by construction.  Anchors and cut must
    now come from the fit split only.
    """
    rng = np.random.default_rng(0)
    X_fit = rng.normal(0.0, 1.0, size=(120, 2))
    X_test = rng.normal(6.0, 0.5, size=(60, 2))  # a clear distribution shift

    report = evaluate_holdout_split(
        _FirstFeatureDetector, X_fit, X_test
    )

    # The normaliser works in "higher = more anomalous" space, which for this
    # detector is simply the first feature.
    ref_fit = X_fit[:, 0]
    expected_offset = float(np.quantile(ref_fit, 0.01))
    expected_scale = float(np.quantile(ref_fit, 0.99)) - expected_offset

    assert report["score_normalization"]["offset"] == pytest.approx(
        expected_offset, rel=1e-9
    )
    assert report["score_normalization"]["scale"] == pytest.approx(
        expected_scale, rel=1e-9
    )
    # Anchors fitted on the held-out batch would give a visibly different offset.
    assert not report["score_normalization"]["offset"] == pytest.approx(
        float(np.quantile(X_test[:, 0], 0.01)), rel=1e-2
    )

    # A threshold derived from the fit rows does not force the held-out rate.
    norm_fit = np.clip((ref_fit - expected_offset) / expected_scale, 0.0, 1.0)
    assert report["threshold_value"] == pytest.approx(
        float(np.percentile(norm_fit, 95.0)), abs=1e-6
    )
    assert report["flag_rate_out_of_sample"] > report["flag_rate_in_sample"]


# ── P0-3: the preprocessor was fitted on the full frame before splitting ─────

def test_holdout_pipeline_is_fitted_on_fit_rows_only(loaded_client, monkeypatch):
    """
    The scaler / imputer / encoder were fitted on all 880 rows and the split
    happened afterwards inside ``evaluate_holdout``.  The held-out rows had
    already contributed their statistics.  Assert the pipeline only ever sees
    the fit frame.
    """
    from src.data.preprocessing import PreprocessingPipeline

    fitted_lengths: list[int] = []
    real_fit_transform = PreprocessingPipeline.fit_transform

    def spy(self, df):
        fitted_lengths.append(len(df))
        return real_fit_transform(self, df)

    monkeypatch.setattr(PreprocessingPipeline, "fit_transform", spy)

    state = state_module.state
    n_total = len(state.df_raw)
    inputs = state._holdout_inputs(test_size=0.3, random_state=42)

    assert fitted_lengths == [len(inputs["X_fit"])]
    assert fitted_lengths[0] < n_total
    assert inputs["X_fit"].shape[0] + inputs["X_test"].shape[0] == n_total
    assert inputs["preprocessing"].startswith("preprocessor fitted on the fit split")


# ── P0-4: counterfactuals mixed scaled deltas with raw values ────────────────

def test_counterfactual_reports_the_neighbours_real_value():
    """
    ``orig[i] + delta[i]`` adds a standardised delta to a raw value: with
    raw = 10 * z + 100, an observation at z=2 (raw 120) and a neighbour at z=0
    (raw 100), the old code reported 118 instead of 100.
    """
    observation = np.array([2.0])
    reference = np.array([[0.0], [10.0]])
    denormalize = lambda z: np.asarray(z, dtype=float) * 10.0 + 100.0  # noqa: E731

    result = nearest_normal_counterfactual(
        observation,
        reference,
        feature_names=["amount"],
        denormalize=denormalize,
    )

    assert result["found"] is True
    change = result["changes"][0]
    assert change["feature"] == "amount"
    assert change["observed_value"] == pytest.approx(120.0)
    assert change["suggested_value"] == pytest.approx(100.0)
    assert change["change_pct"] == pytest.approx(-16.67, abs=0.01)


# ── P0-5: explained and random deletion curves had different lengths ─────────

def test_faithfulness_curves_cover_the_same_feature_set():
    """
    The explained curve had len(top_features) + 1 points while the random
    baseline always had n_features + 1.  Integrating two different x-grids made
    a perfect explanation score 0.3018 against a random 0.6078 and get labelled
    "Counter-indicative".  Both curves must ablate every feature.
    """
    def score(row):
        return float(np.exp(-((row[0] - 5.0) ** 2)))

    observation = np.array([5.0, 1.0, 1.0, 1.0, 1.0])
    result = faithfulness(observation, score, [0], n_random=5)

    assert len(result["curve"]) == 6
    assert len(result["insertion_curve"]) == 6
    assert result["n_features_ablated"] == 5
    assert result["relative_improvement"] < 0
    assert result["verdict"].startswith("Strong")


def test_top_k_retention_is_not_identically_one():
    """
    ``top_k_retention`` intersected each top-k set with the *union* of all
    top-k sets, so the ratio was 1.0 for any input, including pure noise.
    """
    from src.explainability.validation import explanation_stability

    rng = np.random.default_rng(3)
    noise = rng.random((6, 8))
    result = explanation_stability(noise, [f"f{i}" for i in range(8)], top_k=3)

    assert 0.0 <= result["top_k_retention"] < 1.0


# ── P0-7: {-1, +1} labels were read backwards ────────────────────────────────

def test_label_mapping_handles_both_conventions():
    assert to_binary_labels(np.array([-1, 1, -1, 1])).tolist() == [1, 0, 1, 0]
    assert to_binary_labels(np.array([0, 1, 0, 1])).tolist() == [0, 1, 0, 1]
    assert to_binary_labels(np.array([0, 1, 2])) is None


def test_state_reads_negative_one_labels_as_anomalies(client):
    """
    ``state.preprocess_dataset`` used ``labels == 1``, which selects the normal
    rows from a {-1, +1} column and inverts every downstream metric.
    """
    import pandas as pd

    state = state_module.state
    rng = np.random.default_rng(11)
    n = 40
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    df["target"] = np.where(np.arange(n) % 8 == 0, -1, 1)
    state.df_raw = df
    state.dataset_name = "neg1.csv"

    state.preprocess_dataset(label_col="target")

    assert state.y_true is not None
    assert int(state.y_true.sum()) == int((np.arange(n) % 8 == 0).sum())
    assert state.y_true[np.arange(n) % 8 == 0].all()
    state.reset_all()


# ── P0-8: the ensemble wrote a non-registry name into active_model ───────────

def test_ensemble_does_not_break_later_endpoints(client):
    """
    ``run_intelligent_ensemble`` set ``active_model = "Ensemble (2 models)"``.
    The next holdout call resolved that through ``DETECTOR_REGISTRY`` and failed
    with ``Unknown model 'Ensemble (2 models)'`` (HTTP 400).  Detection still
    happened, so the failure surfaced later and looked unrelated.
    """
    client.post("/api/dataset/sample")
    response = client.post(
        "/api/detection/ensemble",
        json={"models": ["Isolation Forest", "Local Outlier Factor"]},
    )
    assert response.status_code == 200, response.text

    state = state_module.state
    from src.models.model_manager import DETECTOR_REGISTRY

    assert state.active_model in DETECTOR_REGISTRY
    assert state.display_model_name.startswith("Intelligent Ensemble")

    holdout = client.get("/api/evaluation/holdout")
    assert holdout.status_code == 200, holdout.text
    assert holdout.json()["available"] is True

    health = client.get("/api/health").json()
    assert health["active_model"].startswith("Intelligent Ensemble")

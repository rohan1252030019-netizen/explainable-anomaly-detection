"""Tests for the detector implementations and the model manager."""

import numpy as np
import pytest

from src.models.base_detector import BaseDetector
from src.models.elliptic_envelope import EllipticEnvelopeDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.local_outlier_factor import LOFDetector
from src.models.model_manager import DETECTOR_REGISTRY, ModelManager
from src.models.one_class_svm import OneClassSVMDetector

ALL_DETECTORS = [
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
    EllipticEnvelopeDetector,
]


@pytest.fixture
def toy_X():
    """A cluster with three deliberately planted outliers."""
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(300, 4))
    X[0] = 12.0
    X[1] = -12.0
    X[2] = np.array([12.0, 12.0, 12.0, 12.0])
    return X


class TestBaseDetectorContract:
    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_cannot_score_before_fit(self, cls, toy_X):
        detector = cls()
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.score_samples(toy_X)

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_fit_then_score(self, cls, toy_X):
        detector = cls()
        detector.fit(toy_X)
        scores = detector.score_samples(toy_X)
        assert scores.shape == (toy_X.shape[0],)
        assert np.all(np.isfinite(scores))

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_predict_uses_sklearn_convention(self, cls, toy_X):
        detector = cls()
        detector.fit(toy_X)
        preds = detector.predict(toy_X)
        assert set(np.unique(preds)).issubset({-1, 1})

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_planted_outliers_score_as_more_anomalous(self, cls, toy_X):
        detector = cls()
        detector.fit(toy_X)
        scores = detector.score_samples(toy_X)
        # sklearn convention: lower raw score == more anomalous.
        assert scores[:3].mean() < np.median(scores)

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_rejects_nan_input(self, cls, toy_X):
        bad = toy_X.copy()
        bad[0, 0] = np.nan
        detector = cls()
        with pytest.raises(ValueError, match="NaN"):
            detector.fit(bad)

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_rejects_non_2d_input(self, cls):
        detector = cls()
        with pytest.raises(ValueError, match="2-D"):
            detector.fit(np.zeros(10))

    @pytest.mark.parametrize("cls", ALL_DETECTORS)
    def test_model_info_is_populated(self, cls, toy_X):
        detector = cls()
        detector.fit(toy_X)
        info = detector.get_model_info()
        assert info["is_fitted"] is True
        assert info["n_train_samples"] == toy_X.shape[0]
        assert isinstance(info["params"], dict)

    def test_base_detector_is_abstract(self):
        with pytest.raises(TypeError):
            BaseDetector()  # type: ignore[abstract]

    def test_lof_supports_both_novelty_modes(self, toy_X):
        """
        The LOF wrapper papers over sklearn's novelty=False limitation by
        refitting a temporary novelty model, so scoring unseen rows works in
        either mode; only the reported capability flag differs.
        """
        for novelty in (True, False):
            detector = LOFDetector(novelty=novelty)
            detector.fit(toy_X)
            scores = detector.score_samples(toy_X[:10])
            assert scores.shape == (10,)
            assert np.all(np.isfinite(scores))
            assert detector.get_model_info()["params"]["novelty"] is novelty

    def test_lof_defaults_to_novelty_for_holdout_use(self):
        assert LOFDetector().novelty is True

    def test_elliptic_envelope_reports_fit_failure_instead_of_crashing(self):
        """Non-Gaussian data can make the covariance estimate singular."""
        rng = np.random.default_rng(1)
        weird = rng.standard_cauchy(size=(50, 3))
        detector = EllipticEnvelopeDetector(contamination=0.05)
        detector.fit(weird)
        info = detector.get_model_info()
        assert "is_fitted" in info  # did not raise


class TestModelManager:
    def test_registry_covers_every_detector(self):
        for cls in ALL_DETECTORS:
            assert cls in DETECTOR_REGISTRY.values()

    def test_unknown_model_raises(self, tmp_path):
        manager = ModelManager(models_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown model"):
            manager.create_detector("Does Not Exist")

    def test_fit_records_timing_and_counts(self, tmp_path, toy_X):
        manager = ModelManager(models_dir=str(tmp_path))
        result = manager.fit("Isolation Forest", toy_X, ["a", "b", "c", "d"])
        assert result["success"] is True
        assert result["n_anomalies"] is not None
        assert result["fit_time_sec"] >= 0

    def test_fit_all_reports_each_model(self, tmp_path, toy_X):
        manager = ModelManager(models_dir=str(tmp_path))
        results = manager.fit_all(toy_X, ["a", "b", "c", "d"])
        assert set(results) == set(DETECTOR_REGISTRY)
        # A failure for one model must not abort the others.
        assert any(r.get("success") for r in results.values())

    def test_fit_all_can_skip_elliptic_envelope(self, tmp_path, toy_X):
        manager = ModelManager(models_dir=str(tmp_path))
        results = manager.fit_all(toy_X, ["a", "b", "c", "d"],
                                  include_elliptic=False)
        assert "Elliptic Envelope" not in results

    def test_score_before_fit_raises(self, tmp_path, toy_X):
        manager = ModelManager(models_dir=str(tmp_path))
        with pytest.raises(KeyError):
            manager.score("Isolation Forest", toy_X)

    def test_save_and_load_round_trip(self, tmp_path, toy_X):
        manager = ModelManager(models_dir=str(tmp_path))
        manager.fit("Isolation Forest", toy_X, ["a", "b", "c", "d"])
        path = manager.save_model("Isolation Forest")
        assert path.endswith(".joblib")

        fresh = ModelManager(models_dir=str(tmp_path))
        fresh.load_model("Isolation Forest")
        np.testing.assert_allclose(
            fresh.score("Isolation Forest", toy_X),
            manager.score("Isolation Forest", toy_X),
        )

    def test_load_missing_model_raises(self, tmp_path):
        manager = ModelManager(models_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            manager.load_model("Isolation Forest")

    def test_experiment_log_accumulates(self, tmp_path, toy_X):
        manager = ModelManager(models_dir=str(tmp_path))
        manager.fit("Isolation Forest", toy_X, ["a", "b", "c", "d"])
        manager.fit("One-Class SVM", toy_X, ["a", "b", "c", "d"])
        assert len(manager.get_experiments()) == 2

    def test_factory_builds_independent_instances(self, tmp_path, toy_X):
        """Needed so a holdout fit cannot leak the training rows."""
        manager = ModelManager(models_dir=str(tmp_path))
        a = manager.create_detector("Isolation Forest")
        b = manager.create_detector("Isolation Forest")
        assert a is not b
        assert not a.is_fitted() and not b.is_fitted()

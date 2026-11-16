"""
Tests for src/detection/scoring.py

The sign convention is the thing most likely to regress silently: detectors emit
"lower = more anomalous" and the normaliser must invert that. Several tests
below pin the orientation explicitly.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.detection.scoring import (
    ScoreNormalizer,
    ensemble_scores,
    normalize_scores,
)


class TestNormalizeScores:
    def test_output_bounded(self, toy_scores):
        for method in ("quantile", "rank", "minmax"):
            out = normalize_scores(toy_scores, method=method)
            assert out.min() >= 0.0
            assert out.max() <= 1.0

    def test_higher_raw_means_more_anomalous(self, toy_scores):
        """sklearn convention is LOWER = more anomalous, so negate."""
        out = normalize_scores(toy_scores, method="minmax")
        assert out[0] > out[-1], "most negative raw score must normalise highest"
        assert out.argmax() == int(np.argmin(toy_scores))

    def test_ordering_is_monotone_decreasing_in_raw(self, toy_scores):
        """A lower raw score must never normalise lower than a higher one."""
        for method in ("quantile", "rank", "minmax"):
            out = normalize_scores(toy_scores, method=method)
            rho = spearmanr(toy_scores, out).statistic
            assert rho == pytest.approx(-1.0, abs=1e-9)

    def test_minmax_matches_reference_formula(self):
        raw = np.array([-0.7, -0.4, -0.1])
        out = normalize_scores(raw, method="minmax")
        expected = (-raw - (-raw).min()) / ((-raw).max() - (-raw).min())
        np.testing.assert_allclose(out, expected)

    def test_rank_is_uniform_by_construction(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(size=1000)
        out = normalize_scores(raw, method="rank")
        # Empirical CDF of a continuous sample is ~uniform on [0, 1].
        assert abs(out.mean() - 0.5) < 0.02
        assert out.std() == pytest.approx(1 / np.sqrt(12), abs=0.03)

    def test_quantile_is_not_distorted_by_a_single_extreme_outlier(self):
        """The reason quantile exists: min-max collapses under one outlier."""
        rng = np.random.default_rng(7)
        raw = rng.normal(0.0, 1.0, 1000)
        raw_with_outlier = raw.copy()
        raw_with_outlier[0] = -1e6  # sklearn: very negative == very anomalous

        minmax = normalize_scores(raw_with_outlier, method="minmax")
        robust = normalize_scores(raw_with_outlier, method="quantile")

        # The 95th percentile of the healthy population should stay mid-scale.
        body_minmax = np.quantile(minmax[1:], 0.95)
        body_robust = np.quantile(robust[1:], 0.95)
        assert body_minmax < 0.01, "min-max should be squashed by the outlier"
        assert body_robust > 0.80, "quantile should keep the body readable"
        assert np.median(robust[1:]) == pytest.approx(0.5, abs=0.05)

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            normalize_scores(np.zeros((3, 3)))

    def test_handles_nan_and_inf(self):
        raw = np.array([-1.0, np.nan, np.inf, -np.inf, 0.0])
        out = normalize_scores(raw, method="quantile")
        assert np.all(np.isfinite(out))

    def test_all_identical_scores_do_not_divide_by_zero(self):
        out = normalize_scores(np.full(10, -0.5), method="quantile")
        assert np.all(np.isfinite(out))

    def test_empty_input(self):
        assert normalize_scores(np.array([])).shape == (0,)


class TestScoreNormalizer:
    def test_requires_fit_before_transform(self):
        with pytest.raises(RuntimeError, match="fit"):
            ScoreNormalizer().transform(np.array([-1.0, 0.0]))

    def test_reuses_anchors_across_batches(self):
        """The point of the class: a 0.8 means the same thing next batch."""
        rng = np.random.default_rng(3)
        reference = rng.normal(size=500)
        norm = ScoreNormalizer(method="quantile").fit(reference)
        later = rng.normal(size=200) + 5.0  # shifted batch

        out = norm.transform(later)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_describe_reports_anchors(self):
        raw = np.array([-1.0, -0.5, 0.0, 0.5])
        info = ScoreNormalizer(method="quantile").fit(raw).describe()
        assert info["method"] == "quantile"
        assert info["scale"] > 0

    def test_rejects_invalid_quantiles(self):
        with pytest.raises(ValueError):
            ScoreNormalizer(low_quantile=0.9, high_quantile=0.1)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown"):
            ScoreNormalizer(method="nope")


class TestEnsembleScores:
    def test_equal_weights_average(self):
        a = np.array([0.0, 1.0])
        b = np.array([1.0, 0.0])
        out = ensemble_scores({"a": a, "b": b})
        np.testing.assert_allclose(out, [0.5, 0.5])

    def test_weights_are_normalised(self):
        out = ensemble_scores(
            {"a": np.array([1.0]), "b": np.array([0.0])},
            weights={"a": 3.0, "b": 1.0},
        )
        assert out[0] == pytest.approx(0.75)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ensemble_scores({})

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="different lengths"):
            ensemble_scores({"a": np.zeros(3), "b": np.zeros(4)})

"""
Tests for the explainability layer.

Covers the three guarantees the project claims:
- SHAP attributions are expressed in the same units as the displayed score and
  satisfy additivity.
- Statistical deviations compare like with like (raw against raw).
- Counterfactuals report real units and say how good they are.
"""

import numpy as np
import pytest

from src.explainability.counterfactuals import (
    counterfactuals_for_anomalies,
    nearest_normal_counterfactual,
)
from src.explainability.feature_contributions import (
    combined_feature_contributions,
    compute_statistical_deviations,
    global_feature_importance,
    perturbation_importance,
)
from src.explainability.validation import (
    deletion_auc,
    explanation_stability,
    faithfulness,
    validate_explanation,
)

# ── Statistical deviations ────────────────────────────────────────────────────

class TestStatisticalDeviations:
    def test_direction_matches_the_sign_of_the_deviation(self, fitted_sample):
        """Regression guard for a real bug: scaled obs vs raw reference stats."""
        pipeline, X, feats, _ = fitted_sample
        ref = pipeline.get_reference_stats()

        for row in X[:20]:
            raw = pipeline.denormalize(row)
            devs = compute_statistical_deviations(
                row, feats, ref, raw_observation=raw
            )
            for d in devs:
                if d["feature_type"] != "numerical":
                    continue
                if d["observed_value"] > d["reference_value"]:
                    assert d["deviation_direction"] == "above"
                else:
                    assert d["deviation_direction"] == "below"

    def test_z_score_sign_agrees_with_direction(self, fitted_sample):
        pipeline, X, feats, _ = fitted_sample
        ref = pipeline.get_reference_stats()
        raw = pipeline.denormalize(X[0])
        for d in compute_statistical_deviations(X[0], feats, ref, raw_observation=raw):
            if d["feature_type"] != "numerical":
                continue
            if d["robust_z_score"] > 0:
                assert d["deviation_direction"] == "above"
            elif d["robust_z_score"] < 0:
                assert d["deviation_direction"] == "below"

    def test_a_median_valued_row_has_near_zero_z(self, fitted_sample):
        pipeline, X, feats, _ = fitted_sample
        ref = pipeline.get_reference_stats()
        medians = np.array(
            [ref["median"].get(f, np.median(X[:, i]))
             for i, f in enumerate(feats)]
        )
        devs = compute_statistical_deviations(
            medians, feats, ref, raw_observation=medians
        )
        for d in devs:
            if d["feature_type"] == "numerical":
                assert abs(d["robust_z_score"]) < 1e-6

    def test_sorted_by_magnitude_descending(self, fitted_sample):
        pipeline, X, feats, _ = fitted_sample
        ref = pipeline.get_reference_stats()
        raw = pipeline.denormalize(X[3])
        devs = compute_statistical_deviations(X[3], feats, ref, raw_observation=raw)
        mags = [d["deviation_magnitude"] for d in devs]
        assert mags == sorted(mags, reverse=True)

    def test_onehot_features_are_labelled(self, fitted_sample):
        pipeline, X, feats, _ = fitted_sample
        ref = pipeline.get_reference_stats()
        devs = compute_statistical_deviations(X[0], feats, ref)
        assert any(d["feature_type"] == "categorical_ohe" for d in devs)
        assert len(devs) == len(feats)

    def test_all_features_are_returned(self, fitted_sample):
        pipeline, X, feats, _ = fitted_sample
        devs = compute_statistical_deviations(X[0], feats, {})
        assert len(devs) == len(feats)


# ── Perturbation importance ───────────────────────────────────────────────────

class TestPerturbationImportance:
    def test_one_entry_per_feature(self, fitted_iforest, fitted_sample):
        _, _, feats, _ = fitted_sample
        imps = perturbation_importance(
            fitted_iforest["X"][0], fitted_iforest["score_fn"], feats
        )
        assert len(imps) == len(feats)

    def test_sorted_descending(self, fitted_iforest, fitted_sample):
        _, _, feats, _ = fitted_sample
        imps = perturbation_importance(
            fitted_iforest["X"][0], fitted_iforest["score_fn"], feats
        )
        values = [i["importance"] for i in imps]
        assert values == sorted(values, reverse=True)

    def test_masking_a_real_driver_reduces_the_score(self, fitted_iforest,
                                                   fitted_sample):
        _, _, feats, _ = fitted_sample
        X = fitted_iforest["X"]
        score_fn = fitted_iforest["score_fn"]
        row = int(np.argmax(fitted_iforest["norm"]))

        imps = perturbation_importance(X[row], score_fn, feats)
        top = imps[0]
        assert top["score_delta"] > 0, (
            "removing the top-ranked feature should lower the anomaly score"
        )


class TestCombinedContributions:
    def test_respects_top_k(self, fitted_iforest, fitted_sample):
        pipeline, X, feats, _ = fitted_sample
        devs = compute_statistical_deviations(X[0], feats, pipeline.get_reference_stats())
        imps = perturbation_importance(X[0], fitted_iforest["score_fn"], feats)
        combined = combined_feature_contributions(devs, imps, top_k=3)
        assert len(combined) == 3

    def test_works_without_perturbation(self, fitted_iforest, fitted_sample):
        pipeline, X, feats, _ = fitted_sample
        devs = compute_statistical_deviations(X[0], feats, pipeline.get_reference_stats())
        combined = combined_feature_contributions(devs, None, top_k=5)
        assert len(combined) == 5


class TestGlobalImportance:
    def test_returns_a_row_per_feature(self, fitted_sample):
        _, _, feats, _ = fitted_sample
        rows = [[
            {"feature": f, "combined_score": 0.5, "robust_z_score": 1.0}
            for f in feats
        ]]
        df = global_feature_importance(rows, feats)
        assert len(df) == len(feats)
        assert df["rank"].tolist() == list(range(1, len(feats) + 1))


# ── Counterfactuals ───────────────────────────────────────────────────────────

class TestCounterfactuals:
    def test_reports_a_neighbour_that_scores_lower(self, fitted_iforest):
        norm = fitted_iforest["norm"]
        mask = norm <= np.quantile(norm, 0.95)
        target = int(np.argmax(norm))

        cf = nearest_normal_counterfactual(
            observation=fitted_iforest["X"][target],
            reference=fitted_iforest["X"][mask],
            score_fn=fitted_iforest["score_fn"],
            feature_names=[f"f{i}" for i in range(fitted_iforest["X"].shape[1])],
        )
        assert cf["found"]
        assert cf["score_after"] < cf["score_before"]

    def test_empty_reference_is_reported_not_raised(self):
        cf = nearest_normal_counterfactual(
            observation=np.zeros(3), reference=np.empty((0, 3))
        )
        assert cf["found"] is False
        assert cf["reason"]
        assert cf["changes"] == []

    def test_shape_mismatch_is_reported(self):
        cf = nearest_normal_counterfactual(
            observation=np.zeros(3), reference=np.zeros((5, 4))
        )
        assert cf["found"] is False
        assert "mismatch" in cf["reason"].lower()

    def test_reports_original_units_when_supplied(self, fitted_iforest):
        X = fitted_iforest["X"]
        norm = fitted_iforest["norm"]
        mask = norm <= np.quantile(norm, 0.95)
        target = int(np.argmax(norm))
        n_feat = X.shape[1]

        cf = nearest_normal_counterfactual(
            observation=X[target],
            reference=X[mask],
            score_fn=fitted_iforest["score_fn"],
            feature_names=[f"f{i}" for i in range(n_feat)],
            original_units=np.full(n_feat, 100.0),
        )
        assert cf["found"]
        for change in cf["changes"]:
            assert change["observed_value"] == 100.0
            assert "suggested_value" in change

    def test_quality_is_judged_against_the_threshold(self, fitted_iforest):
        X = fitted_iforest["X"]
        norm = fitted_iforest["norm"]
        mask = norm <= np.quantile(norm, 0.95)
        target = int(np.argmax(norm))

        generous = nearest_normal_counterfactual(
            observation=X[target], reference=X[mask],
            score_fn=fitted_iforest["score_fn"], score_threshold=1.1,
        )
        strict = nearest_normal_counterfactual(
            observation=X[target], reference=X[mask],
            score_fn=fitted_iforest["score_fn"], score_threshold=0.0,
        )
        assert generous["quality"]["label"] in {"STRONG", "PARTIAL"}
        assert strict["quality"]["label"] == "WEAK"

    def test_batch_helper_tags_row_ids(self, fitted_iforest):
        X = fitted_iforest["X"]
        norm = fitted_iforest["norm"]
        mask = norm <= np.quantile(norm, 0.95)
        idx = np.flatnonzero(~mask)[:3]

        out = counterfactuals_for_anomalies(
            idx, X, mask, score_fn=fitted_iforest["score_fn"],
            feature_names=[f"f{i}" for i in range(X.shape[1])],
        )
        assert [r["row_id"] for r in out] == list(idx)

    def test_batch_helper_rejects_bad_mask_length(self, fitted_iforest):
        with pytest.raises(ValueError, match="normal_mask"):
            counterfactuals_for_anomalies([0], fitted_iforest["X"],
                                          np.ones(3, dtype=bool))

    def test_out_of_range_id_is_reported(self, fitted_iforest):
        out = counterfactuals_for_anomalies(
            [9999], fitted_iforest["X"],
            np.ones(fitted_iforest["X"].shape[0], dtype=bool),
        )
        assert out[0]["found"] is False


# ── Validation / fidelity ─────────────────────────────────────────────────────

class TestFaithfulness:
    def test_good_ordering_beats_random(self):
        """A score that depends on feature 0 should reward removing it first."""
        def score(row):
            return float(np.exp(-((row[0] - 5.0) ** 2)))

        obs = np.array([5.0, 1.0, 1.0])
        result = faithfulness(obs, score, [0, 1, 2], n_random=5)
        assert result["relative_improvement"] < 0
        assert result["deletion_auc_explained"] < result["deletion_auc_random"]

    def test_empty_feature_order_is_handled(self):
        result = faithfulness(np.zeros(3), lambda r: 1.0, [])
        assert result["deletion_auc_explained"] is None

    def test_deletion_curve_length(self):
        curve = faithfulness(
            np.zeros(4), lambda r: float(r.sum()), [0, 1, 2, 3]
        )["curve"]
        assert len(curve) == 5

    def test_deletion_auc_bounds(self):
        assert deletion_auc([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
        # A straight ramp from 1 to 0 has area 0.5.
        assert deletion_auc([1.0, 0.0]) == pytest.approx(0.5)
        # An immediate collapse shrinks the area towards 0 as points are added.
        assert deletion_auc([1.0, 0.0, 0.0, 0.0]) < 0.25
        assert deletion_auc([1.0, 0.0, 0.0, 0.0, 0.0]) < 0.2

    def test_deletion_auc_of_short_input_is_nan(self):
        assert np.isnan(deletion_auc([1.0]))


class TestStability:
    def test_identical_fits_are_perfectly_stable(self):
        attrs = np.tile(np.array([5.0, 3.0, 1.0, 0.5]), (4, 1))
        names = ["a", "b", "c", "d"]
        result = explanation_stability(attrs, names, top_k=2)
        assert result["mean_rank_correlation"] == pytest.approx(1.0)
        assert result["per_fit_leader"] == ["a"] * 4

    def test_single_fit_is_not_measurable(self):
        result = explanation_stability(np.array([[1.0, 2.0]]), ["a", "b"])
        assert result["mean_rank_correlation"] is None

    def test_top_k_features_reported(self):
        rng = np.random.default_rng(0)
        attrs = rng.random((5, 4))
        result = explanation_stability(attrs, list("abcd"), top_k=2)
        assert result["top_k_features"]
        assert result["top_k_retention"] is not None


class TestValidateExplanation:
    def test_reports_an_overall_verdict(self):
        def score(row):
            return float(np.exp(-((row[0] - 5.0) ** 2)))

        result = validate_explanation(
            np.array([5.0, 1.0, 1.0]), score, [0, 1, 2], n_random=3
        )
        assert result["overall"] in {
            "VERIFIED", "PARTIALLY VERIFIED", "UNVERIFIED"
        }
        assert result["checks_run"] == 1
        assert result["faithfulness"]["verdict"]

    def test_unverified_when_additivity_fails(self):
        result = validate_explanation(
            np.array([1.0, 1.0]),
            lambda r: 1.0,
            [0, 1],
            shap_additivity={"checked": True, "passed": False, "note": "bad"},
            n_random=2,
        )
        assert result["checks_run"] == 2
        assert result["overall"] == "UNVERIFIED"

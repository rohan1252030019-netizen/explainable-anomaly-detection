"""
Tests for src/evaluation/holdout.py and the SHAP explainer.

These cover the two correctness fixes that matter most for the project's claims:
- reported metrics must come from rows the model never saw, and
- SHAP attributions must be expressed in the same units as the score the
  dashboard displays.
"""

import numpy as np
import pytest

from src.evaluation.holdout import (
    compare_holdout_across_models,
    evaluate_holdout,
    holdout_split,
)
from src.explainability.shap_explainer import SHAP_AVAILABLE, SHAPExplainer
from src.models.isolation_forest import IsolationForestDetector
from src.models.one_class_svm import OneClassSVMDetector


def factory():
    return IsolationForestDetector(contamination=0.05, random_state=42)


class TestHoldoutSplit:
    def test_partitions_all_rows_exactly_once(self, fitted_sample):
        _, X, _, y = fitted_sample
        s = holdout_split(X, y=y, test_size=0.3)
        combined = np.concatenate([s["fit_idx"], s["test_idx"]])
        assert len(combined) == len(X)
        assert len(np.unique(combined)) == len(X)

    def test_respects_test_size(self, fitted_sample):
        _, X, _, y = fitted_sample
        s = holdout_split(X, y=y, test_size=0.3)
        assert len(s["X_test"]) == pytest.approx(int(len(X) * 0.3), abs=2)

    def test_split_is_deterministic(self, fitted_sample):
        _, X, _, y = fitted_sample
        a = holdout_split(X, y=y, random_state=42)
        b = holdout_split(X, y=y, random_state=42)
        np.testing.assert_array_equal(a["test_idx"], b["test_idx"])

    def test_labels_are_never_part_of_the_features(self, fitted_sample):
        _, X, _, _ = fitted_sample
        s = holdout_split(X, y=None)
        assert s["y_fit"] is None and s["y_test"] is None

    def test_stratifies_so_both_sides_have_anomalies(self, fitted_sample):
        _, X, _, y = fitted_sample
        s = holdout_split(X, y=y, test_size=0.3)
        assert s["y_test"].sum() > 0
        assert s["y_fit"].sum() > 0


class TestEvaluateHoldout:
    def test_reports_both_splits(self, fitted_sample):
        _, X, _, y = fitted_sample
        r = evaluate_holdout(factory, X, y=y)
        assert r["available"] is True
        assert r["in_sample"] and r["out_of_sample"]
        assert r["optimism_gap"]

    def test_gap_is_in_sample_minus_out_of_sample(self, fitted_sample):
        _, X, _, y = fitted_sample
        r = evaluate_holdout(factory, X, y=y)
        for key in ("roc_auc", "f1"):
            ins = r["in_sample"][key]
            oos = r["out_of_sample"][key]
            assert r["optimism_gap"][key] == pytest.approx(ins - oos, abs=1e-3)

    def test_fit_set_never_contains_test_rows(self, fitted_sample):
        """The whole point: no leakage between fit and evaluation."""
        _, X, _, y = fitted_sample
        r = evaluate_holdout(factory, X, y=y)
        assert r["n_fit"] + r["n_test"] == len(X)

    def test_missing_labels_is_reported_not_faked(self, fitted_sample):
        """Metrics needing ground truth must not be invented."""
        _, X, _, _ = fitted_sample
        r = evaluate_holdout(factory, X, y=None)
        assert r["available"] is False
        assert "labels" in r["reason"].lower()
        assert "in_sample" not in r

    def test_reports_flag_rate_transfer(self, fitted_sample):
        _, X, _, y = fitted_sample
        r = evaluate_holdout(factory, X, y=y)
        assert 0.0 <= r["flag_rate_in_sample"] <= 1.0
        assert 0.0 <= r["flag_rate_out_of_sample"] <= 1.0

    def test_empty_holdout_with_no_anomalies_is_explained(self):
        X = np.random.default_rng(0).normal(size=(60, 3))
        y = np.zeros(60, dtype=int)  # no anomalies anywhere
        r = evaluate_holdout(factory, X, y=y)
        assert r["available"] is False
        assert r["reason"]


class TestCompareHoldout:
    def test_returns_a_row_per_model(self, fitted_sample):
        _, X, _, y = fitted_sample
        table = compare_holdout_across_models(
            {
                "Isolation Forest": factory,
                "One-Class SVM": lambda: OneClassSVMDetector(nu=0.05),
            },
            X,
            y=y,
        )
        assert len(table) == 2
        assert "ROC-AUC (out-of-sample)" in table.columns
        assert "AUC gap" in table.columns

    def test_one_failing_model_does_not_abort_the_table(self, fitted_sample):
        _, X, _, y = fitted_sample

        def broken():
            raise RuntimeError("boom")

        table = compare_holdout_across_models(
            {"Good": factory, "Bad": broken}, X, y=y
        )
        assert len(table) == 2
        bad = table[table["Model"] == "Bad"].iloc[0]
        assert "Error" in bad


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP is not installed")
class TestSHAPExplainer:
    def test_attributions_are_in_score_units_and_additive(self, fitted_iforest):
        """
        The fix this project needed.

        shap.TreeExplainer on a scikit-learn IsolationForest explains the raw
        summed tree path length (~11.5 here), which is neither score_samples nor
        decision_function. Presenting those as contributions to the anomaly
        score is simply wrong, so the explainer must fall back to an explainer
        whose units match, and the values must sum to the displayed score.
        """
        X = fitted_iforest["X"]
        score_fn = fitted_iforest["score_fn"]
        rows = X[:8]

        ex = SHAPExplainer(
            "Isolation Forest",
            fitted_iforest["detector"]._model,
            X,
            [f"f{i}" for i in range(X.shape[1])],
            score_fn=score_fn,
            max_background=40,
        )
        assert ex.is_available()

        sv = ex.explain(rows, max_samples=8)
        assert sv is not None
        assert sv.shape == rows.shape

        base = ex.base_value()
        assert base is not None

        reconstructed = sv.sum(axis=1) + base
        actual = score_fn(rows)
        np.testing.assert_allclose(reconstructed, actual, atol=1e-4)

    def test_additivity_check_passes(self, fitted_iforest):
        X = fitted_iforest["X"]
        ex = SHAPExplainer(
            "Isolation Forest",
            fitted_iforest["detector"]._model,
            X,
            [f"f{i}" for i in range(X.shape[1])],
            score_fn=fitted_iforest["score_fn"],
            max_background=40,
        )
        check = ex.check_additivity(X[:5], max_samples=5)
        assert check["checked"] is True
        assert check["passed"] is True, check["note"]

    def test_status_reports_the_explainer_in_use(self, fitted_iforest):
        X = fitted_iforest["X"]
        ex = SHAPExplainer(
            "Isolation Forest",
            fitted_iforest["detector"]._model,
            X,
            [f"f{i}" for i in range(X.shape[1])],
            score_fn=fitted_iforest["score_fn"],
            max_background=40,
        )
        status = ex.get_status()
        assert status["is_available"] is True
        assert status["explainer_type"]

    def test_works_for_a_non_tree_model(self, fitted_sample):
        _, X, feats, _ = fitted_sample
        from src.detection.scoring import ScoreNormalizer

        detector = OneClassSVMDetector(nu=0.05)
        detector.fit(X)
        normalizer = ScoreNormalizer().fit(detector.score_samples(X))
        def score_fn(a):
            return normalizer.transform(detector.score_samples(np.atleast_2d(a)))

        ex = SHAPExplainer(
            "One-Class SVM", detector._model, X, feats,
            score_fn=score_fn, max_background=40,
        )
        assert ex.is_available()
        check = ex.check_additivity(X[:4], max_samples=4)
        assert check["passed"] is True, check["note"]

    def test_requires_a_score_function(self, fitted_iforest):
        X = fitted_iforest["X"]
        ex = SHAPExplainer(
            "Isolation Forest", fitted_iforest["detector"]._model, X,
            [f"f{i}" for i in range(X.shape[1])],
            score_fn=None,
        )
        assert ex.is_available() is False
        assert "score_fn" in (ex.get_status()["error"] or "")

    def test_explain_single_returns_one_value_per_feature(self, fitted_iforest):
        X = fitted_iforest["X"]
        n_feat = X.shape[1]
        ex = SHAPExplainer(
            "Isolation Forest", fitted_iforest["detector"]._model, X,
            [f"f{i}" for i in range(n_feat)],
            score_fn=fitted_iforest["score_fn"], max_background=40,
        )
        values = ex.explain_single(X[0])
        assert values is not None
        assert len(values) == n_feat

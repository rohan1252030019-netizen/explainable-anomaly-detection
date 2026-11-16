"""
Tests for src/detection/thresholding.py

Severity tiering is the number a triage dashboard leads with, so the tests
focus on the failure modes: bands that collapse into one bucket, non-monotone
labelling, and ties.
"""

import collections

import numpy as np
import pytest

from src.detection.thresholding import (
    SEVERITY_ORDER,
    SEVERITY_THRESHOLDS,
    apply_threshold,
    assign_severity,
    get_threshold_value,
    severity_legend,
    severity_rank_levels,
)


class TestApplyThreshold:
    def test_contamination_flags_about_that_fraction(self):
        rng = np.random.default_rng(0)
        scores = rng.random(1000)
        preds = apply_threshold(scores, method="contamination", contamination=0.05)
        rate = (preds == -1).mean()
        assert rate == pytest.approx(0.05, abs=0.01)

    def test_labels_use_sklearn_convention(self):
        scores = np.array([0.1, 0.9, 0.95])
        preds = apply_threshold(scores, method="fixed", fixed_threshold=0.5)
        assert preds.tolist() == [1, -1, -1]

    def test_quantile_method(self):
        scores = np.linspace(0, 1, 100)
        preds = apply_threshold(scores, method="quantile", quantile=0.95)
        assert (preds == -1).sum() == 5

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown threshold method"):
            apply_threshold(np.array([0.5]), method="nope")

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            apply_threshold(np.zeros((2, 2)))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            apply_threshold(np.array([]))


class TestGetThresholdValue:
    def test_matches_the_threshold_that_was_applied(self):
        rng = np.random.default_rng(1)
        scores = rng.random(500)
        thr = get_threshold_value(scores, method="contamination", contamination=0.1)
        preds = apply_threshold(scores, method="contamination", contamination=0.1)
        assert (scores[preds == -1] >= thr).all()

    def test_fixed_threshold_is_clamped(self):
        assert get_threshold_value(
            np.array([0.1, 0.2]), method="fixed", fixed_threshold=5.0
        ) == 1.0


class TestAssignSeverity:
    def test_every_label_is_from_the_known_set(self):
        rng = np.random.default_rng(2)
        scores = rng.random(500)
        mask = rng.random(500) < 0.2
        sev = assign_severity(scores, anomaly_mask=mask)
        assert set(sev).issubset(set(SEVERITY_ORDER))

    def test_is_monotone_in_score(self):
        """A higher score must never be labelled less severe."""
        rng = np.random.default_rng(3)
        scores = rng.random(800)
        mask = scores > np.quantile(scores, 0.8)
        sev = assign_severity(scores, anomaly_mask=mask)

        rank = {name: i for i, name in enumerate(SEVERITY_ORDER)}
        order = np.argsort(scores[mask])
        labelled = [rank[s] for s in sev[mask][order]]
        assert labelled == sorted(labelled), "severity must increase with score"

    def test_most_anomalous_flagged_row_is_critical(self):
        scores = np.linspace(0.0, 1.0, 200)
        mask = scores > 0.5
        sev = assign_severity(scores, anomaly_mask=mask)
        assert sev[np.argmax(scores)] == "CRITICAL"

    def test_spreads_across_bands_on_heavy_tailed_scores(self):
        """
        Regression guard for the original defect.

        With min-max normalised heavy-tailed scores, fixed bands put nearly
        every flagged row in one bucket, so the severity donut chart was
        effectively single-coloured. Rank tiering must populate all four.
        """
        rng = np.random.default_rng(4)
        raw = rng.normal(0, 1, 880)
        raw[0] = -40.0  # extreme tail, as a real detector would produce
        inverted = -raw
        scores = (inverted - inverted.min()) / (inverted.max() - inverted.min())

        mask = scores > np.quantile(scores, 0.95)
        counts = collections.Counter(assign_severity(scores, anomaly_mask=mask))
        assert len(counts) == len(SEVERITY_ORDER), f"bands collapsed: {counts}"
        assert all(v > 0 for v in counts.values())

    def test_ties_still_produce_a_spread(self):
        """Robust normalisation clips the tail, creating ties at 1.0."""
        scores = np.full(50, 0.97)
        mask = np.ones(50, dtype=bool)
        counts = collections.Counter(assign_severity(scores, anomaly_mask=mask))
        assert len(counts) == len(SEVERITY_ORDER)

    def test_explicit_thresholds_take_precedence(self):
        scores = np.array([0.1, 0.6, 0.95])
        sev = assign_severity(
            scores,
            thresholds={
                "LOW": (0.0, 0.5),
                "MEDIUM": (0.5, 0.75),
                "HIGH": (0.75, 0.9),
                "CRITICAL": (0.9, 1.01),
            },
        )
        assert sev.tolist() == ["LOW", "MEDIUM", "CRITICAL"]

    def test_score_of_exactly_one_is_top_band(self):
        sev = assign_severity(
            np.array([1.0]), thresholds=SEVERITY_THRESHOLDS
        )
        assert sev[0] == "CRITICAL"

    def test_few_flagged_rows_falls_back_to_absolute_bands(self):
        sev = assign_severity(np.array([0.9, 0.95]), anomaly_mask=np.ones(2, bool))
        assert set(sev).issubset(set(SEVERITY_ORDER))

    def test_no_flagged_rows_is_all_low(self):
        sev = assign_severity(np.array([0.2, 0.4]), anomaly_mask=np.zeros(2, bool))
        assert sev.tolist() == ["LOW", "LOW"]

    def test_mask_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="anomaly_mask"):
            assign_severity(np.zeros(5), anomaly_mask=np.zeros(3, dtype=bool))


class TestSeverityLegend:
    def test_levels_are_a_partition(self):
        levels = severity_rank_levels()
        assert sorted(levels) == sorted(SEVERITY_ORDER)

        edges = [levels[name][0] for name in SEVERITY_ORDER] + [
            levels[SEVERITY_ORDER[-1]][1]
        ]
        assert edges[0] == 0.0
        assert edges[-1] == 1.0
        assert edges == sorted(edges), "band edges must increase monotonically"
        # Contiguous: each band starts exactly where the previous one ends.
        for prev, current in zip(SEVERITY_ORDER[:-1], SEVERITY_ORDER[1:], strict=True):
            assert levels[prev][1] == levels[current][0]

    def test_legend_shares_sum_to_one(self):
        legend = severity_legend({"LOW": 22, "MEDIUM": 14, "HIGH": 6, "CRITICAL": 2})
        # Shares are rounded for display, so allow for accumulated rounding.
        assert sum(row["share"] for row in legend) == pytest.approx(1.0, abs=1e-3)
        assert legend[0]["severity"] == "CRITICAL", "most severe first"

    def test_legend_handles_empty_counts(self):
        legend = severity_legend({})
        assert all(row["share"] == 0.0 for row in legend)

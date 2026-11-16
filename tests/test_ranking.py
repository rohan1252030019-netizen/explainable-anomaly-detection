"""Tests for src/detection/ranking.py."""

import numpy as np
import pandas as pd

from src.detection.ranking import (
    build_results_dataframe,
    lookup_by_original_index,
    rank_anomalies,
    top_anomalies,
)


class TestRankAnomalies:
    def test_only_anomalies_are_ranked(self):
        scores = np.array([0.1, 0.9, 0.5, 0.7])
        preds = np.array([1, -1, 1, -1])
        ranks = rank_anomalies(scores, preds)
        assert np.isnan(ranks[0]) and np.isnan(ranks[2])
        assert ranks[1] == 1  # highest score gets rank 1
        assert ranks[3] == 2

    def test_rank_all_ranks_everything(self):
        scores = np.array([0.1, 0.9, 0.5])
        preds = np.array([1, -1, 1])
        ranks = rank_anomalies(scores, preds, rank_all=True)
        assert ranks.tolist() == [3.0, 1.0, 2.0]

    def test_ranks_are_a_dense_sequence(self):
        scores = np.random.default_rng(0).random(20)
        preds = np.full(20, -1)
        ranks = rank_anomalies(scores, preds)
        assert sorted(ranks.tolist()) == list(range(1, 21))


class TestBuildResultsDataframe:
    def _frame(self, n=10):
        raw = pd.DataFrame({"a": range(n)})
        scores = np.linspace(-0.9, -0.1, n)
        preds = np.array([-1] * 3 + [1] * (n - 3))
        severity = np.array(["CRITICAL", "HIGH", "LOW"] + ["LOW"] * (n - 3))
        return build_results_dataframe(raw, scores, scores, preds, severity, "m")

    def test_adds_expected_columns(self):
        df = self._frame()
        for col in (
            "anomaly_score", "normalized_score", "prediction",
            "severity", "rank", "is_anomaly", "model", "original_index",
        ):
            assert col in df.columns

    def test_anomalies_sort_first(self):
        df = self._frame()
        flags = df["is_anomaly"].tolist()
        assert flags == sorted(flags, reverse=True)

    def test_original_index_tracks_the_input_row(self):
        """The sorted frame must remain addressable by its input position."""
        n = 10
        raw = pd.DataFrame({"a": range(n)})
        df = build_results_dataframe(
            raw,
            np.linspace(-0.9, -0.1, n),
            np.linspace(-0.9, -0.1, n),
            np.array([-1] * 3 + [1] * (n - 3)),
            np.array(["LOW"] * n),
            "m",
        )
        # Row value 'a' must always match original_index.
        assert (df["a"].values == df["original_index"].values).all()

    def test_lookup_returns_the_right_record_after_sorting(self):
        """
        Regression guard.

        df_results is sorted, so positional lookup used to return whichever
        row happened to land at that offset — a different observation with a
        different rank.
        """
        n = 50
        raw = pd.DataFrame({"a": range(n)})
        scores = np.linspace(-0.9, -0.1, n)
        preds = np.array([-1] * 10 + [1] * (n - 10))
        df = build_results_dataframe(
            raw, scores, scores, preds, np.array(["LOW"] * n), "m"
        )

        # Positional lookup must NOT be used here; index lookup must be exact.
        for row_id in (0, 5, 25, 49):
            row = lookup_by_original_index(df, row_id)
            assert row is not None
            assert row["a"] == row_id
            assert int(row["original_index"]) == row_id

    def test_lookup_returns_none_out_of_range(self):
        assert lookup_by_original_index(self._frame(), 999) is None

    def test_lookup_tolerates_legacy_frame_without_the_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert lookup_by_original_index(df, 0) is None


class TestTopAnomalies:
    def test_returns_ranked_head(self):
        n = 20
        raw = pd.DataFrame({"a": range(n)})
        scores = np.linspace(-0.9, -0.1, n)
        preds = np.array([-1] * 10 + [1] * 10)
        df = build_results_dataframe(
            raw, scores, scores, preds, np.array(["LOW"] * n), "m"
        )
        top = top_anomalies(df, n=3)
        assert len(top) == 3
        assert top["rank"].tolist() == [1, 2, 3]

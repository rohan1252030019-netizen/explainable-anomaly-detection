"""Tests for src/data/preprocessing.py, focusing on the unit-space contract."""

import numpy as np
import pytest

from src.data.preprocessing import PreprocessingPipeline


class TestPreprocessingPipeline:
    def test_drops_the_label_column(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label")
        X, feats = pipeline.fit_transform(sample_df)
        assert "label" not in feats
        assert X.shape[0] == len(sample_df)

    def test_output_has_no_nan_or_inf(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label")
        X, _ = pipeline.fit_transform(sample_df)
        assert np.all(np.isfinite(X))

    def test_standard_scaler_centres_numerical_block(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label", scaler_type="standard")
        X, _ = pipeline.fit_transform(sample_df)
        n_num = len(pipeline.numerical_cols_)
        assert np.allclose(X[:, :n_num].mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(X[:, :n_num].std(axis=0), 1.0, atol=1e-8)

    def test_transform_matches_fit_transform(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label")
        X_fit, _ = pipeline.fit_transform(sample_df)
        X_again = pipeline.transform(sample_df)
        np.testing.assert_allclose(X_fit, X_again)

    def test_get_labels_returns_the_label_column(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label")
        pipeline.fit_transform(sample_df)
        labels = pipeline.get_labels(sample_df)
        assert labels is not None
        assert len(labels) == len(sample_df)

    def test_reference_stats_are_in_raw_units(self, sample_df):
        """
        Contract that the deviation maths depends on.

        get_reference_stats() deliberately reports raw dataset values so the UI
        can display real numbers, even though the model consumes scaled input.
        """
        pipeline = PreprocessingPipeline(label_col="label")
        pipeline.fit_transform(sample_df)
        stats = pipeline.get_reference_stats()

        for col in pipeline.numerical_cols_:
            expected = float(sample_df[col].median())
            assert stats["median"][col] == pytest.approx(expected, rel=1e-6)

    def test_log_records_each_step(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label")
        pipeline.fit_transform(sample_df)
        assert pipeline.get_log()


class TestDenormalize:
    def test_round_trips_standard_scaler(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label", scaler_type="standard")
        X, _ = pipeline.fit_transform(sample_df)
        recovered = pipeline.denormalize(X[0])

        for i, col in enumerate(pipeline.numerical_cols_):
            assert recovered[i] == pytest.approx(
                float(sample_df[col].iloc[0]), rel=1e-4
            )

    def test_denormalised_row_matches_raw_medians(self, sample_df):
        """The whole point: raw row against raw reference stats is coherent."""
        pipeline = PreprocessingPipeline(label_col="label")
        pipeline.fit_transform(sample_df)

        n_num = len(pipeline.numerical_cols_)
        raw_medians = [float(sample_df[c].median()) for c in pipeline.numerical_cols_]

        # Round-trip through the scaler: scaled medians -> denormalize() must
        # return the raw medians, so raw-vs-raw comparisons are meaningful.
        scaler = pipeline._pipeline.named_transformers_["num"].named_steps["scaler"]
        scaled = np.zeros(len(pipeline.feature_names_out_))
        scaled[:n_num] = scaler.transform(
            np.array(raw_medians, dtype=float).reshape(1, -1)
        ).ravel()

        recovered = pipeline.denormalize(scaled)
        for i, expected in enumerate(raw_medians):
            assert recovered[i] == pytest.approx(expected, rel=1e-4)

    def test_onehot_block_is_left_as_activation(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label")
        X, _ = pipeline.fit_transform(sample_df)
        n_num = len(pipeline.numerical_cols_)
        recovered = pipeline.denormalize(X[0])
        np.testing.assert_allclose(recovered[n_num:], X[0][n_num:])

    def test_handles_wrong_length_input_gracefully(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label")
        pipeline.fit_transform(sample_df)
        # Should not raise; returns what it can.
        assert pipeline.denormalize(np.zeros(2)).shape == (2,)


class TestFeatureUnits:
    def test_declares_each_feature_space(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label")
        X, feats = pipeline.fit_transform(sample_df)
        units = pipeline.feature_units()
        assert set(units) == set(feats)
        assert all(v in {"raw", "onehot"} for v in units.values())

    def test_numerical_features_are_raw(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label")
        pipeline.fit_transform(sample_df)
        units = pipeline.feature_units()
        for col in pipeline.numerical_cols_:
            assert units[col] == "raw"


class TestRobustScaler:
    def test_robust_scaling_runs(self, sample_df):
        pipeline = PreprocessingPipeline(label_col="label", scaler_type="robust")
        X, _ = pipeline.fit_transform(sample_df)
        assert np.all(np.isfinite(X))

"""
tests/test_ulb_dataset.py

Integration checks for the real ULB credit-card fraud dataset.

These tests are skipped automatically when the dataset has not been prepared:

    python scripts/prepare_ulb_dataset.py

They assert provenance-level facts (shape, fraud count, no missing values),
that the label never leaks into the feature matrix, and that an unsupervised
detector beats chance on a time-ordered holdout.  The thresholds are floors,
not targets: they catch a broken pipeline, not a research result.
"""

import os

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(PROJECT_ROOT, "data", "sample", "creditcard_fraud_sample.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SAMPLE),
    reason="ULB sample not prepared; run scripts/prepare_ulb_dataset.py",
)

EXPECTED_SAMPLE_ROWS = 20_000
EXPECTED_FRAUDS = 492
EXPECTED_COLUMNS = 31


@pytest.fixture(scope="module")
def ulb_df():
    import pandas as pd

    df = pd.read_csv(SAMPLE)
    return df.sort_values("Time", kind="mergesort").reset_index(drop=True)


def test_schema_and_fraud_count(ulb_df):
    assert ulb_df.shape == (EXPECTED_SAMPLE_ROWS, EXPECTED_COLUMNS)
    assert int(ulb_df["Class"].sum()) == EXPECTED_FRAUDS
    assert set(ulb_df["Class"].unique()) == {0, 1}
    assert ulb_df.isna().sum().sum() == 0

    expected_pca = {f"V{i}" for i in range(1, 29)}
    assert expected_pca <= set(ulb_df.columns)
    assert {"Time", "Amount", "Class"} <= set(ulb_df.columns)


def test_label_is_excluded_from_features(ulb_df):
    from src.data.preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline(label_col="Class", scaler_type="robust")
    X, feature_names = pipeline.fit_transform(ulb_df.head(2_000))

    assert "Class" not in feature_names
    assert X.shape[1] == len(feature_names)
    assert np.isfinite(X).all()


def test_time_split_detector_beats_chance(ulb_df):
    from src.data.preprocessing import PreprocessingPipeline
    from src.evaluation.holdout import evaluate_holdout_split
    from src.models.model_manager import ModelManager

    cut = int(len(ulb_df) * 0.7)
    fit_df = ulb_df.iloc[:cut]
    test_df = ulb_df.iloc[cut:]

    pipeline = PreprocessingPipeline(label_col="Class", scaler_type="robust")
    X_fit, _ = pipeline.fit_transform(fit_df)
    X_test = pipeline.transform(test_df)

    y_fit = fit_df["Class"].to_numpy(dtype=int)
    y_test = test_df["Class"].to_numpy(dtype=int)
    assert y_test.sum() > 0, "holdout window must contain frauds"

    contamination = float(np.clip(y_fit.mean(), 1e-4, 0.5))
    manager = ModelManager()

    def factory():
        return manager.create_detector(
            "Isolation Forest", {"contamination": contamination}
        )

    report = evaluate_holdout_split(
        factory, X_fit, X_test, y_fit=y_fit, y_test=y_test,
        threshold_method="contamination", contamination=contamination,
    )

    assert report["available"] is True
    oos = report["out_of_sample"]
    # Chance-level PR-AUC here is ~0.018 (test prevalence). A working detector
    # must clear it by a wide margin; ROC-AUC guards the ranking direction.
    assert oos["pr_auc"] > 0.15, oos
    assert oos["roc_auc"] > 0.85, oos

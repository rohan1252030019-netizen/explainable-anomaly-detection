"""Shared pytest fixtures."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """The bundled benchmark dataset, regenerated if the file is absent."""
    path = os.path.join(PROJECT_ROOT, "data", "sample", "sample_anomaly_data.csv")
    if not os.path.exists(path):
        from src.data.loader import generate_sample_dataset
        os.makedirs(os.path.dirname(path), exist_ok=True)
        generate_sample_dataset(
            n_normal=200, n_anomalies=20, random_state=42
        ).to_csv(path, index=False)
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def toy_scores() -> np.ndarray:
    """A small score vector with the sklearn sign convention."""
    return np.array([-0.9, -0.5, -0.45, -0.4, 0.0, 0.1, 0.2, 0.3], dtype=float)


@pytest.fixture
def fitted_sample(sample_df):
    """(pipeline, X, feature_names, y) for the sample dataset."""
    from src.data.preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline(label_col="label")
    X, feature_names = pipeline.fit_transform(sample_df)
    y = (sample_df["label"].values == 1).astype(int)
    return pipeline, X, feature_names, y


@pytest.fixture
def fitted_iforest(fitted_sample):
    """A fitted Isolation Forest plus its scores over the same rows."""
    from src.detection.scoring import ScoreNormalizer
    from src.models.isolation_forest import IsolationForestDetector

    _, X, _, _ = fitted_sample
    detector = IsolationForestDetector(contamination=0.05, random_state=42)
    detector.fit(X)
    raw = detector.score_samples(X)
    normalizer = ScoreNormalizer().fit(raw)

    def score_fn(arr):
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        return normalizer.transform(detector.score_samples(arr))

    return {
        "detector": detector,
        "X": X,
        "raw": raw,
        "norm": normalizer.transform(raw),
        "normalizer": normalizer,
        "score_fn": score_fn,
    }

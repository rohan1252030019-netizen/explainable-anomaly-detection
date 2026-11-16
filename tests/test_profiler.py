"""
tests/test_profiler.py

Unit tests for DataProfiler and Dataset Health Score calculation.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.profiler import DataProfiler


@pytest.fixture
def clean_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feature_1": rng.normal(0, 1, 100),
        "feature_2": rng.uniform(10, 50, 100),
        "feature_3": rng.exponential(2, 100),
        "category": rng.choice(["A", "B", "C"], 100),
    })


@pytest.fixture
def messy_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    # 70 distinct categories guarantee > 50 unique values
    categories = [f"cat_{i}" for i in range(70)] + [f"cat_{i}" for i in range(30)]
    df = pd.DataFrame({
        "id_col": [f"ID_{i}" for i in range(100)],
        "const_col": [1.0] * 100,
        "feature_a": rng.normal(0, 1, 100),
        "feature_b": rng.normal(0, 1, 100),
        "category": categories,  # >50 unique
    })
    # Make feature_b almost perfectly collinear with feature_a
    df["feature_b"] = df["feature_a"] * 2.0 + rng.normal(0, 0.001, 100)
    # Inject missing values
    df.loc[0:15, "feature_a"] = np.nan
    # Inject duplicate rows
    df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
    return df


def test_profiler_clean_dataset(clean_dataframe: pd.DataFrame):
    profiler = DataProfiler()
    report = profiler.profile(clean_dataframe)

    assert report["n_rows"] == 100
    assert report["n_cols"] == 4
    assert len(report["schema"]["numerical_cols"]) == 3
    assert len(report["schema"]["categorical_cols"]) == 1
    assert report["health"]["score"] >= 85.0
    assert "A" in report["health"]["grade"]
    assert report["quality"]["total_missing_cells"] == 0
    assert report["quality"]["duplicate_rows_count"] == 0


def test_profiler_messy_dataset(messy_dataframe: pd.DataFrame):
    profiler = DataProfiler()
    report = profiler.profile(messy_dataframe)

    assert report["n_rows"] == 103
    # ID detection
    assert "id_col" in report["schema"]["id_candidates"]
    # Constant column
    assert "const_col" in report["schema"]["constant_cols"]
    # High cardinality (>50 unique categories)
    assert "category" in report["schema"]["high_cardinality_cols"]
    # Collinear pair detected
    assert len(report["collinearity"]["collinear_pairs"]) >= 1
    # Missing cells detected
    assert report["quality"]["total_missing_cells"] > 0
    # Duplicate rows detected
    assert report["quality"]["duplicate_rows_count"] >= 3
    # Health score should have deductions
    assert report["health"]["score"] < 80.0
    assert len(report["health"]["deductions"]) >= 3
    assert len(report["recommendations"]) > 0


def test_profiler_empty_dataset():
    profiler = DataProfiler()
    report = profiler.profile(pd.DataFrame())
    assert report["n_rows"] == 0
    assert report["health_score"] == 0.0

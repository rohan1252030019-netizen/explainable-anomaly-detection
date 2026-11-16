"""
tests/test_experiments.py

Unit tests for ExperimentTracker:
- Dataset hashing
- Recording and persisting experiments
- Retrieval
"""

import pandas as pd

from backend.experiments import ExperimentTracker


def test_experiment_tracker(tmp_path):
    log_dir = str(tmp_path / "experiments")
    tracker = ExperimentTracker(log_dir=log_dir)

    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
    })

    h1 = tracker.compute_dataset_hash(df)
    assert len(h1) == 16

    # Log an experiment
    entry = tracker.record_experiment(
        dataset_name="test_data",
        df=df,
        model_name="Isolation Forest",
        parameters={"n_estimators": 50},
        results={"fit_time_sec": 0.05, "n_anomalies": 1, "anomaly_fraction": 0.33},
    )

    assert entry["dataset_hash"] == h1
    assert entry["model_name"] == "Isolation Forest"

    all_exps = tracker.list_experiments()
    assert len(all_exps) == 1
    assert all_exps[0]["experiment_id"] == entry["experiment_id"]

    retrieved = tracker.get_experiment(entry["experiment_id"])
    assert retrieved is not None
    assert retrieved["dataset_name"] == "test_data"

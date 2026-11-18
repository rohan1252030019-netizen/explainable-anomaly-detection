"""
tests/test_phase5_timeseries.py

Tests for SENTINEL Time-Series & Streaming Anomaly Intelligence (Phase 5):
1. STL Seasonal-Trend Decomposition Residual Anomaly Detection.
2. Strictly Causal Rolling Features & Zero Lookahead Leakage Verification.
3. Change-Point Detection (PELT and BOCPD).
4. Incremental Streaming Half-Space Trees (HS-Trees).
5. Sliding-Window PSI Drift Monitor and Human-in-the-Loop Recalibration Policy.
6. API v2 Time-Series & Streaming Endpoints (including WebSocket).
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.timeseries.changepoint import detect_changepoints_bocpd, detect_changepoints_pelt
from src.timeseries.rolling_features import (
    generate_causal_rolling_features,
    verify_temporal_causality,
)
from src.timeseries.stl_detector import STLAnomalyDetector
from src.timeseries.streaming_detector import (
    StreamingDriftMonitor,
    StreamingHalfSpaceTrees,
)


@pytest.fixture
def seasonal_series():
    """Synthetic time-series with linear trend, 7-day seasonality, and 2 injected anomalies."""
    n = 100
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 5.0 * np.sin(2 * np.pi * t / 7.0)
    noise = np.random.default_rng(42).normal(0, 0.5, size=n)
    y = trend + seasonal + noise
    # Injected residual anomalies
    y[30] += 15.0
    y[75] -= 12.0
    return y


def test_stl_anomaly_detector(seasonal_series):
    detector = STLAnomalyDetector(period=7, robust_z_threshold=3.0)
    res = detector.fit_transform(seasonal_series)

    assert res["n_anomalies"] >= 2
    assert 30 in res["anomaly_indices"]
    assert 75 in res["anomaly_indices"]
    assert len(res["trend"]) == len(seasonal_series)
    assert len(res["seasonal"]) == len(seasonal_series)
    assert len(res["residual"]) == len(seasonal_series)
    assert res["anomaly_scores"][30] > 0.8
    assert res["anomaly_scores"][75] > 0.8


def test_causal_rolling_features_leakage_proof():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=60, freq="D"),
        "val": np.random.default_rng(42).normal(10, 2, size=60),
    })

    df_feats, new_cols = generate_causal_rolling_features(
        df=df,
        target_columns=["val"],
        windows=[5, 10],
        lags=[1, 2],
    )

    assert len(new_cols) > 0
    for col in new_cols:
        assert col in df_feats.columns

    # Mathematically verify that future corruption has zero effect on past features
    is_causal = verify_temporal_causality(
        df=df,
        target_col="val",
        feature_generator_fn=generate_causal_rolling_features,
        t_test=25,
    )
    assert is_causal is True, "Lookahead data leakage detected in rolling features!"


def test_changepoint_detection():
    # Segment 1: mean 0; Segment 2: mean 10 (regime shift at index 50)
    rng = np.random.default_rng(42)
    s1 = rng.normal(0, 1, size=50)
    s2 = rng.normal(10, 1, size=50)
    series = np.concatenate([s1, s2])

    # 1. PELT
    pelt_res = detect_changepoints_pelt(series, min_size=10)
    assert pelt_res["n_changepoints"] >= 1
    # Check that changepoint is close to 50 (within margin of 5 steps)
    detected_cp = min(pelt_res["changepoints"], key=lambda cp: abs(cp - 50))
    assert abs(detected_cp - 50) <= 5

    # 2. BOCPD
    bocpd_res = detect_changepoints_bocpd(series, hazard_rate=0.02, threshold=0.4)
    assert bocpd_res["n_changepoints"] >= 1
    detected_bocpd = min(bocpd_res["flagged_changepoints"], key=lambda cp: abs(cp - 50))
    assert abs(detected_bocpd - 50) <= 5


def test_streaming_half_space_trees():
    rng = np.random.default_rng(42)
    X_ref = rng.normal(0, 1, size=(200, 4))

    hst = StreamingHalfSpaceTrees(n_features=4, n_trees=20, max_depth=6)
    hst.fit_reference(X_ref)

    # In-distribution sample
    norm_sample = np.array([0.1, -0.2, 0.05, -0.1])
    norm_res = hst.score_one(norm_sample)

    # Extreme anomaly
    anom_sample = np.array([5.5, -6.0, 7.2, 8.0])
    anom_res = hst.score_one(anom_sample)

    assert anom_res["anomaly_score"] > norm_res["anomaly_score"]
    assert anom_res["raw_mass"] < norm_res["raw_mass"]


def test_streaming_drift_monitor_and_recalibration():
    rng = np.random.default_rng(42)
    ref_data = rng.normal(0, 1, size=(100, 3))
    feature_names = ["feat1", "feat2", "feat3"]

    monitor = StreamingDriftMonitor(
        reference_data=ref_data,
        feature_names=feature_names,
        window_size=50,
        psi_threshold_warning=0.10,
        psi_threshold_alarm=0.25,
    )

    # 1. Stream in-distribution normal data
    for _ in range(50):
        row = rng.normal(0, 1, size=3)
        res = monitor.add_observation(row)

    assert res["status"] in ["STABLE", "MODERATE_DRIFT_WARNING", "Buffering"]
    assert res["alarm"] is False

    # 2. Stream severely drifted data (mean shift +4 SD)
    for _ in range(50):
        shifted_row = rng.normal(4.0, 1, size=3)
        drift_res = monitor.add_observation(shifted_row)

    assert drift_res["alarm"] is True
    assert drift_res["status"] == "CRITICAL_DRIFT_ALARM"
    assert drift_res["has_recalibration_proposal"] is True

    # 3. Human analyst approval to recalibrate
    approval = monitor.approve_recalibration(approved=True)
    assert approval["success"] is True
    assert "recalibrated" in approval["action"].lower()
    assert monitor.pending_recalibration_proposal is None


def test_api_v2_timeseries_and_streaming_endpoints():
    client = TestClient(app)

    # 1. Load benchmark dataset & run detection
    load_resp = client.post("/api/dataset/sample")
    assert load_resp.status_code == 200

    det_resp = client.post("/api/detection/run", json={"model_name": "Isolation Forest"})
    assert det_resp.status_code == 200

    # 2. Test STL decompose
    stl_resp = client.post("/api/v2/timeseries/stl-decompose", json={"period": 7, "threshold": 2.5})
    assert stl_resp.status_code == 200
    stl_data = stl_resp.json()
    assert "trend" in stl_data
    assert "seasonal" in stl_data
    assert "residual" in stl_data

    # 3. Test Rolling features
    roll_resp = client.post(
        "/api/v2/timeseries/rolling-features",
        json={"target_columns": ["transaction_amount"], "windows": [5], "lags": [1]},
    )
    assert roll_resp.status_code == 200
    roll_data = roll_resp.json()
    assert roll_data["is_causal_verified"] is True

    # 4. Test Changepoint detection
    cp_resp = client.post("/api/v2/timeseries/changepoints", json={"column": "transaction_amount", "method": "pelt"})
    assert cp_resp.status_code == 200
    cp_data = cp_resp.json()
    assert "changepoints" in cp_data

    # 5. Test Streaming ingest
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    n_feats = health_resp.json()["preprocessed_features"]
    ingest_val = [0.0] * n_feats
    ingest_resp = client.post("/api/v2/streaming/ingest", json={"values": ingest_val})
    assert ingest_resp.status_code == 200
    ing_data = ingest_resp.json()
    assert "anomaly_score" in ing_data
    assert "overall_psi" in ing_data

    # 6. Test WebSocket streaming
    with client.websocket_connect("/api/v2/streaming/ws") as ws:
        ws.send_text('{"values": ' + str(ingest_val) + "}")
        msg = ws.receive_text()
        assert "anomaly_score" in msg

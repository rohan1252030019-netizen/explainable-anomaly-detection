"""
backend/routers/timeseries_v2.py

API v2 router for SENTINEL Time-Series & Streaming Anomaly Intelligence:
- Seasonal-Trend Decomposition (STL) Residual Anomaly Detection
- Strictly Causal Rolling Window Features
- Change-Point Detection (PELT & BOCPD)
- Streaming Half-Space Trees (HS-Trees) Incremental Ingest
- Sliding-Window Population Stability Index (PSI) Drift Monitor & Recalibration Policy
- WebSocket Streaming Endpoint
"""

import json

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from backend.schemas import (
    ChangepointRequest,
    ChangepointResponse,
    RecalibrationApprovalRequest,
    RecalibrationApprovalResponse,
    RollingFeaturesRequest,
    RollingFeaturesResponse,
    STLDecomposeRequest,
    STLDecomposeResponse,
    StreamingIngestRequest,
    StreamingIngestResponse,
)
from backend.state import state
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

router = APIRouter(prefix="/api/v2", tags=["timeseries-and-streaming"])

# Module-level streaming runtime instances
_streaming_hst: StreamingHalfSpaceTrees | None = None
_streaming_drift: StreamingDriftMonitor | None = None


def _get_or_init_streaming():
    global _streaming_hst, _streaming_drift
    if state.X is None or len(state.feature_names) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A reference dataset must be loaded before initializing streaming.",
        )
    if _streaming_hst is None or _streaming_hst.n_features != len(state.feature_names):
        _streaming_hst = StreamingHalfSpaceTrees(n_features=len(state.feature_names))
        _streaming_hst.fit_reference(state.X)

    if _streaming_drift is None:
        _streaming_drift = StreamingDriftMonitor(
            reference_data=state.X,
            feature_names=state.feature_names,
            window_size=100,
        )
    return _streaming_hst, _streaming_drift


@router.post("/timeseries/stl-decompose", response_model=STLDecomposeResponse)
def stl_decomposition(req: STLDecomposeRequest) -> STLDecomposeResponse:
    """Run Seasonal-Trend Decomposition (STL) and detect residual anomalies."""
    if state.df_raw is None or len(state.df_raw) == 0:
        raise HTTPException(status_code=400, detail="Load a dataset first.")

    target_col = req.column
    if target_col is None:
        # Pick first numeric column
        num_cols = state.df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise HTTPException(status_code=400, detail="No numeric columns available in dataset.")
        target_col = num_cols[0]
    elif target_col not in state.df_raw.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found.")

    series = state.df_raw[target_col].dropna().to_numpy(dtype=np.float64)
    if len(series) < req.period * 2:
        raise HTTPException(
            status_code=400,
            detail=f"Series length ({len(series)}) too short for period {req.period} (requires at least {req.period*2}).",
        )

    detector = STLAnomalyDetector(period=req.period, robust_z_threshold=req.threshold)
    res = detector.fit_transform(series)

    return STLDecomposeResponse(
        period=res["period"],
        n_samples=len(series),
        n_anomalies=res["n_anomalies"],
        anomaly_fraction=round(float(res["n_anomalies"] / len(series)), 4),
        trend=[round(float(v), 4) for v in res["trend"]],
        seasonal=[round(float(v), 4) for v in res["seasonal"]],
        residual=[round(float(v), 4) for v in res["residual"]],
        anomaly_scores=[round(float(v), 4) for v in res["anomaly_scores"]],
        anomaly_indices=res["anomaly_indices"],
    )


@router.post("/timeseries/rolling-features", response_model=RollingFeaturesResponse)
def create_causal_rolling_features(req: RollingFeaturesRequest) -> RollingFeaturesResponse:
    """Generate strictly causal rolling statistics with zero look-ahead leakage."""
    if state.df_raw is None or len(state.df_raw) == 0:
        raise HTTPException(status_code=400, detail="Load a dataset first.")

    for col in req.target_columns:
        if col not in state.df_raw.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not in dataset.")

    df_feats, new_cols = generate_causal_rolling_features(
        df=state.df_raw,
        target_columns=req.target_columns,
        windows=req.windows,
        lags=req.lags,
        include_ema=req.include_ema,
    )

    # Verify temporal causality on first column
    is_causal = verify_temporal_causality(
        df=state.df_raw,
        target_col=req.target_columns[0],
        feature_generator_fn=generate_causal_rolling_features,
        t_test=min(20, len(state.df_raw) - 6),
    )

    return RollingFeaturesResponse(
        original_columns=req.target_columns,
        engineered_columns=new_cols,
        n_rows=len(df_feats),
        is_causal_verified=is_causal,
    )


@router.post("/timeseries/changepoints", response_model=ChangepointResponse)
def changepoint_detection(req: ChangepointRequest) -> ChangepointResponse:
    """Detect structural regime shifts using PELT or BOCPD."""
    if state.df_raw is None or len(state.df_raw) == 0:
        raise HTTPException(status_code=400, detail="Load a dataset first.")

    target_col = req.column
    if target_col is None:
        num_cols = state.df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise HTTPException(status_code=400, detail="No numeric columns available in dataset.")
        target_col = num_cols[0]
    elif target_col not in state.df_raw.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found.")

    series = state.df_raw[target_col].dropna().to_numpy(dtype=np.float64)

    if req.method == "pelt":
        res = detect_changepoints_pelt(series=series, penalty=req.penalty)
        return ChangepointResponse(
            method="PELT",
            n_changepoints=res["n_changepoints"],
            changepoints=res["changepoints"],
            details={"segments": res["segments"], "penalty_beta": res["penalty_beta"], "total_cost": res["total_cost"]},
        )
    elif req.method == "bocpd":
        res = detect_changepoints_bocpd(series=series, hazard_rate=req.hazard_rate)
        return ChangepointResponse(
            method="BOCPD",
            n_changepoints=res["n_changepoints"],
            changepoints=res["flagged_changepoints"],
            details={"max_prob": res["max_prob"], "changepoint_probabilities": res["changepoint_probabilities"]},
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported method '{req.method}'.")


@router.post("/streaming/ingest", response_model=StreamingIngestResponse)
def streaming_ingest(req: StreamingIngestRequest) -> StreamingIngestResponse:
    """Ingest a single streaming vector, update online detector and live PSI drift monitor."""
    hst, drift = _get_or_init_streaming()

    vec = np.asarray(req.values, dtype=np.float64)
    if len(vec) != hst.n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Dimension mismatch: expected {hst.n_features} features, got {len(vec)}.",
        )

    # 1. Incremental scoring via Half-Space Trees
    score_info = hst.score_one(vec)

    # 2. Live PSI drift tracking
    drift_info = drift.add_observation(vec)

    return StreamingIngestResponse(
        anomaly_score=score_info["anomaly_score"],
        raw_mass=score_info["raw_mass"],
        total_samples_processed=score_info["total_samples_processed"],
        drift_status=drift_info["status"],
        overall_psi=drift_info["overall_psi"],
        drift_alarm=drift_info.get("alarm", False),
        has_recalibration_proposal=drift_info.get("has_recalibration_proposal", False),
        recalibration_proposal=drift_info.get("recalibration_proposal"),
    )


@router.post("/streaming/recalibrate", response_model=RecalibrationApprovalResponse)
def streaming_recalibrate(req: RecalibrationApprovalRequest) -> RecalibrationApprovalResponse:
    """Human-in-the-loop signoff to approve or dismiss a drift recalibration proposal."""
    _, drift = _get_or_init_streaming()
    res = drift.approve_recalibration(approved=req.approved)
    return RecalibrationApprovalResponse(
        success=res["success"],
        action=res["action"] if "action" in res else res.get("message", ""),
        proposal_id=res.get("proposal_id"),
    )


@router.websocket("/streaming/ws")
async def streaming_websocket(websocket: WebSocket):
    """
    Real-time bidirectional WebSocket stream for low-latency scoring and live PSI alarms.

    Protocol:
    - Client sends JSON: {"values": [f1, f2, ...]}
    - Server responds JSON: {"anomaly_score": ..., "overall_psi": ..., "drift_alarm": ...}
    """
    await websocket.accept()
    try:
        hst, drift = _get_or_init_streaming()
    except Exception as exc:
        await websocket.send_text(json.dumps({"error": str(exc)}))
        await websocket.close()
        return

    try:
        while True:
            text = await websocket.receive_text()
            data = json.loads(text)
            values = data.get("values")
            if values is None:
                await websocket.send_text(json.dumps({"error": "Missing 'values' key"}))
                continue

            vec = np.asarray(values, dtype=np.float64)
            if len(vec) != hst.n_features:
                await websocket.send_text(
                    json.dumps({"error": f"Expected {hst.n_features} features, got {len(vec)}"})
                )
                continue

            score_info = hst.score_one(vec)
            drift_info = drift.add_observation(vec)

            payload = {
                "anomaly_score": score_info["anomaly_score"],
                "raw_mass": score_info["raw_mass"],
                "drift_status": drift_info["status"],
                "overall_psi": drift_info["overall_psi"],
                "drift_alarm": drift_info.get("alarm", False),
                "has_recalibration_proposal": drift_info.get("has_recalibration_proposal", False),
            }
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
            await websocket.close()
        except Exception:
            pass

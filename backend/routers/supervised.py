"""
backend/routers/supervised.py

FastAPI endpoints for SENTINEL Supervised, Surrogate, and Stacking Anomaly Detection.
Versioned under /api/v2/supervised/.
"""

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.schemas import (
    FidelityReportResponse,
    HybridStackingRequest,
    HybridStackingResponse,
    HyperparameterTuneRequest,
    HyperparameterTuneResponse,
    SupervisedTrainRequest,
    SupervisedTrainResponse,
    SurrogateTrainRequest,
    SurrogateTrainResponse,
)
from backend.state import state
from src.models.supervised.classifier import SupervisedClassifierDetector
from src.models.supervised.stacking import HybridStackingDetector
from src.models.supervised.surrogate import SurrogateDetector
from src.models.supervised.tuning import tune_xgboost_hyperparameters

router = APIRouter(prefix="/api/v2/supervised", tags=["supervised-v2"])


@router.post("/surrogate/train", response_model=SurrogateTrainResponse)
def train_surrogate(req: SurrogateTrainRequest) -> SurrogateTrainResponse:
    """
    Train an XGBoost surrogate on ensemble-derived pseudo-labels using K-fold out-of-fold CV.
    Guarantees no data leakage: scalers and estimators are fit only on training folds.
    """
    if state.X is None or len(state.feature_names) == 0:
        raise HTTPException(
            status_code=400,
            detail="No preprocessed dataset loaded. Please upload or load a dataset first.",
        )

    # Derive ensemble scores if detection hasn't been run yet
    if state.norm_scores is None:
        state.run_detection(use_ensemble=True)

    ensemble_scores = state.norm_scores
    if ensemble_scores is None or len(ensemble_scores) != len(state.X):
        raise HTTPException(
            status_code=500,
            detail="Failed to obtain valid ensemble scores for surrogate training.",
        )

    surrogate = SurrogateDetector(
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        learning_rate=req.learning_rate,
        contamination=req.contamination,
        n_splits=req.n_splits,
        random_state=req.random_state,
    )

    surrogate.fit_from_ensemble(
        X=state.X,
        ensemble_scores=ensemble_scores,
        feature_names=state.feature_names,
    )

    state.surrogate_detector = surrogate
    fidelity_dict = (
        surrogate.fidelity_report_.to_dict()
        if surrogate.fidelity_report_ is not None
        else None
    )
    state.surrogate_fidelity = fidelity_dict

    # Record reproducible experiment
    state.experiment_tracker.record_experiment(
        dataset_name=state.dataset_name or "in_memory",
        df=state.df_raw,
        model_name="Pseudo-Label Surrogate",
        parameters={
            "n_estimators": req.n_estimators,
            "max_depth": req.max_depth,
            "learning_rate": req.learning_rate,
            "contamination": req.contamination,
            "n_splits": req.n_splits,
        },
        results={
            "fit_time_sec": surrogate.get_metadata().get("fit_time_sec", 0.0),
            "n_anomalies": surrogate.get_metadata().get("pseudo_anomaly_count", 0),
            "anomaly_fraction": surrogate.get_metadata().get("pseudo_anomaly_rate", 0.0),
            "fidelity": fidelity_dict,
        },
        random_state=req.random_state,
    )

    meta = surrogate.get_metadata()
    return SurrogateTrainResponse(
        model_type=surrogate.MODEL_TYPE,
        provenance=meta.get("provenance", "surrogate of ensemble"),
        is_ground_truth=False,
        fit_time_sec=meta.get("fit_time_sec", 0.0),
        n_samples=surrogate._n_train_samples,
        n_features=surrogate._n_features,
        pseudo_anomaly_count=meta.get("pseudo_anomaly_count", 0),
        pseudo_anomaly_rate=meta.get("pseudo_anomaly_rate", 0.0),
        pseudo_threshold=meta.get("pseudo_threshold"),
        feature_names=surrogate.feature_names_,
        fidelity=FidelityReportResponse(**fidelity_dict) if fidelity_dict else None,
    )


@router.get("/surrogate/fidelity", response_model=FidelityReportResponse)
def get_surrogate_fidelity() -> FidelityReportResponse:
    """Retrieve the distillation fidelity report comparing surrogate to ensemble."""
    if state.surrogate_fidelity is None:
        raise HTTPException(
            status_code=404,
            detail="Surrogate model has not been trained yet. Call /api/v2/supervised/surrogate/train first.",
        )
    return FidelityReportResponse(**state.surrogate_fidelity)


@router.post("/surrogate/explain")
def explain_surrogate(max_samples: int = 300) -> dict[str, Any]:
    """Compute exact TreeSHAP feature attributions on the trained surrogate model."""
    if state.surrogate_detector is None or not state.surrogate_detector.is_fitted():
        raise HTTPException(
            status_code=400,
            detail="Surrogate model is not fitted. Train surrogate before requesting TreeSHAP.",
        )
    return state.surrogate_detector.explain(state.X, max_samples=max_samples)


@router.post("/classify/train", response_model=SupervisedTrainResponse)
def train_supervised_classifier(req: SupervisedTrainRequest) -> SupervisedTrainResponse:
    """
    Train a supervised anomaly classifier when ground truth label column exists.
    Handles severe class imbalance and optimizes threshold via asymmetric cost matrix.
    """
    if state.df_raw is None or state.X is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset loaded. Upload a dataset with ground truth labels first.",
        )

    if req.label_column not in state.df_raw.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Label column '{req.label_column}' not found in dataset columns: {list(state.df_raw.columns)}",
        )

    raw_labels = state.df_raw[req.label_column].to_numpy()
    # Normalize labels: 1 is anomaly, 0 is normal
    unique_vals = set(np.unique(raw_labels))
    if unique_vals.issubset({0, 1}):
        y = raw_labels.astype(int)
    elif unique_vals.issubset({-1, 1}):
        # Sklearn convention: -1 anomaly, 1 normal
        y = np.where(raw_labels == -1, 1, 0)
    else:
        # Map lowest frequency class to 1 (anomaly)
        counts = pd.Series(raw_labels).value_counts()
        rare_label = counts.index[-1]
        y = (raw_labels == rare_label).astype(int)

    clf = SupervisedClassifierDetector(
        framework=req.framework,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        learning_rate=req.learning_rate,
        time_aware=req.time_aware,
        n_splits=req.n_splits,
        cost_fp=req.cost_fp,
        cost_fn=req.cost_fn,
        random_state=req.random_state,
    )

    clf.fit(X=state.X, y=y, feature_names=state.feature_names)
    state.supervised_classifier = clf

    # Record reproducible experiment
    state.experiment_tracker.record_experiment(
        dataset_name=state.dataset_name or "in_memory",
        df=state.df_raw,
        model_name=f"Supervised {req.framework.upper()}",
        parameters={
            "framework": req.framework,
            "label_column": req.label_column,
            "n_estimators": req.n_estimators,
            "max_depth": req.max_depth,
            "learning_rate": req.learning_rate,
            "cost_fp": req.cost_fp,
            "cost_fn": req.cost_fn,
        },
        results={
            "fit_time_sec": clf.get_metadata().get("fit_time_sec", 0.0),
            "metrics": clf.cv_metrics_,
            "optimal_threshold": clf.optimal_threshold_,
        },
        random_state=req.random_state,
    )

    meta = clf.get_metadata()
    return SupervisedTrainResponse(
        model_type=clf.MODEL_TYPE,
        provenance="ground_truth_supervised",
        is_ground_truth=True,
        fit_time_sec=meta.get("fit_time_sec", 0.0),
        n_samples=clf._n_train_samples,
        n_features=clf._n_features,
        n_positives=meta.get("n_positives", 0),
        n_negatives=meta.get("n_negatives", 0),
        metrics=clf.cv_metrics_,
        cost_matrix=clf.cost_matrix_summary_,
        calibration=clf.calibration_report_,
        optimal_threshold=clf.optimal_threshold_,
    )


@router.post("/stacking/train", response_model=HybridStackingResponse)
def train_hybrid_stacking(req: HybridStackingRequest) -> HybridStackingResponse:
    """
    Train a hybrid stacking meta-learner on out-of-fold rank-normalized scores
    from 8 unsupervised detectors.
    """
    if state.X is None or len(state.feature_names) == 0:
        raise HTTPException(
            status_code=400,
            detail="No preprocessed dataset loaded. Please upload or load a dataset first.",
        )

    # Target label resolution: HybridStacking refuses ensemble pseudo-labels (HTTP 422)
    y: np.ndarray | None = None
    if req.label_column and state.df_raw is not None and req.label_column in state.df_raw.columns:
        y = state.df_raw[req.label_column].to_numpy(dtype=int)
    elif state.y_true is not None:
        y = np.asarray(state.y_true, dtype=int)
    elif state.df_raw is not None and "is_anomaly" in state.df_raw.columns:
        y = state.df_raw["is_anomaly"].to_numpy(dtype=int)
    elif state.df_raw is not None and "Class" in state.df_raw.columns:
        y = state.df_raw["Class"].to_numpy(dtype=int)

    if y is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "HybridStackingDetector refuses ensemble pseudo-label targets (HTTP 422). "
                "Hybrid stacking meta-learners require verified ground-truth labels "
                "(from a user label column or synthetic benchmark lab). Training a meta-learner "
                "on pseudo-labels derived from the same base ensemble creates circular leakage."
            ),
        )

    stacker = HybridStackingDetector(
        passthrough=req.passthrough,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        learning_rate=req.learning_rate,
        n_splits=req.n_splits,
        random_state=req.random_state,
    )

    stacker.fit(X=state.X, y=y, feature_names=state.feature_names)
    state.stacking_detector = stacker

    # Record experiment
    state.experiment_tracker.record_experiment(
        dataset_name=state.dataset_name or "in_memory",
        df=state.df_raw,
        model_name="Hybrid Stacking Meta-Learner",
        parameters={
            "passthrough": req.passthrough,
            "n_estimators": req.n_estimators,
            "max_depth": req.max_depth,
            "learning_rate": req.learning_rate,
            "n_splits": req.n_splits,
        },
        results={
            "fit_time_sec": stacker.get_metadata().get("fit_time_sec", 0.0),
            "meta_features_shape": stacker.get_metadata().get("meta_features_shape"),
        },
        random_state=req.random_state,
    )

    meta = stacker.get_metadata()
    return HybridStackingResponse(
        model_type=stacker.MODEL_TYPE,
        provenance=meta.get("provenance", "hybrid_stacking_ensemble"),
        fit_time_sec=meta.get("fit_time_sec", 0.0),
        n_base_detectors=meta.get("n_base_detectors", 8),
        base_detectors=meta.get("base_detectors", []),
        passthrough=meta.get("passthrough", True),
        meta_features_shape=meta.get("meta_features_shape", []),
    )


@router.post("/tune", response_model=HyperparameterTuneResponse)
def tune_hyperparameters(req: HyperparameterTuneRequest) -> HyperparameterTuneResponse:
    """
    Optuna Bayesian hyperparameter optimization with nested CV to prevent test leakage.
    """
    if state.X is None or len(state.feature_names) == 0:
        raise HTTPException(
            status_code=400,
            detail="No preprocessed dataset loaded. Please upload or load a dataset first.",
        )

    if req.label_column and state.df_raw is not None and req.label_column in state.df_raw.columns:
        raw_labels = state.df_raw[req.label_column].to_numpy()
        unique_vals = set(np.unique(raw_labels))
        if unique_vals.issubset({0, 1}):
            y = raw_labels.astype(int)
        elif unique_vals.issubset({-1, 1}):
            y = np.where(raw_labels == -1, 1, 0)
        else:
            counts = pd.Series(raw_labels).value_counts()
            y = (raw_labels == counts.index[-1]).astype(int)
    else:
        if state.norm_scores is None:
            state.run_detection(use_ensemble=True)
        ensemble_scores = state.norm_scores
        if ensemble_scores is None:
            raise HTTPException(status_code=500, detail="Failed to compute ensemble scores.")
        thresh = float(np.quantile(ensemble_scores, 0.95))
        y = (ensemble_scores >= thresh).astype(int)

    result = tune_xgboost_hyperparameters(
        X=state.X,
        y=y,
        n_trials=req.n_trials,
        cv_splits=req.cv_splits,
        metric=req.metric,
        time_aware=req.time_aware,
        random_state=req.random_state,
    )

    # Persist best tuned params in experiment tracker
    state.experiment_tracker.record_experiment(
        dataset_name=state.dataset_name or "in_memory",
        df=state.df_raw,
        model_name="Optuna XGBoost Tuning",
        parameters={
            "metric": req.metric,
            "n_trials": req.n_trials,
            "cv_splits": req.cv_splits,
            "time_aware": req.time_aware,
            "best_params": result.get("best_params"),
        },
        results={
            "best_score": result.get("best_score"),
            "tuning_engine": result.get("tuning_engine"),
        },
        random_state=req.random_state,
    )

    return HyperparameterTuneResponse(**result)

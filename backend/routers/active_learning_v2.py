"""
backend/routers/active_learning_v2.py

API v2 router for SENTINEL Human-in-the-Loop, Active Learning & Root-Cause Auditing:
- Analyst Feedback Persistence & Curation
- Uncertainty & Disagreement Active Learning Sampling
- Semi-Supervised Model Updating with Zero-Leakage Held-Out Evaluation
- Root-Cause Attribution Clustering (HDBSCAN on SHAP)
- Subgroup Fairness Equity & Adversarial Noise Robustness Auditing
"""

import numpy as np
from fastapi import APIRouter, HTTPException, status

from backend.schemas import (
    ActiveLearningQueueRequest,
    ActiveLearningQueueResponse,
    FeedbackSubmitRequest,
    FeedbackSubmitResponse,
    FeedbackSummaryResponse,
    RobustnessAuditRequest,
    RobustnessAuditResponse,
    RootCauseClusterRequest,
    RootCauseClusterResponse,
    SemiSupervisedRetrainRequest,
    SemiSupervisedRetrainResponse,
)
from backend.state import state
from src.audit.fairness_robustness import (
    evaluate_noise_robustness,
    evaluate_subgroup_fairness,
)
from src.audit.root_cause import cluster_root_causes
from src.learning.feedback_store import AnalystFeedbackStore
from src.learning.query_strategies import query_active_learning_batch
from src.learning.semi_supervised import SemiSupervisedAnomalyUpdater

router = APIRouter(prefix="/api/v2", tags=["active-learning-and-audit"])

# Session feedback store
_feedback_store = AnalystFeedbackStore()


def _check_detection_ready():
    if not state.detection_run or state.X is None or state.norm_scores is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Detection must be executed before requesting active learning or root-cause audits.",
        )


@router.post("/feedback/submit", response_model=FeedbackSubmitResponse)
def submit_feedback(req: FeedbackSubmitRequest) -> FeedbackSubmitResponse:
    """Submit human analyst confirmation (1) or false-positive rejection (0)."""
    _check_detection_ready()

    if req.record_id < 0 or req.record_id >= len(state.X):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Record id {req.record_id} out of bounds (0..{len(state.X)-1}).",
        )

    entry = _feedback_store.add_feedback(
        record_id=req.record_id,
        user_label=req.user_label,
        analyst_id=req.analyst_id,
        notes=req.notes,
    )
    summary = _feedback_store.get_summary_metrics()

    return FeedbackSubmitResponse(
        success=True,
        entry=entry,
        summary_metrics=summary,
    )


@router.get("/feedback/summary", response_model=FeedbackSummaryResponse)
def get_feedback_summary() -> FeedbackSummaryResponse:
    """Retrieve human annotation summary statistics."""
    summary = _feedback_store.get_summary_metrics()
    return FeedbackSummaryResponse(
        total_reviewed=summary["total_reviewed"],
        confirmed_anomalies=summary["confirmed_anomalies"],
        false_positives=summary["false_positives"],
        alert_precision=summary["alert_precision"],
        is_ready_for_semi_supervised=summary["is_ready_for_semi_supervised"],
    )


@router.post("/active-learning/queue", response_model=ActiveLearningQueueResponse)
def get_active_learning_queue(req: ActiveLearningQueueRequest) -> ActiveLearningQueueResponse:
    """Prioritize unlabelled records for review using uncertainty and disagreement sampling."""
    _check_detection_ready()

    reviewed_ids = set(_feedback_store.get_labeled_pairs()[0])

    # Pass all model results for disagreement scoring if available
    detector_scores = {}
    if hasattr(state, "all_model_results") and state.all_model_results:
        for m_name, res in state.all_model_results.items():
            if "norm_scores" in res:
                detector_scores[m_name] = res["norm_scores"]

    candidates = query_active_learning_batch(
        norm_scores=state.norm_scores,
        threshold=state.threshold_val,
        all_detector_scores=detector_scores if len(detector_scores) >= 2 else None,
        X=state.X,
        already_reviewed=reviewed_ids,
        batch_size=req.batch_size,
        strategy=req.strategy,
    )

    return ActiveLearningQueueResponse(
        batch_size=req.batch_size,
        strategy=req.strategy,
        candidates=candidates,
    )


@router.post("/active-learning/retrain-semi-supervised", response_model=SemiSupervisedRetrainResponse)
def retrain_semi_supervised(req: SemiSupervisedRetrainRequest) -> SemiSupervisedRetrainResponse:
    """Update model with human feedback and validate strictly on a held-out evaluation set."""
    _check_detection_ready()

    record_ids, labels = _feedback_store.get_labeled_pairs()
    if len(record_ids) < 4 or len(set(labels)) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 4 analyst reviews containing both confirmed anomalies and false positives.",
        )

    updater = SemiSupervisedAnomalyUpdater()
    try:
        res = updater.update_with_feedback(
            X=state.X,
            unsupervised_scores=state.norm_scores,
            labeled_indices=record_ids,
            labels=labels,
            held_out_ratio=req.held_out_ratio,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Semi-supervised update failed: {exc}") from exc

    return SemiSupervisedRetrainResponse(
        success=True,
        n_labeled_samples=res["n_labeled_samples"],
        n_held_out_unlabelled=res["n_held_out_unlabelled"],
        metrics=res["metrics"],
        evaluation_policy=res["evaluation_policy"],
    )


@router.post("/root-cause/cluster", response_model=RootCauseClusterResponse)
def cluster_anomaly_root_causes(req: RootCauseClusterRequest) -> RootCauseClusterResponse:
    """Group anomalies by feature attribution similarity into named root-cause archetypes."""
    _check_detection_ready()

    anomaly_indices = np.where(state.predictions == -1)[0].tolist()
    if len(anomaly_indices) < req.min_cluster_size:
        # Fallback to top 5% highest scores if few discrete anomaly labels
        cutoff = np.percentile(state.norm_scores, 95)
        anomaly_indices = np.where(state.norm_scores >= cutoff)[0].tolist()

    if len(anomaly_indices) < req.min_cluster_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too few anomalies ({len(anomaly_indices)}) for clustering with min_cluster_size={req.min_cluster_size}.",
        )

    # Compute or retrieve attributions for the anomalies
    score_fn, _ = state._make_score_fn()
    from src.explainability.feature_contributions import perturbation_importance

    attr_rows = []
    # Subsample if large
    eval_indices = anomaly_indices[:150]
    for anom_idx in eval_indices:
        x_row = state.X[anom_idx]
        pert = perturbation_importance(
            observation=x_row,
            score_fn=score_fn,
            feature_names=state.feature_names,
        )
        # Vector in feature order
        feat_dict = {item["feature"]: float(item["importance"]) for item in pert}
        row_vec = [feat_dict.get(name, 0.0) for name in state.feature_names]
        attr_rows.append(row_vec)

    attr_matrix = np.array(attr_rows, dtype=np.float64)

    res = cluster_root_causes(
        attribution_matrix=attr_matrix,
        anomaly_indices=eval_indices,
        feature_names=state.feature_names,
        min_cluster_size=req.min_cluster_size,
    )

    return RootCauseClusterResponse(
        n_clusters=res["n_clusters"],
        unclustered_anomalies=res["unclustered_anomalies"],
        clusters=res["clusters"],
        summary=res["summary"],
    )


@router.post("/robustness/audit", response_model=RobustnessAuditResponse)
def audit_fairness_and_robustness(req: RobustnessAuditRequest) -> RobustnessAuditResponse:
    """Run subgroup parity and adversarial noise perturbation audits."""
    _check_detection_ready()

    fairness_res = None
    if req.group_column:
        if state.df_raw is None or req.group_column not in state.df_raw.columns:
            raise HTTPException(status_code=400, detail=f"Column '{req.group_column}' not found in raw dataset.")
        fairness_res = evaluate_subgroup_fairness(
            df=state.df_raw,
            group_column=req.group_column,
            predictions=state.predictions,
            norm_scores=state.norm_scores,
        )

    noise_res = None
    if req.test_noise:
        score_fn, _ = state._make_score_fn()
        noise_res = evaluate_noise_robustness(
            X=state.X,
            score_fn=score_fn,
            n_samples=min(100, len(state.X)),
        )

    return RobustnessAuditResponse(
        fairness=fairness_res,
        noise_robustness=noise_res,
    )

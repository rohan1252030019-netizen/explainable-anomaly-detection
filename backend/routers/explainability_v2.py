"""
backend/routers/explainability_v2.py

API v2 router for SENTINEL Advanced Explainability Engine:
- Dual-explainer Cross-Validation (LIME vs SHAP Agreement)
- Second-order Feature Interactions
- Cohort Analysis & Subgroup Shifts
- DiCE-style Diverse Counterfactuals (with Plausibility Check)
- SENTINEL Forensic Investigation Narrative
"""

import numpy as np
from fastapi import APIRouter, HTTPException, status

from backend.schemas import (
    AgreementRequest,
    AgreementResponse,
    CohortExplanationRequest,
    CohortExplanationResponse,
    DiverseCounterfactualsRequest,
    DiverseCounterfactualsResponse,
    InteractionsRequest,
    InteractionsResponse,
    NarrativeRequest,
    NarrativeResponse,
)
from backend.state import state
from src.explainability.cohort import explain_cohort
from src.explainability.diverse_counterfactuals import generate_diverse_counterfactuals
from src.explainability.interactions import compute_pairwise_interactions
from src.explainability.lime_explainer import LIMEExplainer, compute_explanation_agreement
from src.explainability.narrative import generate_investigation_narrative

router = APIRouter(prefix="/api/v2/explainability", tags=["explainability-v2"])


def _check_detection_ready():
    if not state.detection_run or state.X is None or state.norm_scores is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Detection must be executed before requesting explanations.",
        )


@router.post("/agreement", response_model=AgreementResponse)
def get_explanation_agreement(req: AgreementRequest) -> AgreementResponse:
    """Evaluate agreement between SHAP and LIME on a specific observation."""
    _check_detection_ready()

    if req.record_index < 0 or req.record_index >= len(state.X):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Record index {req.record_index} out of bounds (0..{len(state.X)-1}).",
        )

    score_fn, detector = state._make_score_fn()
    x_row = state.X[req.record_index]

    # 1. Compute LIME
    lime = LIMEExplainer(
        score_fn=score_fn,
        feature_names=state.feature_names,
        X_background=state.X,
        n_samples=200,
    )
    lime_res = lime.explain_instance(x_row, top_k=req.top_k)
    lime_attr = lime_res["attributions"]

    # 2. Compute SHAP / Perturbation attribution
    shap_attr: dict[str, float] = {}
    try:
        explainer = state.get_explanation_generator()
        if explainer and explainer._shap_explainer and explainer._shap_explainer.is_available():
            shap_vals = explainer._shap_explainer.explain(x_row.reshape(1, -1), max_samples=1)
            if shap_vals is not None:
                shap_attr = {name: float(val) for name, val in zip(state.feature_names, shap_vals[0], strict=False)}
    except Exception:
        pass

    # If SHAP is unavailable or empty, use perturbation importance
    if not shap_attr:
        from src.explainability.feature_contributions import perturbation_importance
        pert = perturbation_importance(
            observation=x_row,
            score_fn=score_fn,
            feature_names=state.feature_names,
        )
        shap_attr = {item["feature"]: float(item["importance"]) for item in pert}

    # 3. Cross-method agreement
    agreement = compute_explanation_agreement(
        attributions_a=shap_attr,
        attributions_b=lime_attr,
        feature_names=state.feature_names,
        top_k=req.top_k,
    )

    return AgreementResponse(
        record_index=req.record_index,
        spearman_rank_correlation=agreement["spearman_rank_correlation"],
        spearman_p_value=agreement["spearman_p_value"],
        top_k_jaccard_overlap=agreement["top_k_jaccard_overlap"],
        sign_concordance=agreement["sign_concordance"],
        agreement_level=agreement["agreement_level"],
        summary=agreement["summary"],
        agreed_top_features=agreement["agreed_top_features"],
        lime_attributions=lime_attr,
        shap_attributions=shap_attr,
    )


@router.post("/interactions", response_model=InteractionsResponse)
def get_feature_interactions(req: InteractionsRequest) -> InteractionsResponse:
    """Discover compound 2nd-order non-linear feature interactions."""
    _check_detection_ready()

    score_fn, detector = state._make_score_fn()

    # Subsample flagged anomalies or top rows
    anomaly_indices = np.where(state.predictions == -1)[0]
    if len(anomaly_indices) > 0:
        sample_pts = state.X[anomaly_indices[:req.max_samples]]
    else:
        sample_pts = state.X[:req.max_samples]

    pairs = compute_pairwise_interactions(
        model_or_score_fn=score_fn,
        X=sample_pts,
        feature_names=state.feature_names,
        top_n_pairs=req.top_n_pairs,
        max_samples=req.max_samples,
    )

    return InteractionsResponse(
        model_name=state.display_model_name,
        n_evaluated_samples=len(sample_pts),
        top_interactions=pairs,
    )


@router.post("/cohort", response_model=CohortExplanationResponse)
def get_cohort_explanation(req: CohortExplanationRequest) -> CohortExplanationResponse:
    """Analyze distribution shifts for a cohort of anomalies vs normal reference data."""
    _check_detection_ready()

    if req.cohort_indices is not None and len(req.cohort_indices) > 0:
        cohort_idx = [i for i in req.cohort_indices if 0 <= i < len(state.X)]
        if not cohort_idx:
            raise HTTPException(status_code=400, detail="No valid indices provided in cohort_indices.")
    else:
        cohort_idx = np.where(state.predictions == -1)[0].tolist()
        if not cohort_idx:
            # Fall back to top 10% highest scores if no discrete predictions
            cutoff = np.percentile(state.norm_scores, 90)
            cohort_idx = np.where(state.norm_scores >= cutoff)[0].tolist()

    normal_idx = np.where(state.predictions == 1)[0].tolist()
    if not normal_idx:
        cutoff = np.percentile(state.norm_scores, 90)
        normal_idx = np.where(state.norm_scores < cutoff)[0].tolist()

    cohort_X = state.X[cohort_idx]
    ref_X = state.X[normal_idx]

    res = explain_cohort(
        cohort_X=cohort_X,
        reference_X=ref_X,
        feature_names=state.feature_names,
        cohort_name=req.cohort_name,
        top_k=req.top_k,
    )

    return CohortExplanationResponse(
        cohort_name=res["cohort_name"],
        n_cohort=res["n_cohort"],
        n_reference=res["n_reference"],
        dominant_drivers=res["dominant_drivers"],
        all_feature_shifts=res["all_feature_shifts"],
        summary=res["summary"],
    )


@router.post("/diverse-counterfactuals", response_model=DiverseCounterfactualsResponse)
def get_diverse_counterfactuals(req: DiverseCounterfactualsRequest) -> DiverseCounterfactualsResponse:
    """Generate k diverse, plausible counterfactual paths to return an anomaly to normality."""
    _check_detection_ready()

    if req.record_index < 0 or req.record_index >= len(state.X):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Record index {req.record_index} out of bounds.",
        )

    obs = state.X[req.record_index]
    obs_score = float(state.norm_scores[req.record_index])

    normal_idx = np.where(state.predictions == 1)[0]
    if len(normal_idx) == 0:
        cutoff = np.percentile(state.norm_scores, 80)
        normal_idx = np.where(state.norm_scores <= cutoff)[0]

    reference_normals = state.X[normal_idx]
    score_fn, _ = state._make_score_fn()

    denorm = state.pipeline.denormalize if state.pipeline else None

    cfs = generate_diverse_counterfactuals(
        observation=obs,
        reference_normals=reference_normals,
        score_fn=score_fn,
        feature_names=state.feature_names,
        k_diverse=req.k_diverse,
        max_features_modified=req.max_features_modified,
        immutable_features=req.immutable_features,
        score_threshold=state.threshold_val,
        denormalize=denorm,
    )

    return DiverseCounterfactualsResponse(
        record_index=req.record_index,
        observed_score=obs_score,
        threshold=state.threshold_val,
        k_candidates_found=len(cfs),
        counterfactuals=cfs,
    )


@router.post("/narrative", response_model=NarrativeResponse)
def get_investigation_narrative(req: NarrativeRequest) -> NarrativeResponse:
    """Generate a factual, tamper-evident SENTINEL Investigation Narrative."""
    _check_detection_ready()

    if req.record_index < 0 or req.record_index >= len(state.X):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Record index {req.record_index} out of bounds.",
        )

    score = float(state.norm_scores[req.record_index])
    thresh = float(state.threshold_val)
    score_fn, _ = state._make_score_fn()

    # Determine severity
    if score >= 0.85:
        severity = "Critical"
    elif score >= thresh:
        severity = "High"
    elif score >= thresh * 0.7:
        severity = "Medium"
    else:
        severity = "Low"

    # Top features via quick perturbation or stored explanations
    x_row = state.X[req.record_index]
    from src.explainability.feature_contributions import perturbation_importance
    pert = perturbation_importance(
        observation=x_row,
        score_fn=score_fn,
        feature_names=state.feature_names,
    )
    top_features = [
        {"feature": item["feature"], "attribution": item["importance"]}
        for item in pert[:5]
    ]

    # Optional conformal p-value if calibrated
    p_val = None
    fdr_sig = None
    if req.include_conformal and hasattr(state, "conformal_calibrator") and state.conformal_calibrator is not None:
        try:
            p_val = float(state.conformal_calibrator.compute_p_values(score))
            fdr_sig = p_val < state.conformal_calibrator.alpha
        except Exception:
            pass

    # Optional counterfactual
    cf_summary = None
    if req.include_counterfactual:
        normal_idx = np.where(state.predictions == 1)[0]
        if len(normal_idx) > 0:
            cfs = generate_diverse_counterfactuals(
                observation=x_row,
                reference_normals=state.X[normal_idx],
                score_fn=score_fn,
                feature_names=state.feature_names,
                k_diverse=1,
                max_features_modified=3,
                score_threshold=thresh,
                denormalize=state.pipeline.denormalize if state.pipeline else None,
            )
            if cfs:
                cf_summary = cfs[0]

    # Top interaction pair
    interactions = compute_pairwise_interactions(
        model_or_score_fn=score_fn,
        X=x_row.reshape(1, -1),
        feature_names=state.feature_names,
        top_n_pairs=1,
    )

    narrative = generate_investigation_narrative(
        record_id=req.record_index,
        anomaly_score=score,
        threshold=thresh,
        severity_tier=severity,
        top_features=top_features,
        conformal_p_value=p_val,
        fdr_significant=fdr_sig,
        interaction_pairs=interactions,
        counterfactual=cf_summary,
        detector_name=state.display_model_name,
    )

    return NarrativeResponse(
        record_index=req.record_index,
        anomaly_score=score,
        threshold=thresh,
        severity_tier=severity,
        narrative_text=narrative["narrative_text"],
        executive_summary=narrative["executive_summary"],
        forensic_fingerprint=narrative["forensic_fingerprint"],
        audit_provenance=narrative["audit_provenance"],
    )

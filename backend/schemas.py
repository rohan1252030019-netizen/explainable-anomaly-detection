"""
backend/schemas.py

Pydantic request *and response* models for the XAUD API.

Two classes of defect are addressed here.

Requests
    Free-form ``dict[str, Any]`` hyperparameters were forwarded straight into
    sklearn constructors.  A typo such as ``n_estimatrs`` was not an error: it
    was silently ignored, the detector trained with defaults, and the caller was
    told their configuration had been applied.  Each detector now has an
    explicit allow-list derived from ``DEFAULT_PARAMS`` and an unknown key is a
    422, not a surprise.

Responses
    No endpoint declared a ``response_model``, so OpenAPI documented every route
    as returning ``{}`` and nothing verified the payload shape.  The models below
    are deliberately *open* (``extra="allow"``) wherever the payload legitimately
    carries dataset columns or free-form report data: they document and validate
    the contract without silently deleting fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

VALID_THRESHOLD_METHODS = ("contamination", "quantile", "fixed")
VALID_AGGREGATIONS = ("rank_borda", "weighted_score")
VALID_ANOMALY_TYPES = ("point", "multivariate", "cluster", "contextual")
VALID_DIFFICULTIES = ("easy", "medium", "hard")
VALID_SCALERS = ("standard", "robust")


def _detector_names() -> list[str]:
    """Registry keys, imported lazily so module import stays cheap."""
    from src.models.model_manager import DETECTOR_REGISTRY

    return list(DETECTOR_REGISTRY)


def _default_params(model_name: str) -> dict[str, Any]:
    from src.models.model_manager import DEFAULT_PARAMS

    return dict(DEFAULT_PARAMS.get(model_name, {}))


class _OpenModel(BaseModel):
    """Base response model that preserves keys it does not declare."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ──────────────────────────────────────────────────────────────────────────────
# Requests
# ──────────────────────────────────────────────────────────────────────────────


class PreprocessRequest(BaseModel):
    scaler_type: Literal["standard", "robust"] = Field(
        default="standard", description="standard or robust"
    )
    label_col: str | None = Field(
        default=None,
        description="Optional ground truth column for evaluation only",
    )
    id_cols: list[str] | None = Field(
        default=[], description="Columns to exclude from training"
    )


class DetectionRequest(BaseModel):
    model_name: str = Field(default="Isolation Forest")
    params: dict[str, Any] | None = Field(default_factory=dict)
    threshold_method: Literal["contamination", "quantile", "fixed"] = Field(
        default="contamination",
        description="contamination, quantile, or fixed",
    )
    contamination: float = Field(default=0.05, ge=0.001, le=0.5)
    quantile_val: float = Field(default=0.95, ge=0.5, le=0.999)
    fixed_val: float = Field(default=0.5, ge=0.0, le=1.0)
    use_ensemble: bool = Field(default=False)
    ensemble_weights: dict[str, float] | None = Field(default=None)

    @field_validator("model_name")
    @classmethod
    def _known_model(cls, value: str) -> str:
        names = _detector_names()
        if value not in names:
            raise ValueError(
                f"Unknown model '{value}'. Available models: {', '.join(names)}."
            )
        return value

    @model_validator(mode="after")
    def _known_hyperparameters(self) -> DetectionRequest:
        """
        Reject hyperparameters the selected detector does not accept.

        This is the difference between "your configuration was applied" and
        "your configuration was typed wrong and ignored".
        """
        params = self.params or {}
        allowed = _default_params(self.model_name)
        unknown = sorted(set(params) - set(allowed))
        if unknown:
            raise ValueError(
                f"Unknown hyperparameter(s) for '{self.model_name}': "
                f"{', '.join(unknown)}. Accepted: "
                f"{', '.join(sorted(allowed)) or '(none)'}."
            )
        return self

    @model_validator(mode="after")
    def _ensemble_weights_match_models(self) -> DetectionRequest:
        if self.ensemble_weights:
            names = _detector_names()
            unknown = sorted(set(self.ensemble_weights) - set(names))
            if unknown:
                raise ValueError(
                    f"ensemble_weights names unknown model(s): {', '.join(unknown)}."
                )
            negative = sorted(k for k, v in self.ensemble_weights.items() if v < 0)
            if negative:
                raise ValueError(
                    f"ensemble_weights must be non-negative; got negative "
                    f"weight for {', '.join(negative)}."
                )
        return self


class ExplanationRequest(BaseModel):
    use_shap: bool = Field(default=False)
    top_k: int = Field(default=8, ge=3, le=20)


class BenchmarkRequest(BaseModel):
    contamination: float = Field(default=0.05, ge=0.001, le=0.5)


class EnsembleRunRequest(BaseModel):
    models: list[str] = Field(
        default=["Isolation Forest", "Local Outlier Factor", "k-Nearest Neighbors"],
        description="List of models to train and ensemble",
    )
    aggregation_method: Literal["rank_borda", "weighted_score"] = Field(
        default="rank_borda", description="'rank_borda' or 'weighted_score'"
    )
    weights: dict[str, float] | None = Field(
        default=None, description="Optional per-model weights"
    )
    min_consensus_votes: int | None = Field(
        default=None, ge=1, description="Minimum agreeing models"
    )

    @field_validator("models")
    @classmethod
    def _known_models(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one model is required.")
        names = _detector_names()
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}.")
        # Preserve caller order, drop duplicates.
        seen: list[str] = []
        for name in value:
            if name not in seen:
                seen.append(name)
        return seen

    @model_validator(mode="after")
    def _weights_belong_to_selected_models(self) -> EnsembleRunRequest:
        if self.weights:
            unknown = sorted(set(self.weights) - set(self.models))
            if unknown:
                raise ValueError(
                    f"weights names model(s) that are not being ensembled: "
                    f"{', '.join(unknown)}."
                )
            negative = sorted(k for k, v in self.weights.items() if v < 0)
            if negative:
                raise ValueError(
                    f"weights must be non-negative; got negative weight for "
                    f"{', '.join(negative)}."
                )
        if (
            self.min_consensus_votes is not None
            and self.min_consensus_votes > len(self.models)
        ):
            raise ValueError(
                "min_consensus_votes cannot exceed the number of models."
            )
        return self


class SyntheticBenchmarkRequest(BaseModel):
    anomaly_type: Literal["point", "multivariate", "cluster", "contextual"] = Field(
        default="multivariate",
        description="'point', 'multivariate', 'cluster', or 'contextual'",
    )
    contamination: float = Field(default=0.05, ge=0.005, le=0.30)
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        description="'easy', 'medium', or 'hard' — scales the injected displacement",
    )
    severity: float = Field(default=3.5, gt=0.0, le=20.0)
    models_to_test: list[str] | None = Field(
        default=None, description="Models to benchmark; if None, tests standard suite"
    )

    @field_validator("models_to_test")
    @classmethod
    def _known_models(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        names = _detector_names()
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}.")
        return value


class DriftEvaluateRequest(BaseModel):
    psi_bins: int = Field(default=10, ge=4, le=50)


class ConstrainedCounterfactualRequest(BaseModel):
    immutable_features: list[str] = Field(
        default=[], description="Features that cannot be changed (e.g. IDs, categories)"
    )
    max_changes: int = Field(default=4, ge=1, le=10)


# ──────────────────────────────────────────────────────────────────────────────
# Responses
# ──────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    framework: str
    version: str
    dataset_loaded: bool
    dataset_name: str | None = None
    detection_ready: bool
    active_model: str
    anomalies_detected: int | None = None
    preprocessed_features: int
    # Deployment posture, surfaced so a client can tell whether the API is
    # protected without probing a protected route.
    auth_required: bool = False


class DatasetSummary(_OpenModel):
    loaded: bool
    dataset_name: str | None = None
    n_rows: int | None = None
    n_cols: int | None = None
    feature_matrix_ready: bool | None = None
    n_preprocessed_features: int | None = None


class DatasetLoadedResponse(BaseModel):
    success: bool
    dataset: DatasetSummary


class SimpleSuccessResponse(BaseModel):
    success: bool
    message: str | None = None


class DatasetProfileResponse(_OpenModel):
    health: dict[str, Any]
    # "schema" shadows the deprecated ``BaseModel.schema`` method, so the
    # attribute is renamed and the wire name is preserved with an alias.
    schema_: dict[str, Any] = Field(alias="schema")
    quality: dict[str, Any]
    collinearity: dict[str, Any]
    recommendations: list[str]


class PreprocessResponse(BaseModel):
    success: bool
    preprocessing: dict[str, Any]


class DetectionModelsResponse(BaseModel):
    models: dict[str, dict[str, Any]]


class DetectionOverview(_OpenModel):
    ready: bool
    model_name: str | None = None
    n_total: int | None = None
    n_anomalies: int | None = None
    anomaly_rate: float | None = None
    message: str | None = None


class DetectionRunResponse(BaseModel):
    success: bool
    overview: DetectionOverview


class EnsembleRunResponse(BaseModel):
    success: bool
    ensemble: dict[str, Any]
    overview: DetectionOverview


class DetectionExperimentsResponse(BaseModel):
    experiments: list[dict[str, Any]]


class AnomalyRecord(_OpenModel):
    rank: int | None = None
    original_index: int | None = None
    normalized_score: float | None = None
    is_anomaly: bool | None = None
    severity: str | None = None
    model: str | None = None


class AnomalyListResponse(BaseModel):
    total: int
    page: int | None = None
    page_size: int | None = None
    total_pages: int | None = None
    items: list[AnomalyRecord]


class InvestigationResponse(_OpenModel):
    row_id: int
    rank: int | None = None
    model_name: str | None = None
    normalized_score: float | None = None
    is_anomaly: bool | None = None
    severity: str | None = None
    threshold: float | None = None
    explanation_text: str | None = None
    top_features: list[dict[str, Any]] | None = None
    feature_comparison: list[dict[str, Any]] | None = None


class GlobalImportanceResponse(_OpenModel):
    ready: bool
    n_explained: int | None = None
    top_driver: str | None = None
    global_importance: list[dict[str, Any]] | None = None
    message: str | None = None


class ExplainabilityGenerateResponse(BaseModel):
    n_explained: int
    top_driver: str | None = None
    global_importance: list[dict[str, Any]]


class StabilityResponse(_OpenModel):
    n_fits: int
    mean_rank_correlation: float | None = None
    top_k_retention: float | None = None
    verdict: str | None = None


class FeatureDistributionResponse(BaseModel):
    feature: str
    normal_median: float
    anomaly_median: float
    separation: float
    distribution: list[dict[str, Any]]


class PcaAnalyticsResponse(BaseModel):
    variance_explained: list[float]
    points: list[dict[str, Any]]


class CorrelationsResponse(BaseModel):
    features: list[str]
    matrix: list[list[float | None]]


class ModelCapability(_OpenModel):
    family: str | None = None
    time_complexity: str | None = None
    space_complexity: str | None = None
    scaling_required: bool | None = None
    sensitive_to_dim: bool | None = None
    explanation_methods: list[str] | None = None
    score_direction: str | None = None
    strengths: str | None = None
    limitations: str | None = None


class AvailableModelsResponse(BaseModel):
    models: list[str]
    capabilities: dict[str, ModelCapability]
    default_params: dict[str, dict[str, Any]]


class BenchmarkResponse(BaseModel):
    comparison_table: list[dict[str, Any]]
    all_model_consensus: int | None = None
    disagreed_count: int | None = None
    agreed_normal: int | None = None
    model_anomaly_counts: list[dict[str, Any]] | None = None
    pairwise_jaccard: list[dict[str, Any]] | None = None


class HoldoutMetrics(_OpenModel):
    available: bool
    reason: str | None = None
    model: str | None = None
    summary: str | None = None
    optimism_gap: dict[str, Any] | None = None


class HoldoutComparisonResponse(_OpenModel):
    test_size: float
    random_state: int
    note: str | None = None
    table: list[dict[str, Any]]


class DriftReportResponse(_OpenModel):
    ready: bool | None = None
    message: str | None = None
    overall_severity: str | None = None
    mean_feature_psi: float | None = None
    n_features_evaluated: int | None = None
    features_high_drift: list[str] | None = None
    features_moderate_drift: list[str] | None = None
    feature_reports: list[dict[str, Any]] | None = None
    score_drift: dict[str, Any] | None = None
    missingness_drift: list[dict[str, Any]] | None = None
    actions: list[str] | None = None
    recommendations: list[str] | None = None


class SyntheticBenchmarkResponse(_OpenModel):
    ready: bool | None = None
    message: str | None = None
    anomaly_type: str | None = None
    contamination: float | None = None
    n_injected: int | None = None
    results_table: list[dict[str, Any]] | None = None
    best_model: str | None = None
    note: str | None = None


class ExperimentRecord(_OpenModel):
    experiment_id: str
    timestamp: str
    dataset_name: str
    # Full SHA-256.  Historic records on disk predate the field, so it is
    # derived from the short hash when absent rather than failing the response.
    dataset_fingerprint: str | None = None
    dataset_hash: str | None = None
    model_name: str
    parameters: dict[str, Any]
    random_state: int | None = None
    threshold_method: str | None = None
    threshold_value: float | None = None
    # Flat copies (legacy) and the nested block the dashboard reads.
    fit_time_sec: float | None = None
    n_anomalies_flagged: int | None = None
    anomaly_rate: float | None = None
    results: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _backfill_fingerprint(self) -> ExperimentRecord:
        """Keep old registry entries readable after the fingerprint change."""
        if not self.dataset_fingerprint:
            self.dataset_fingerprint = self.dataset_hash or "unknown"
        if self.results is None:
            self.results = {
                "fit_time_sec": self.fit_time_sec,
                "n_anomalies": self.n_anomalies_flagged,
                "anomaly_fraction": self.anomaly_rate,
            }
        return self


class AuditReportResponse(_OpenModel):
    title: str
    framework: str
    generated_at: str
    dataset: dict[str, Any]
    detection: dict[str, Any]
    methodology_note: str | None = None


class ErrorResponse(BaseModel):
    """Shape of FastAPI's ``HTTPException`` body, documented in OpenAPI."""

    detail: str


# ──────────────────────────────────────────────────────────────────────────────
# Supervised & Surrogate Models (SENTINEL Phase 1)
# ──────────────────────────────────────────────────────────────────────────────

class FidelityReportResponse(_OpenModel):
    provenance: str = "surrogate of ensemble"
    is_ground_truth: bool = False
    spearman_rank_corr: float
    spearman_pvalue: float
    top_k_overlaps: dict[str, float]
    pseudo_label_accuracy: float
    pseudo_label_f1: float
    surrogate_mean_score: float
    ensemble_mean_score: float
    n_samples: int


class SurrogateTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contamination: float = Field(0.05, ge=0.001, le=0.5, description="Top quantile cutoff for pseudo-labels")
    n_estimators: int = Field(150, ge=10, le=1000)
    max_depth: int = Field(4, ge=1, le=12)
    learning_rate: float = Field(0.05, gt=0, le=1.0)
    n_splits: int = Field(5, ge=2, le=20, description="K-fold out-of-fold cross-validation splits")
    random_state: int = Field(42)


class SurrogateTrainResponse(_OpenModel):
    model_type: str = "surrogate_xgboost"
    provenance: str = "surrogate of ensemble"
    is_ground_truth: bool = False
    fit_time_sec: float
    n_samples: int
    n_features: int
    pseudo_anomaly_count: int
    pseudo_anomaly_rate: float
    pseudo_threshold: float | None = None
    feature_names: list[str]
    fidelity: FidelityReportResponse | None = None


class SupervisedTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label_column: str = Field(..., description="Column containing binary 0=normal, 1=anomaly ground truth")
    framework: Literal["xgboost", "lightgbm", "catboost"] = "xgboost"
    n_estimators: int = Field(150, ge=10, le=1000)
    max_depth: int = Field(4, ge=1, le=12)
    learning_rate: float = Field(0.05, gt=0, le=1.0)
    time_aware: bool = Field(False, description="Use TimeSeriesSplit to preserve temporal sequence")
    cost_fp: float = Field(1.0, ge=0, description="Cost of false positive alarm")
    cost_fn: float = Field(10.0, ge=0, description="Cost of false negative missed anomaly")
    n_splits: int = Field(5, ge=2, le=20)
    random_state: int = Field(42)


class SupervisedTrainResponse(_OpenModel):
    model_type: str = "supervised_xgboost"
    provenance: str = "ground_truth_supervised"
    is_ground_truth: bool = True
    fit_time_sec: float
    n_samples: int
    n_features: int
    n_positives: int
    n_negatives: int
    metrics: dict[str, Any]
    cost_matrix: dict[str, Any]
    calibration: dict[str, Any]
    optimal_threshold: float


class HybridStackingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passthrough: bool = Field(True, description="Concatenate original features with 8 unsupervised meta-features")
    n_estimators: int = Field(150, ge=10, le=1000)
    max_depth: int = Field(4, ge=1, le=12)
    learning_rate: float = Field(0.05, gt=0, le=1.0)
    n_splits: int = Field(5, ge=2, le=20)
    random_state: int = Field(42)


class HybridStackingResponse(_OpenModel):
    model_type: str = "hybrid_stacking"
    provenance: str = "hybrid_stacking_ensemble"
    fit_time_sec: float
    n_base_detectors: int
    base_detectors: list[str]
    passthrough: bool
    meta_features_shape: list[int]


class HyperparameterTuneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label_column: str | None = Field(None, description="Ground truth column; if None, tunes on ensemble pseudo-labels")
    n_trials: int = Field(15, ge=2, le=100)
    metric: Literal["pr_auc", "roc_auc"] = "pr_auc"
    time_aware: bool = False
    cv_splits: int = Field(3, ge=2, le=10)
    random_state: int = Field(42)


class HyperparameterTuneResponse(_OpenModel):
    tuning_engine: str
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    metric: str
    tuning_history: list[dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# Calibrated Uncertainty & Conformal Prediction (SENTINEL Phase 2)
# ──────────────────────────────────────────────────────────────────────────────

class ConformalCalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Significance error budget (1 - confidence)")
    split_ratio: float = Field(0.3, gt=0.05, lt=0.5, description="Fraction of reference split reserved for calibration")
    group_column: str | None = Field(None, description="Optional column for Mondrian group-conditional calibration")


class ConformalCalibrationResponse(_OpenModel):
    is_calibrated: bool
    alpha: float
    confidence_level: float
    n_calibration_samples: int
    threshold_at_alpha: float
    is_mondrian: bool
    groups: list[str] = []
    coverage_statement: str


class ConformalScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float | None = Field(None, gt=0.0, lt=1.0)
    target_fdr: float = Field(0.05, gt=0.0, lt=1.0, description="Target False Discovery Rate for Benjamini-Hochberg")


class ConformalScoreItem(_OpenModel):
    index: int
    score: float
    conformal_p_value: float
    fdr_q_value: float
    is_conformal_anomaly: bool
    is_fdr_significant: bool


class ConformalScoreResponse(_OpenModel):
    total_samples: int
    alpha: float
    confidence_level: float
    target_fdr: float
    conformal_anomalies_flagged: int
    fdr_discoveries_flagged: int
    coverage_statement: str
    fdr_statement: str
    items: list[ConformalScoreItem]


class FDRControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    p_values: list[float] = Field(..., description="Array of p-values to adjust")
    target_fdr: float = Field(0.05, gt=0.0, lt=1.0)


class FDRControlResponse(_OpenModel):
    q_values: list[float]
    is_discovery: list[bool]
    n_discoveries: int
    discovery_fraction: float
    critical_p_cutoff: float
    target_fdr: float
    guarantee_statement: str


# ──────────────────────────────────────────────────────────────────────────────
# Advanced Explainability (SENTINEL Phase 4)
# ──────────────────────────────────────────────────────────────────────────────

class AgreementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_index: int = Field(..., ge=0, description="Row index of anomaly to explain")
    top_k: int = Field(5, ge=1, le=20)


class AgreementResponse(_OpenModel):
    record_index: int
    spearman_rank_correlation: float
    spearman_p_value: float
    top_k_jaccard_overlap: float
    sign_concordance: float
    agreement_level: str
    summary: str
    agreed_top_features: list[str]
    lime_attributions: dict[str, float]
    shap_attributions: dict[str, float]


class InteractionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_n_pairs: int = Field(5, ge=1, le=20)
    max_samples: int = Field(50, ge=10, le=200)


class InteractionPairItem(_OpenModel):
    feature_1: str
    feature_2: str
    interaction_strength: float
    method: str
    interpretation: str


class InteractionsResponse(_OpenModel):
    model_name: str
    n_evaluated_samples: int
    top_interactions: list[InteractionPairItem]


class CohortExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cohort_name: str = Field("Severe Anomalies", description="Label for the target cohort")
    cohort_indices: list[int] | None = Field(None, description="Optional explicit row indices; if None, uses flagged anomalies")
    top_k: int = Field(5, ge=1, le=20)


class CohortShiftItem(_OpenModel):
    feature: str
    cohort_mean: float
    reference_mean: float
    standardized_shift: float
    effect_size_d: float
    ks_statistic: float
    ks_pvalue: float
    direction: str


class CohortExplanationResponse(_OpenModel):
    cohort_name: str
    n_cohort: int
    n_reference: int
    dominant_drivers: list[CohortShiftItem]
    all_feature_shifts: list[CohortShiftItem]
    summary: str


class DiverseCounterfactualsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_index: int = Field(..., ge=0)
    k_diverse: int = Field(3, ge=1, le=5)
    max_features_modified: int = Field(3, ge=1, le=10)
    immutable_features: list[str] = Field(default_factory=list)


class DiverseCounterfactualItem(_OpenModel):
    candidate_id: int
    target_score: float
    achieves_normality: bool
    is_in_distribution_plausible: bool
    sparsity_l0: int
    l2_distance: float
    modifications: list[dict[str, Any]]


class DiverseCounterfactualsResponse(_OpenModel):
    record_index: int
    observed_score: float
    threshold: float
    k_candidates_found: int
    counterfactuals: list[DiverseCounterfactualItem]


class NarrativeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_index: int = Field(..., ge=0)
    include_conformal: bool = True
    include_counterfactual: bool = True


class NarrativeResponse(_OpenModel):
    record_index: int
    anomaly_score: float
    threshold: float
    severity_tier: str
    narrative_text: str
    executive_summary: str
    forensic_fingerprint: str
    audit_provenance: dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Time-Series & Streaming (SENTINEL Phase 5)
# ──────────────────────────────────────────────────────────────────────────────

class STLDecomposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str | None = Field(None, description="Target column; if None, defaults to first numeric column")
    period: int = Field(7, ge=2, le=365, description="Seasonal cycle period")
    threshold: float = Field(3.0, ge=1.0, le=10.0, description="Robust Z-score residual threshold")


class STLDecomposeResponse(_OpenModel):
    period: int
    n_samples: int
    n_anomalies: int
    anomaly_fraction: float
    trend: list[float]
    seasonal: list[float]
    residual: list[float]
    anomaly_scores: list[float]
    anomaly_indices: list[int]


class RollingFeaturesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_columns: list[str] = Field(..., description="Columns to engineer causal rolling features on")
    windows: list[int] = Field(default_factory=lambda: [5, 15])
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3])
    include_ema: bool = True


class RollingFeaturesResponse(_OpenModel):
    original_columns: list[str]
    engineered_columns: list[str]
    n_rows: int
    is_causal_verified: bool


class ChangepointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str | None = Field(None, description="Target column; if None, defaults to first numeric column")
    method: Literal["pelt", "bocpd"] = "pelt"
    penalty: float | None = Field(None, description="Complexity penalty for PELT; if None, uses BIC (2*log(N))")
    hazard_rate: float = Field(0.02, gt=0.0, lt=1.0, description="Expected changepoint frequency for BOCPD")


class ChangepointResponse(_OpenModel):
    method: str
    n_changepoints: int
    changepoints: list[int]
    details: dict[str, Any]


class StreamingIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: list[float] = Field(..., description="Vector of feature values for incoming stream observation")


class StreamingIngestResponse(_OpenModel):
    anomaly_score: float
    raw_mass: float
    total_samples_processed: int
    drift_status: str
    overall_psi: float
    drift_alarm: bool
    has_recalibration_proposal: bool
    recalibration_proposal: dict[str, Any] | None = None


class RecalibrationApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool = Field(..., description="True to recalibrate baseline to current stream; False to dismiss")


class RecalibrationApprovalResponse(_OpenModel):
    success: bool
    action: str
    proposal_id: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Active Learning & Root-Cause Audit (SENTINEL Phase 6)
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: int = Field(..., ge=0, description="Row index of anomaly to annotate")
    user_label: int = Field(..., description="1 = Confirmed True Anomaly, 0 = False Positive / Normal")
    analyst_id: str = Field("analyst_1", description="Analyst identifier")
    notes: str = Field("", description="Optional justification note")


class FeedbackSubmitResponse(_OpenModel):
    success: bool
    entry: dict[str, Any]
    summary_metrics: dict[str, Any]


class FeedbackSummaryResponse(_OpenModel):
    total_reviewed: int
    confirmed_anomalies: int
    false_positives: int
    alert_precision: float
    is_ready_for_semi_supervised: bool


class ActiveLearningQueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(10, ge=1, le=50)
    strategy: Literal["hybrid", "uncertainty", "disagreement"] = "hybrid"


class ActiveLearningQueueItem(_OpenModel):
    index: int
    score: float
    uncertainty_score: float
    disagreement_score: float
    priority_score: float
    query_reason: str


class ActiveLearningQueueResponse(_OpenModel):
    batch_size: int
    strategy: str
    candidates: list[ActiveLearningQueueItem]


class SemiSupervisedRetrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    held_out_ratio: float = Field(0.25, gt=0.05, lt=0.5, description="Fraction of unlabelled pool strictly reserved for held-out evaluation")


class SemiSupervisedRetrainResponse(_OpenModel):
    success: bool
    n_labeled_samples: int
    n_held_out_unlabelled: int
    metrics: dict[str, Any]
    evaluation_policy: str


class RootCauseClusterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_cluster_size: int = Field(3, ge=2, le=50)


class RootCauseClusterItem(_OpenModel):
    cluster_id: int
    cluster_name: str
    size: int
    fraction_of_anomalies: float
    dominant_features: list[dict[str, Any]]
    exemplar_record_index: int
    member_record_indices: list[int]


class RootCauseClusterResponse(_OpenModel):
    n_clusters: int
    unclustered_anomalies: int
    clusters: list[RootCauseClusterItem]
    summary: str


class RobustnessAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_column: str | None = Field(None, description="Sensitive attribute column to evaluate subgroup parity on")
    test_noise: bool = Field(True, description="Evaluate Gaussian noise perturbation sensitivity")


class RobustnessAuditResponse(_OpenModel):
    fairness: dict[str, Any] | None = None
    noise_robustness: dict[str, Any] | None = None






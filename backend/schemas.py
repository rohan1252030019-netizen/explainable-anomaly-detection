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

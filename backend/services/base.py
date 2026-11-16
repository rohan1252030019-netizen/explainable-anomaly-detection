"""
backend/services/base.py

Declarations shared by every state service mixin.

``AppState`` is composed from mixins that all read and write the same set of
cached fields.  Splitting those methods into separate modules therefore needs a
single place that states what ``self`` is: without it a type checker cannot see
that ``DetectionServiceMixin`` may touch ``self.X`` because a *different* mixin
assigned it.

``ServiceState`` carries annotations only — no values are ever assigned here, so
instantiation behaviour is unchanged and ``AppState._init_defaults`` remains the
single place where runtime defaults are set.  The four cross-mixin method stubs
at the bottom are implemented by the mixins that own them.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from backend.experiments import ExperimentTracker
from src.data.loader import DataLoader
from src.data.preprocessing import PreprocessingPipeline
from src.detection.scoring import ScoreNormalizer
from src.models.model_manager import ModelManager

_F = TypeVar("_F", bound=Callable[..., Any])


def serialized(method: _F) -> _F:
    """
    Run ``method`` while holding the state mutation lock.

    ``AppState`` is one mutable object shared by every request.  FastAPI runs
    synchronous endpoints in a thread pool, so two requests can be inside
    ``run_detection`` at the same time and interleave their writes to
    ``self.predictions``, ``self.X`` or ``self.active_model``.  The failure mode
    is not a crash: it is a result set that belongs to neither request, which is
    worse.

    The lock is re-entrant because detection internally calls ``reset_detection``
    and the reporting path calls several readers.  It is deliberately coarse:
    this is a single-session, in-memory application, and correctness beats
    parallel throughput here.  A deployment that needs concurrent sessions
    should hold one ``AppState`` per session (see ``backend.session``).
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        lock = getattr(self, "_mutation_lock", None)
        if lock is None:  # pragma: no cover - defensive
            return method(self, *args, **kwargs)
        with lock:
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class ServiceState:
    """Typing-only description of the shared ``AppState`` fields."""

    # Dataset
    df_raw: pd.DataFrame | None
    df_results: pd.DataFrame | None
    dataset_name: str
    loader: DataLoader | None
    validator_report: dict[str, Any] | None
    pipeline: PreprocessingPipeline | None
    X: np.ndarray | None
    feature_names: list[str]
    y_true: np.ndarray | None
    label_col: str | None
    id_cols: list[str]
    scaler_type: str
    profiler_report: dict[str, Any] | None

    # Models and detection
    model_manager: ModelManager
    active_model: str
    ensemble_models: list[str]
    norm_scores: np.ndarray | None
    predictions: np.ndarray | None
    severity: np.ndarray | None
    threshold_val: float
    threshold_method: str
    contamination: float
    detection_run: bool
    normalizer: ScoreNormalizer | None
    ensemble_report: dict[str, Any] | None
    all_model_results: dict[str, Any]
    comparison_df: pd.DataFrame | None
    severity_bands: dict[str, Any] | None

    # Explanations
    explanations: list[dict[str, Any]]
    global_importance: pd.DataFrame | None
    explanation_run: bool

    # Evaluation and analytics
    holdout_report: dict[str, Any] | None
    holdout_table: pd.DataFrame | None
    pca_coordinates: np.ndarray | None
    pca_variance_ratio: list[float] | None
    drift_report: dict[str, Any] | None
    benchmark_report: dict[str, Any] | None
    experiment_history: list[dict[str, Any]]
    experiment_tracker: ExperimentTracker

    @property
    def display_model_name(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    # ── Cross-mixin calls ────────────────────────────────────────────────────
    # Implemented in dataset.py, detection.py and explainability.py.  Declared
    # here so a mixin that calls them type-checks in isolation.

    def get_dataset_summary(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def get_detection_overview(self) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def reset_detection(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_anomalies(
        self,
        search: str | None = None,
        severity_filter: list[str] | None = None,
        min_score: float = 0.0,
        max_score: float = 1.0,
        anomalies_only: bool = True,
        sort_by: str = "rank",
        ascending: bool = True,
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

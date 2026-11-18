"""
backend/state.py

Thread-safe in-memory state manager for SENTINEL.

``AppState`` used to be a single ~1,500-line class that loaded datasets,
trained detectors, computed explanations, ran the benchmark lab and wrote
the audit report.  Those concerns now live in ``backend/services/`` as
mixins over the same instance; this module keeps what genuinely belongs to
a state object: construction, the cached analytics fields, and reset.

Every public method the API routers call is still reachable on
``state``, so the split is invisible to callers.
"""

import os
import threading
from typing import Any

import numpy as np
import pandas as pd

from backend.experiments import ExperimentTracker
from backend.services import (
    AnalyticsServiceMixin,
    DatasetServiceMixin,
    DetectionServiceMixin,
    EvaluationServiceMixin,
    ExplainabilityServiceMixin,
    ReportingServiceMixin,
)
from backend.services.base import serialized
from backend.services.util import (
    PROJECT_ROOT,
)
from src.data.loader import DataLoader
from src.data.preprocessing import PreprocessingPipeline
from src.detection.scoring import ScoreNormalizer
from src.models.model_manager import (
    ModelManager,
)

__all__ = ["AppState", "state", "PROJECT_ROOT"]


class AppState(
    DatasetServiceMixin,
    DetectionServiceMixin,
    ExplainabilityServiceMixin,
    EvaluationServiceMixin,
    AnalyticsServiceMixin,
    ReportingServiceMixin,
):
    """
    One analysis session: a dataset, its preprocessing, results and caches.

    This used to be a hard singleton.  That made the process's state impossible
    to isolate (every test and every tenant shared one object) and it meant a
    second ``AppState()`` silently returned the first.  It is now an ordinary
    class; ``state`` below is the process-wide default session, exactly as
    before for every existing caller, and ``backend.session.SessionRegistry``
    can create additional independent ones.

    ``_mutation_lock`` serialises mutations so two concurrent requests cannot
    interleave writes to the same session.
    """

    _mutation_lock = threading.RLock()

    def __init__(self) -> None:
        self._init_defaults()

    def _init_defaults(self):
        self.df_raw: pd.DataFrame | None = None
        self.df_results: pd.DataFrame | None = None
        self.dataset_name: str = ""
        self.loader: DataLoader | None = None
        self.validator_report: dict[str, Any] | None = None
        self.pipeline: PreprocessingPipeline | None = None
        self.X: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.y_true: np.ndarray | None = None
        self.model_manager: ModelManager = ModelManager(models_dir=os.path.join(PROJECT_ROOT, "models"))
        self.active_model: str = "Isolation Forest"
        # Registry name of every detector contributing to the current consensus
        # run.  Empty when the last detection was a single model.
        self.ensemble_models: list[str] = []
        self.norm_scores: np.ndarray | None = None
        self.predictions: np.ndarray | None = None
        self.severity: np.ndarray | None = None
        self.threshold_val: float = 0.5
        self.threshold_method: str = "contamination"
        self.contamination: float = 0.05
        self.explanations: list[dict[str, Any]] = []
        self.global_importance: pd.DataFrame | None = None
        self.all_model_results: dict[str, Any] = {}
        self.comparison_df: pd.DataFrame | None = None
        self.experiment_history: list[dict[str, Any]] = []
        self.label_col: str | None = None
        self.id_cols: list[str] = []
        self.scaler_type: str = "standard"
        self.detection_run: bool = False
        self.explanation_run: bool = False
        self.pca_coordinates: np.ndarray | None = None
        self.pca_variance_ratio: list[float] | None = None
        self.normalizer: ScoreNormalizer | None = None
        self.holdout_report: dict[str, Any] | None = None
        self.holdout_table: pd.DataFrame | None = None
        self.severity_bands: dict[str, Any] | None = None
        self.profiler_report: dict[str, Any] | None = None
        self.ensemble_report: dict[str, Any] | None = None
        self.drift_report: dict[str, Any] | None = None
        self.benchmark_report: dict[str, Any] | None = None
        self.surrogate_detector: Any = None
        self.supervised_classifier: Any = None
        self.stacking_detector: Any = None
        self.surrogate_fidelity: dict[str, Any] | None = None
        self.conformal_calibrator: Any = None
        self.conformal_results: dict[str, Any] | None = None
        self.experiment_tracker: ExperimentTracker = ExperimentTracker(
            log_dir=os.path.join(PROJECT_ROOT, "outputs", "experiments")
        )

    @serialized
    def reset_all(self):
        """Clear entire application state."""
        self._init_defaults()


# Singleton accessor
state = AppState()

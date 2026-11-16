"""
backend/services/dataset.py

Dataset lifecycle: sample generation, CSV ingestion, preprocessing and profiling.

Extracted verbatim from the former monolithic ``backend/state.py``.  The
methods still read and write the shared :class:`backend.state.AppState`
instance through ``self``; the split is organisational, and it lets a
reviewer read one concern at a time.
"""

import os
from typing import Any

import numpy as np

from backend.services.base import ServiceState, serialized
from backend.services.util import PROJECT_ROOT
from src.data.labels import to_binary_labels
from src.data.loader import DataLoader, generate_sample_dataset
from src.data.preprocessing import PreprocessingPipeline
from src.data.profiler import DataProfiler
from src.data.validator import DataValidator


class DatasetServiceMixin(ServiceState):
    """Dataset lifecycle: sample generation, CSV ingestion, preprocessing and profiling."""

    @serialized
    def load_sample_dataset(self) -> dict[str, Any]:
        """Load or create the benchmark sample dataset."""
        sample_path = os.path.join(PROJECT_ROOT, "data", "sample", "sample_anomaly_data.csv")
        if not os.path.exists(sample_path):
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            df = generate_sample_dataset(n_normal=800, n_anomalies=80, random_state=42)
            df.to_csv(sample_path, index=False)

        loader = DataLoader()
        df = loader.load_csv(sample_path)
        self.df_raw = df
        self.dataset_name = "sample_anomaly_data.csv"
        self.loader = loader

        validator = DataValidator()
        self.validator_report = validator.validate(df)
        self.profiler_report = None

        # Reset downstream
        self.reset_detection()
        self.X = None
        self.feature_names = []
        self.pipeline = None

        # Automatically fit default preprocessing
        self.preprocess_dataset()

        return self.get_dataset_summary()

    @serialized
    def load_user_csv(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        """Load uploaded CSV file bytes."""
        loader = DataLoader()
        df = loader.load_from_bytes(file_bytes, filename)
        self.df_raw = df
        self.dataset_name = filename
        self.loader = loader

        validator = DataValidator()
        self.validator_report = validator.validate(df)
        self.profiler_report = None

        self.reset_detection()
        self.X = None
        self.feature_names = []
        self.pipeline = None

        # Automatically run preprocessing with standard defaults
        self.preprocess_dataset()

        return self.get_dataset_summary()

    @serialized
    def preprocess_dataset(
        self,
        scaler_type: str = "standard",
        label_col: str | None = None,
        id_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fit preprocessing pipeline and prepare feature matrix X."""
        if self.df_raw is None:
            raise ValueError("No dataset loaded.")

        self.scaler_type = scaler_type
        self.id_cols = id_cols or []

        # Auto-detect label column if not specified
        if label_col is None:
            summary = self.loader.get_summary() if self.loader else {}
            potential_label = summary.get("potential_label_col")
            if potential_label and potential_label in self.df_raw.columns:
                label_col = potential_label
            elif "label" in self.df_raw.columns:
                label_col = "label"

        self.label_col = label_col
        if label_col and label_col in self.df_raw.columns:
            # One shared mapping for {0,1} and {-1,+1} columns.  ``== 1`` picks
            # the *normal* rows out of a {-1,+1} column, silently inverting
            # every metric computed against y_true.
            self.y_true = to_binary_labels(self.df_raw[label_col])
            if self.y_true is None:
                self.validator_report = self.validator_report or {}
                self.validator_report.setdefault("warnings", []).append(
                    f"Label column '{label_col}' does not follow the 0/1 or "
                    f"-1/+1 convention; its values are ignored for evaluation."
                )
        else:
            self.y_true = None

        pipeline = PreprocessingPipeline(
            label_col=self.label_col,
            scaler_type=self.scaler_type,
            drop_duplicates=False,
            id_cols=self.id_cols,
        )
        X, feature_names = pipeline.fit_transform(self.df_raw)
        self.pipeline = pipeline
        self.X = X
        self.feature_names = feature_names

        # Clear any prior detection
        self.reset_detection()

        return {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": feature_names,
            "scaler_type": scaler_type,
            "label_col": self.label_col,
            "has_labels": self.y_true is not None,
            "pipeline_log": pipeline.get_log(),
        }

    def get_dataset_summary(self) -> dict[str, Any]:
        """Return dataset summary and column profiling."""
        if self.df_raw is None:
            return {"loaded": False}

        summary = self.loader.get_summary() if self.loader else {}
        col_info = self.loader.get_column_info().to_dict(orient="records") if self.loader else []

        # Preview top 10 rows safely
        preview_df = self.df_raw.head(10).copy()
        # Clean NaNs and Infinities for JSON serialization
        preview_records = preview_df.replace({np.nan: None}).to_dict(orient="records")

        return {
            "loaded": True,
            "dataset_name": self.dataset_name,
            "n_rows": int(self.df_raw.shape[0]),
            "n_cols": int(self.df_raw.shape[1]),
            "numerical_cols": summary.get("numerical_cols", []),
            "categorical_cols": summary.get("categorical_cols", []),
            "total_missing": int(summary.get("total_missing", 0)),
            "duplicate_rows": int(summary.get("duplicate_rows", 0)),
            "memory_mb": float(summary.get("memory_mb", 0.0)),
            "columns": col_info,
            "preview": preview_records,
            "feature_matrix_ready": self.X is not None,
            "n_preprocessed_features": len(self.feature_names),
            "label_col": self.label_col,
            "warnings": self.validator_report.get("warnings", []) if self.validator_report else [],
        }

    # ── Ensemble helpers ──────────────────────────────────────────────────

    @serialized
    def get_profiler_report(self) -> dict[str, Any]:
        """Return cached or compute fresh dataset intelligence profile."""
        if self.df_raw is None:
            raise ValueError("No dataset loaded.")
        if self.profiler_report is None:
            profiler = DataProfiler()
            self.profiler_report = profiler.profile(self.df_raw)
        return self.profiler_report

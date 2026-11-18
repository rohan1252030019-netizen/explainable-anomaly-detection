"""
backend/experiments.py

Reproducibility and Experiment Tracking Engine for SENTINEL.

Provides:
- Cryptographic SHA-256 dataset fingerprinting.
- Immutable experiment recording: algorithm, hyperparameters, seeds, execution times.
- Evaluation metrics and explanation fidelity logging.
- JSON-based persistent experiment registry in outputs/experiments/.
"""

import datetime
import hashlib
import json
import os
import platform
import uuid
from typing import Any

import numpy as np
import pandas as pd
import sklearn


class ExperimentTracker:
    """
    Manages reproducible experiment logs and artifact metadata.
    """

    def __init__(self, log_dir: str = "outputs/experiments"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "experiment_history.json")
        self._history: list[dict[str, Any]] = self._load_history()

    def _load_history(self) -> list[dict[str, Any]]:
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self) -> None:
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
        """
        Full 64-character SHA-256 over values and column names.

        Truncating to 16 hex characters (64 bits) is fine for a UI badge but is
        not a fingerprint you can certify a result against: collisions become
        plausible at the scale of a shared registry.  The full digest is stored
        as ``dataset_fingerprint``; the 16-character prefix stays as
        ``dataset_hash`` for clients that were written against it.
        """
        if df is None or df.empty:
            return "empty"
        data_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        col_bytes = "".join(str(c) for c in df.columns).encode("utf-8")
        return hashlib.sha256(data_bytes + col_bytes).hexdigest()

    @staticmethod
    def compute_dataset_hash(df: pd.DataFrame) -> str:
        """Short 16-character prefix of the dataset fingerprint (legacy field)."""
        digest = ExperimentTracker.compute_dataset_fingerprint(df)
        return digest if digest == "empty" else digest[:16]

    def record_experiment(
        self,
        dataset_name: str,
        df: pd.DataFrame | None,
        model_name: str,
        parameters: dict[str, Any],
        results: dict[str, Any],
        threshold_method: str = "contamination",
        threshold_val: float = 0.05,
        holdout_report: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Record and persist an experiment run.
        """
        exp_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        fingerprint = self.compute_dataset_fingerprint(df) if df is not None else "unknown"
        short_hash = self.compute_dataset_hash(df) if df is not None else "unknown"

        fit_time = results.get("fit_time_sec", 0.0)
        n_anomalies = results.get("n_anomalies", 0)
        anomaly_rate = results.get("anomaly_fraction", 0.0)

        entry = {
            "experiment_id": exp_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset_name": dataset_name,
            # Full SHA-256 plus the historical prefix.
            "dataset_fingerprint": fingerprint,
            "dataset_hash": short_hash,
            "n_samples": int(df.shape[0]) if df is not None else 0,
            "n_features": int(df.shape[1]) if df is not None else 0,
            "model_name": model_name,
            "parameters": parameters,
            "random_state": random_state,
            "threshold_method": threshold_method,
            "threshold_value": threshold_val,
            # Flat copies kept for clients written against the older payload …
            "fit_time_sec": fit_time,
            "n_anomalies_flagged": n_anomalies,
            "anomaly_rate": anomaly_rate,
            # … and the nested block the dashboard reads.  Both come from the
            # same values, so they cannot drift.
            "results": {
                "fit_time_sec": fit_time,
                "n_anomalies": n_anomalies,
                "anomaly_fraction": anomaly_rate,
            },
            "holdout_optimism_gap": holdout_report.get("optimism_gap") if holdout_report else None,
            "software_environment": {
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
            },
        }

        self._history.insert(0, entry)
        # Cap log at 200 entries
        if len(self._history) > 200:
            self._history = self._history[:200]

        self._save_history()
        return entry

    def list_experiments(self) -> list[dict[str, Any]]:
        return list(self._history)

    def get_experiment(self, exp_id: str) -> dict[str, Any] | None:
        for exp in self._history:
            if exp["experiment_id"] == exp_id:
                return exp
        return None

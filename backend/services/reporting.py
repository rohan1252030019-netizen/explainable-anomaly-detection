"""
backend/services/reporting.py

Audit report assembly and experiment history.

Extracted verbatim from the former monolithic ``backend/state.py``.  The
methods still read and write the shared :class:`backend.state.AppState`
instance through ``self``; the split is organisational, and it lets a
reviewer read one concern at a time.
"""

from typing import Any

from backend.services.base import ServiceState, serialized


class ReportingServiceMixin(ServiceState):
    """Audit report assembly and experiment history."""

    @serialized
    def generate_report(self) -> dict[str, Any]:
        """Generate presentation-ready academic and technical audit report."""
        import datetime
        summary = self.get_dataset_summary()
        overview = self.get_detection_overview()

        anomalies_data = self.get_anomalies(anomalies_only=True, page=1, page_size=10)

        global_drivers = []
        if self.global_importance is not None and not self.global_importance.empty:
            global_drivers = self.global_importance.head(5).to_dict(orient="records")

        return {
            "title": "Explainable Anomaly Detection Audit Report",
            "framework": "SENTINEL Autonomous Anomaly Intelligence Platform",
            "generated_at": datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S"),
            "dataset": {
                "name": self.dataset_name,
                "rows": summary.get("n_rows", 0),
                "columns": summary.get("n_cols", 0),
                "features_preprocessed": len(self.feature_names),
                "scaler": self.scaler_type,
            },
            "detection": {
                "model": self.display_model_name,
                "threshold_method": self.threshold_method,
                "threshold_value": self.threshold_val,
                "total_observations": overview.get("n_total", 0),
                "anomalies_detected": overview.get("n_anomalies", 0),
                "anomaly_rate": overview.get("anomaly_rate", 0),
                "severity_distribution": overview.get("severity_counts", {}),
            },
            "top_drivers": global_drivers,
            "sample_anomalies": anomalies_data.get("items", []),
            "methodology_note": (
                "Unsupervised anomaly detection identifies patterns that deviate statistically from "
                "the majority distribution without requiring target labels. Feature attributions indicate "
                "relative deviation magnitudes and model sensitivities, serving as decision support."
            ),
        }

    def get_experiments_history(self) -> list[dict[str, Any]]:
        return self.experiment_tracker.list_experiments()

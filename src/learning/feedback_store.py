"""
src/learning/feedback_store.py

Analyst feedback storage and curation for human-in-the-loop anomaly intelligence.

Maintains human decisions (confirmed anomaly vs false positive) per session or persistent disk.
Tracks analyst consensus, precision metrics, and provides training sets for active learning.
"""

import time
from typing import Any


class AnalystFeedbackStore:
    """
    Session-persisted analyst feedback storage.

    Parameters
    ----------
    session_id : str, default="default"
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        # Map record_id -> dict of feedback details
        self._feedback: dict[int, dict[str, Any]] = {}

    def add_feedback(
        self,
        record_id: int,
        user_label: int,  # 1 = True Anomaly, 0 = False Positive / Normal
        analyst_id: str = "analyst_1",
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Record analyst feedback for a specific observation.

        Parameters
        ----------
        record_id : int
        user_label : int (1 for confirmed anomaly, 0 for false positive)
        analyst_id : str
        notes : str
        """
        if user_label not in (0, 1):
            raise ValueError(f"user_label must be 0 (Normal / False Positive) or 1 (Confirmed Anomaly), got {user_label}")

        entry = {
            "record_id": int(record_id),
            "user_label": int(user_label),
            "label_name": "Confirmed Anomaly" if user_label == 1 else "False Positive",
            "analyst_id": str(analyst_id),
            "notes": str(notes),
            "timestamp": time.time(),
        }
        self._feedback[int(record_id)] = entry
        return entry

    def get_feedback(self, record_id: int) -> dict[str, Any] | None:
        return self._feedback.get(int(record_id))

    def get_all_feedback(self) -> list[dict[str, Any]]:
        return list(self._feedback.values())

    def get_labeled_pairs(self) -> tuple[list[int], list[int]]:
        """Return (list of record_ids, list of binary labels)."""
        record_ids = list(self._feedback.keys())
        labels = [self._feedback[r]["user_label"] for r in record_ids]
        return record_ids, labels

    def get_summary_metrics(self) -> dict[str, Any]:
        """Compute summary statistics of human analyst annotations."""
        total = len(self._feedback)
        if total == 0:
            return {
                "total_reviewed": 0,
                "confirmed_anomalies": 0,
                "false_positives": 0,
                "alert_precision": 0.0,
                "is_ready_for_semi_supervised": False,
            }

        confirmed = sum(1 for e in self._feedback.values() if e["user_label"] == 1)
        fps = total - confirmed
        precision = confirmed / total if total > 0 else 0.0

        # Ready for semi-supervised update if at least 6 labeled points with >= 1 positive and >= 1 negative
        ready = total >= 6 and confirmed >= 1 and fps >= 1

        return {
            "total_reviewed": total,
            "confirmed_anomalies": confirmed,
            "false_positives": fps,
            "alert_precision": round(precision, 4),
            "is_ready_for_semi_supervised": ready,
        }

    def clear(self) -> None:
        self._feedback.clear()

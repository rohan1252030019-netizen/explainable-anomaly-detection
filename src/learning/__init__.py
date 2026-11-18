"""
src/learning package

SENTINEL Human-in-the-Loop & Active Learning Engine:
- Analyst Feedback Store & Annotation Metrics
- Uncertainty, Disagreement, and Hybrid Query Strategies
- Semi-Supervised Model Updater with Zero-Leakage Held-Out Evaluation
"""

from src.learning.feedback_store import AnalystFeedbackStore
from src.learning.query_strategies import query_active_learning_batch
from src.learning.semi_supervised import SemiSupervisedAnomalyUpdater

__all__ = [
    "AnalystFeedbackStore",
    "query_active_learning_batch",
    "SemiSupervisedAnomalyUpdater",
]

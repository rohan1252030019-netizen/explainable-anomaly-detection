"""
tests/test_ensemble.py

Unit tests for IntelligentEnsembleEngine:
- Rank-based Borda aggregation
- Weighted score aggregation
- Consensus count and strength
- Model disagreement metrics (Jaccard and Spearman)
"""

import numpy as np
import pytest

from src.detection.ensemble import IntelligentEnsembleEngine


@pytest.fixture
def mock_detector_results() -> dict[str, dict[str, np.ndarray]]:
    """Simulate 3 models on 100 observations."""
    rng = np.random.default_rng(42)
    n = 100

    # Model 1
    scores_1 = rng.uniform(0, 0.5, n)
    scores_1[-5:] = rng.uniform(0.8, 1.0, 5)
    preds_1 = np.ones(n, dtype=int)
    preds_1[-5:] = -1

    # Model 2: agrees on last 5, plus flags index 0
    scores_2 = rng.uniform(0, 0.5, n)
    scores_2[-5:] = rng.uniform(0.7, 0.95, 5)
    scores_2[0] = 0.9
    preds_2 = np.ones(n, dtype=int)
    preds_2[-5:] = -1
    preds_2[0] = -1

    # Model 3: agrees on last 4, flags index 10
    scores_3 = rng.uniform(0, 0.5, n)
    scores_3[-4:] = rng.uniform(0.85, 1.0, 4)
    scores_3[10] = 0.85
    preds_3 = np.ones(n, dtype=int)
    preds_3[-4:] = -1
    preds_3[10] = -1

    return {
        "Isolation Forest": {"scores": scores_1, "predictions": preds_1},
        "Local Outlier Factor": {"scores": scores_2, "predictions": preds_2},
        "k-Nearest Neighbors": {"scores": scores_3, "predictions": preds_3},
    }


def test_ensemble_borda_aggregation(mock_detector_results):
    engine = IntelligentEnsembleEngine(aggregation_method="rank_borda", min_consensus_votes=2)
    res = engine.run_ensemble(mock_detector_results)

    scores = np.asarray(res["ensemble_scores"])
    counts = np.asarray(res["consensus_counts"])
    strength = np.asarray(res["consensus_strength"])
    preds = np.asarray(res["consensus_predictions"])

    assert len(scores) == 100
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    # Outliers should have high ensemble scores
    assert np.mean(scores[-5:]) > np.mean(scores[:95])

    # Consensus counts: last 4 samples flagged by all 3 models -> count == 3
    assert np.all(counts[-4:] == 3)
    assert np.all(strength[-4:] == 1.0)

    # Required votes is 2: samples flagged by at least 2 models should be -1
    assert np.all(preds[-5:] == -1)

    # Model disagreement
    disag = res["disagreement"]
    assert "mean_pairwise_jaccard_distance" in disag
    assert "mean_score_rank_correlation" in disag
    assert len(disag["pairwise_jaccard"]) == 3


def test_ensemble_weighted_score(mock_detector_results):
    weights = {
        "Isolation Forest": 2.0,
        "Local Outlier Factor": 1.0,
        "k-Nearest Neighbors": 1.0,
    }
    engine = IntelligentEnsembleEngine(aggregation_method="weighted_score")
    res = engine.run_ensemble(mock_detector_results, weights=weights)

    assert len(res["ensemble_scores"]) == 100
    assert res["model_contributions"]["Isolation Forest"]["weight"] == 0.5

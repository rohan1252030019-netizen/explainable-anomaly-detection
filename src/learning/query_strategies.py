"""
src/learning/query_strategies.py

Active learning query strategies for human-in-the-loop triage.

Selects the most informative unlabelled instances for human analysts to review:
1. Uncertainty Sampling: Prioritizes observations closest to the decision threshold.
2. Disagreement Sampling (Query-by-Committee): Prioritizes observations with high variance
   across detector ensemble scores.
3. Hybrid Sampling: Balances uncertainty, detector divergence, and feature diversity.
"""

from typing import Any

import numpy as np


def query_active_learning_batch(
    norm_scores: np.ndarray,
    threshold: float,
    all_detector_scores: dict[str, np.ndarray] | None = None,
    X: np.ndarray | None = None,
    already_reviewed: set[int] | None = None,
    batch_size: int = 10,
    strategy: str = "hybrid",
) -> list[dict[str, Any]]:
    """
    Select top batch_size candidates for human analyst review.

    Parameters
    ----------
    norm_scores : np.ndarray, shape (N,)
        Primary ensemble or active model normalized scores in [0, 1].
    threshold : float
    all_detector_scores : dict of detector_name -> score_array or None
    X : np.ndarray or None
    already_reviewed : set of int or None
        Indices already annotated by the analyst.
    batch_size : int, default=10
    strategy : str, 'uncertainty', 'disagreement', or 'hybrid'

    Returns
    -------
    list of dicts describing recommended review candidates:
        - index: int
        - score: float
        - uncertainty_rank: float
        - disagreement_score: float
        - priority_score: float
        - query_reason: str
    """
    scores = np.asarray(norm_scores, dtype=np.float64)
    n = len(scores)
    reviewed = set(already_reviewed) if already_reviewed else set()

    unlabelled_indices = [i for i in range(n) if i not in reviewed]
    if not unlabelled_indices:
        return []

    # 1. Uncertainty Score: proximity to threshold (closest to threshold -> highest uncertainty)
    # distance = |S(x) - threshold|
    dist_to_thresh = np.abs(scores - threshold)
    max_dist = max(threshold, 1.0 - threshold, 1e-4)
    uncertainty_score = 1.0 - np.clip(dist_to_thresh / max_dist, 0.0, 1.0)

    # 2. Disagreement Score: standard deviation across detectors if available
    if all_detector_scores and len(all_detector_scores) >= 2:
        matrix = np.vstack([np.asarray(s, dtype=np.float64) for s in all_detector_scores.values()])
        disagreement_score = np.std(matrix, axis=0)
        # Normalize to [0, 1]
        max_dis = np.max(disagreement_score) if np.max(disagreement_score) > 1e-6 else 1.0
        disagreement_norm = disagreement_score / max_dis
    else:
        # Fall back to uncertainty if no multi-detector scores provided
        disagreement_norm = uncertainty_score.copy()

    # 3. Combine according to strategy
    candidates = []
    for idx in unlabelled_indices:
        unc = float(uncertainty_score[idx])
        dis = float(disagreement_norm[idx])
        sc = float(scores[idx])

        if strategy == "uncertainty":
            priority = unc
            reason = f"Decision boundary uncertainty: score {sc:.3f} is near threshold {threshold:.3f}"
        elif strategy == "disagreement":
            priority = dis
            reason = "High detector conflict: ensemble detectors diverge on risk severity"
        else:  # hybrid
            priority = 0.5 * unc + 0.5 * dis
            reason = f"Ambiguous risk profile: border score ({sc:.3f}) with detector disagreement"

        candidates.append({
            "index": idx,
            "score": round(sc, 4),
            "uncertainty_score": round(unc, 4),
            "disagreement_score": round(dis, 4),
            "priority_score": round(priority, 4),
            "query_reason": reason,
        })

    # Sort descending by priority score
    candidates.sort(key=lambda c: c["priority_score"], reverse=True)
    return candidates[:batch_size]

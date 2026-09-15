"""
src/detection/ranking.py

Anomaly ranking utilities.

Assigns a rank to every observation so that the most anomalous observation
receives rank 1 and the least anomalous receives rank N.

Only anomalous observations (prediction == -1) are given a numeric rank in
the primary anomaly ranking.  Normal observations are assigned rank NaN.
An alternative function ranks ALL observations regardless of prediction.

Ranking is based entirely on normalised anomaly scores, which are independent
of ground-truth labels.
"""

from typing import Optional

import numpy as np
import pandas as pd


def rank_anomalies(
    norm_scores: np.ndarray,
    predictions: np.ndarray,
    rank_all: bool = False,
) -> np.ndarray:
    """
    Rank observations by anomaly score.

    Parameters
    ----------
    norm_scores : np.ndarray, shape (n_samples,)
        Normalised scores in [0, 1].  1 = most anomalous.
    predictions : np.ndarray, shape (n_samples,)
        Binary labels: -1 = anomaly, +1 = normal.
    rank_all : bool
        If False (default), only anomalous observations are ranked.
        Normal observations receive NaN.
        If True, all observations are ranked by score.

    Returns
    -------
    ranks : np.ndarray, shape (n_samples,), dtype float64
        Integer ranks (1 = most anomalous) for anomalies.
        NaN for normal observations when rank_all=False.
    """
    norm_scores = np.asarray(norm_scores, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=int)

    n = len(norm_scores)
    ranks = np.full(n, np.nan)

    if rank_all:
        # Rank all observations by descending score
        order = np.argsort(-norm_scores)  # descending
        for rank_pos, idx in enumerate(order, start=1):
            ranks[idx] = rank_pos
    else:
        # Rank only anomalous observations
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_scores = norm_scores[anomaly_mask]

        if len(anomaly_scores) > 0:
            order = np.argsort(-anomaly_scores)  # descending within anomalies
            for rank_pos, rel_idx in enumerate(order, start=1):
                ranks[anomaly_indices[rel_idx]] = rank_pos

    return ranks


def build_results_dataframe(
    original_df: pd.DataFrame,
    raw_scores: np.ndarray,
    norm_scores: np.ndarray,
    predictions: np.ndarray,
    severity: np.ndarray,
    model_name: str,
    feature_names: Optional[list] = None,
) -> pd.DataFrame:
    """
    Assemble a comprehensive results DataFrame.

    Contains:
    - All original columns
    - anomaly_score (raw)
    - normalized_score
    - prediction (+1 / -1)
    - severity
    - rank (1 = most anomalous, NaN for normal)
    - is_anomaly (boolean)
    - model

    Parameters
    ----------
    original_df : pd.DataFrame
        The original (un-preprocessed) DataFrame.
    raw_scores, norm_scores, predictions, severity : np.ndarray
    model_name : str
    feature_names : list or None

    Returns
    -------
    pd.DataFrame
    """
    ranks = rank_anomalies(norm_scores, predictions)

    result_df = original_df.copy().reset_index(drop=True)
    result_df["anomaly_score"] = np.round(raw_scores, 6)
    result_df["normalized_score"] = np.round(norm_scores, 4)
    result_df["prediction"] = predictions
    result_df["severity"] = severity
    result_df["rank"] = ranks
    result_df["is_anomaly"] = predictions == -1
    result_df["model"] = model_name

    # Sort so anomalies appear first, ordered by rank
    anomalies = result_df[result_df["is_anomaly"]].sort_values("rank")
    normals = result_df[~result_df["is_anomaly"]].sort_values(
        "normalized_score", ascending=False
    )
    return pd.concat([anomalies, normals], ignore_index=True)


def top_anomalies(
    results_df: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """
    Return the top-n anomalies sorted by rank.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from build_results_dataframe().
    n : int
        Number of top anomalies to return.

    Returns
    -------
    pd.DataFrame
    """
    anomaly_df = results_df[results_df["is_anomaly"]].copy()
    anomaly_df = anomaly_df.sort_values("rank")
    return anomaly_df.head(n).reset_index(drop=True)

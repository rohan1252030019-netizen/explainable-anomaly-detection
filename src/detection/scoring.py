"""
src/detection/scoring.py

Anomaly score normalisation utilities.

Problem
-------
Different anomaly detection algorithms produce scores on incompatible scales:
- IsolationForest  : typically [-0.5, 0.5]
- LOF              : typically [-inf, -1]
- OneClassSVM      : signed distance, arbitrary scale
- EllipticEnvelope : negative Mahalanobis distance

Comparing raw scores across models is therefore MISLEADING.

Solution
--------
Normalise each model's raw scores to a common [0, 1] scale where:
    0.0 → least anomalous (most normal)
    1.0 → most anomalous

Normalisation strategy (min-max on inverted scores):
    1. All models use the convention: LOWER raw score = MORE anomalous.
    2. Invert: inverted = -raw_score
    3. Min-max normalise to [0, 1].

This preserves the relative ordering while making scores comparable.

Documented clearly so reviewers can verify the methodology.
"""

from typing import Optional

import numpy as np


# ── Core normalisation ─────────────────────────────────────────────────────────

def normalize_scores(
    raw_scores: np.ndarray,
    method: str = "minmax",
    clip: bool = True,
) -> np.ndarray:
    """
    Normalise raw anomaly scores to [0, 1].

    Convention: input scores follow "lower = more anomalous" (sklearn default).
    Output scores follow "higher = more anomalous".

    Parameters
    ----------
    raw_scores : np.ndarray, shape (n_samples,)
        Raw scores from any of the supported detectors.
    method : {"minmax"}
        Normalisation method.  Only min-max is currently supported.
    clip : bool
        If True, clip the output to [0, 1] to guard against numerical drift.

    Returns
    -------
    norm_scores : np.ndarray, shape (n_samples,), dtype float64
        Values in [0, 1] where 1 = most anomalous.
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)

    if raw_scores.ndim != 1:
        raise ValueError(f"raw_scores must be 1-D, got shape {raw_scores.shape}.")

    if len(raw_scores) == 0:
        return raw_scores.copy()

    # Replace any NaN/inf
    raw_scores = _sanitize(raw_scores)

    # Step 1: Invert so that higher = more anomalous
    inverted = -raw_scores

    # Step 2: Min-max normalise to [0, 1]
    s_min = inverted.min()
    s_max = inverted.max()

    if np.isclose(s_min, s_max):
        # All scores are identical — no discrimination possible
        norm = np.full_like(inverted, 0.5, dtype=np.float64)
    else:
        norm = (inverted - s_min) / (s_max - s_min)

    if clip:
        norm = np.clip(norm, 0.0, 1.0)

    return norm


def _sanitize(scores: np.ndarray) -> np.ndarray:
    """Replace NaN and ±inf with finite extremes."""
    scores = scores.copy()
    finite_mask = np.isfinite(scores)
    if not np.all(finite_mask):
        finite_vals = scores[finite_mask]
        if len(finite_vals) == 0:
            return np.zeros_like(scores)
        s_min = finite_vals.min()
        s_max = finite_vals.max()
        scores[np.isposinf(scores)] = s_max
        scores[np.isneginf(scores)] = s_min
        scores[np.isnan(scores)] = s_min
    return scores


# ── Ensemble scoring ───────────────────────────────────────────────────────────

def ensemble_scores(
    score_dict: dict,
    weights: Optional[dict] = None,
) -> np.ndarray:
    """
    Combine normalised scores from multiple models into an ensemble score.

    Each model's normalised scores [0, 1] are weighted and averaged.
    Individual scores must be normalised BEFORE calling this function.

    Parameters
    ----------
    score_dict : dict
        {model_name: normalised_score_array (np.ndarray)}
    weights : dict or None
        {model_name: weight (float)}
        If None, equal weights are used.
        Weights are automatically normalised to sum to 1.0.

    Returns
    -------
    ensemble : np.ndarray, shape (n_samples,)
        Weighted average of individual normalised scores.
    """
    if not score_dict:
        raise ValueError("score_dict is empty.")

    names = list(score_dict.keys())
    arrays = [score_dict[n] for n in names]

    # Validate lengths
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Score arrays have different lengths: "
            f"{dict(zip(names, lengths))}"
        )

    # Build weight vector
    if weights is None:
        w = np.ones(len(names)) / len(names)
    else:
        w = np.array([weights.get(n, 1.0) for n in names], dtype=np.float64)
        w_sum = w.sum()
        if np.isclose(w_sum, 0.0):
            w = np.ones(len(names)) / len(names)
        else:
            w /= w_sum  # normalise to sum to 1

    # Weighted average
    stacked = np.vstack(arrays).T   # shape: (n_samples, n_models)
    ensemble = (stacked * w).sum(axis=1)
    return np.clip(ensemble, 0.0, 1.0)

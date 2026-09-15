"""
src/detection/thresholding.py

Anomaly thresholding strategies.

The threshold converts continuous normalised anomaly scores (0–1) into a
binary anomaly decision: normal (+1) or anomaly (-1).

Supported methods
-----------------
contamination   : Flag the top-k% of observations, where k = contamination.
quantile        : Flag observations above a specified percentile (e.g. 95th).
fixed           : Flag observations above a user-supplied fixed score.

IMPORTANT:
- The threshold is NOT learnt from labels.  It is a post-hoc decision rule
  applied to normalised scores.
- Changing the threshold changes the number of detected anomalies.
  Clearly communicate this sensitivity to users.
- Different thresholds suit different operational contexts.
"""

from typing import Union

import numpy as np


# ── Severity thresholds (configurable) ────────────────────────────────────────

# Default severity bands (on normalised score 0–1)
SEVERITY_THRESHOLDS = {
    "LOW":      (0.00, 0.50),
    "MEDIUM":   (0.50, 0.75),
    "HIGH":     (0.75, 0.90),
    "CRITICAL": (0.90, 1.01),   # 1.01 so exactly 1.0 is CRITICAL
}

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ── Core thresholding ──────────────────────────────────────────────────────────

def apply_threshold(
    norm_scores: np.ndarray,
    method: str = "contamination",
    contamination: float = 0.05,
    quantile: float = 0.95,
    fixed_threshold: float = 0.5,
) -> np.ndarray:
    """
    Convert normalised anomaly scores into binary labels.

    Parameters
    ----------
    norm_scores : np.ndarray, shape (n_samples,)
        Normalised scores in [0, 1].  1 = most anomalous.
    method : {"contamination", "quantile", "fixed"}
        Thresholding strategy.
    contamination : float
        Used when method="contamination".
        Fraction of observations to flag as anomalous.
        Range: (0, 0.5].
    quantile : float
        Used when method="quantile".
        Percentile threshold.  E.g. 0.95 = flag top 5%.
    fixed_threshold : float
        Used when method="fixed".
        Observations with normalised score >= fixed_threshold are anomalies.

    Returns
    -------
    predictions : np.ndarray, shape (n_samples,)
        +1 = normal, -1 = anomaly  (sklearn convention)
    """
    norm_scores = np.asarray(norm_scores, dtype=np.float64)
    _validate_norm_scores(norm_scores)

    if method == "contamination":
        threshold = _contamination_threshold(norm_scores, contamination)
    elif method == "quantile":
        threshold = float(np.percentile(norm_scores, quantile * 100))
    elif method == "fixed":
        threshold = float(np.clip(fixed_threshold, 0.0, 1.0))
    else:
        raise ValueError(
            f"Unknown threshold method '{method}'. "
            "Choose from: 'contamination', 'quantile', 'fixed'."
        )

    predictions = np.where(norm_scores >= threshold, -1, 1)
    return predictions


def _contamination_threshold(
    norm_scores: np.ndarray,
    contamination: float,
) -> float:
    """Compute score threshold corresponding to a contamination fraction."""
    contamination = float(np.clip(contamination, 1e-6, 0.5))
    return float(np.percentile(norm_scores, (1.0 - contamination) * 100))


def get_threshold_value(
    norm_scores: np.ndarray,
    method: str = "contamination",
    contamination: float = 0.05,
    quantile: float = 0.95,
    fixed_threshold: float = 0.5,
) -> float:
    """
    Return the numeric threshold value without applying it.

    Useful for displaying the threshold line on score distribution plots.
    """
    norm_scores = np.asarray(norm_scores, dtype=np.float64)
    _validate_norm_scores(norm_scores)

    if method == "contamination":
        return _contamination_threshold(norm_scores, contamination)
    elif method == "quantile":
        return float(np.percentile(norm_scores, quantile * 100))
    elif method == "fixed":
        return float(np.clip(fixed_threshold, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown threshold method '{method}'.")


# ── Severity classification ────────────────────────────────────────────────────

def assign_severity(
    norm_scores: np.ndarray,
    thresholds: dict = None,
) -> np.ndarray:
    """
    Assign a severity label to each observation based on its normalised score.

    Parameters
    ----------
    norm_scores : np.ndarray, shape (n_samples,)
    thresholds : dict or None
        Custom severity bands.  Keys: "LOW", "MEDIUM", "HIGH", "CRITICAL".
        Values: (lower_bound_inclusive, upper_bound_exclusive) tuples.
        Defaults to SEVERITY_THRESHOLDS.

    Returns
    -------
    severity : np.ndarray of str, shape (n_samples,)
        One of: "LOW", "MEDIUM", "HIGH", "CRITICAL"

    Notes
    -----
    These severity bands are configurable demonstration thresholds.
    They are NOT universally correct — they depend on the domain and
    operational requirements.
    """
    norm_scores = np.asarray(norm_scores, dtype=np.float64)
    bands = thresholds or SEVERITY_THRESHOLDS

    severity = np.empty(len(norm_scores), dtype=object)
    for label, (lo, hi) in bands.items():
        mask = (norm_scores >= lo) & (norm_scores < hi)
        severity[mask] = label

    # Any score == 1.0 should be CRITICAL
    severity[norm_scores >= 1.0] = "CRITICAL"
    # Default fallback
    severity[severity == None] = "LOW"  # noqa: E711

    return severity.astype(str)


# ── Validation helper ──────────────────────────────────────────────────────────

def _validate_norm_scores(scores: np.ndarray) -> None:
    if scores.ndim != 1:
        raise ValueError(f"norm_scores must be 1-D, got shape {scores.shape}.")
    if len(scores) == 0:
        raise ValueError("norm_scores is empty.")

"""
src/data/labels.py

Single source of truth for turning a label column into a binary anomaly mask.

Why this module exists
----------------------
Ground-truth labels arrive in two equally common conventions:

``{0, 1}``     0 = normal, 1 = anomaly   (the bundled benchmark)
``{-1, +1}``  +1 = normal, -1 = anomaly  (scikit-learn's ``predict`` output)

Two call sites used to hard-code ``labels == 1``.  On a ``{-1, +1}`` column that
expression selects the *normal* rows, so every downstream metric (precision,
recall, ROC-AUC, the confidence note in the UI) was computed against an inverted
target while looking perfectly plausible.  The rule now lives in one place and
every caller goes through it.

Return ``None`` instead of guessing when the label column cannot be interpreted
unambiguously.  A silently wrong target is worse than a missing one: the holdout
report already knows how to say "labels unavailable".

Documented semantics
--------------------
- ``{0, 1}``        -> 1 marks the anomaly.
- ``{-1, 1}``       -> -1 marks the anomaly.
- ``{False, True}`` -> True marks the anomaly.
- anything else     -> ``None`` (not interpretable; callers must report it).
"""

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["to_binary_labels", "anomaly_mask_from_labels"]


def to_binary_labels(labels: Any) -> np.ndarray | None:
    """
    Map a label vector to ``{0 = normal, 1 = anomaly}``.

    Parameters
    ----------
    labels : array-like, pandas.Series or pandas.DataFrame (single column)

    Returns
    -------
    np.ndarray of int, shape (n_samples,) or None
        ``None`` when the values do not follow a supported convention.
    """
    if labels is None:
        return None

    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] != 1:
            return None
        arr = labels.iloc[:, 0].to_numpy()
    elif isinstance(labels, pd.Series):
        arr = labels.to_numpy()
    else:
        arr = np.asarray(labels)

    if arr.ndim != 1:
        arr = arr.ravel()

    if arr.size == 0:
        return None

    # Nullable / object columns: drop nothing, but bail out on missing values.
    if pd.isna(arr).any():
        return None

    uniq = set(np.unique(arr).tolist())

    if uniq <= {0, 1}:
        return arr.astype(int)
    if uniq <= {-1, 1}:
        return (arr == -1).astype(int)
    if uniq <= {False, True}:
        return arr.astype(int)

    return None


def anomaly_mask_from_labels(labels: Any) -> np.ndarray | None:
    """Boolean mask where ``True`` marks a labelled anomaly."""
    binary = to_binary_labels(labels)
    return None if binary is None else binary.astype(bool)

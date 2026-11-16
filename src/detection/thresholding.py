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

from typing import Any

import numpy as np

# ── Severity thresholds (configurable) ────────────────────────────────────────

# Fallback severity bands on absolute normalised score.
#
# These edges are the FALLBACK only.  assign_severity() prefers the
# distribution-aware bands returned by derive_severity_bands().
#
# Why fixed edges are not enough
# ------------------------------
# Normalised anomaly scores are heavy tailed: the bulk of the population sits
# in a narrow low band and a thin tail reaches 1.0.  On the bundled 880-row
# benchmark the median normalised score is ~0.16 and the 95th percentile is
# ~0.53, so the fixed edge at 0.50 pushes almost every flagged anomaly into
# "LOW" and leaves MEDIUM/HIGH/CRITICAL nearly empty.  The severity breakdown
# is a headline number on the dashboard, so it has to be calibrated to the
# score distribution that was actually produced.
SEVERITY_THRESHOLDS = {
    "LOW":      (0.00, 0.50),
    "MEDIUM":   (0.50, 0.75),
    "HIGH":     (0.75, 0.90),
    "CRITICAL": (0.90, 1.01),   # 1.01 so exactly 1.0 is CRITICAL
}

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Upper edges of each band expressed as quantile levels of the score
# distribution.  Used by derive_severity_bands().
DEFAULT_SEVERITY_QUANTILES = {
    "MEDIUM": 0.50,
    "HIGH": 0.80,
    "CRITICAL": 0.95,
}


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
#
# Severity answers "of the rows we flagged, how urgently does each need
# review?"  It is deliberately a RANK question, not an absolute score
# question, because absolute score semantics are not comparable across
# datasets or models.
#
# Why not fixed score bands?
# ----------------------------
# Normalised anomaly scores are heavy tailed.  On the bundled 880-row
# benchmark the median normalised score is ~0.16 and the 95th percentile is
# ~0.53, so a fixed edge at 0.50 classifies almost every flagged anomaly as
# "LOW" and leaves MEDIUM / HIGH / CRITICAL nearly empty.  The severity
# breakdown is a headline number on the dashboard, so it has to be calibrated
# to the distribution that was actually produced.
#
# Why rank-based and not quantile-based?
# -------------------------------------
# Quantile-based bands would still be degenerate: robust normalisation clips
# the extreme tail to exactly 1.0, producing many tied scores, and a handful
# of ties near the top collapse whole bands to zero width.  Ranking first and
# tiering on the rank is immune to ties, so the four bands always receive a
# meaningful share of the flagged rows.


def severity_rank_levels(
    quantiles: dict[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Express each severity band as a fraction of the flagged population.

    Returns
    -------
    dict mapping band name -> (lower_fraction, upper_fraction), where the
    fractions are cumulative shares of the flagged rows ordered most-anomalous
    first.  e.g. {"CRITICAL": (0.95, 1.0), "HIGH": (0.80, 0.95), ...}
    """
    levels = dict(quantiles or DEFAULT_SEVERITY_QUANTILES)
    ordered = sorted(levels.items(), key=lambda kv: kv[1])
    if len(ordered) != len(SEVERITY_ORDER) - 1:
        # Non-standard quantile set: fall back to an even split.
        n = len(SEVERITY_ORDER)
        step = 1.0 / n
        return {
            name: (i * step, (i + 1) * step)
            for i, name in enumerate(SEVERITY_ORDER)
        }

    edges = [0.0] + [float(level) for _, level in ordered] + [1.0]
    return {
        name: (edges[i], edges[i + 1])
        for i, name in enumerate(SEVERITY_ORDER)
    }


def assign_severity(
    norm_scores: np.ndarray,
    thresholds: dict[str, tuple[float, float]] | None = None,
    adaptive: bool = True,
    anomaly_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Assign a severity label to each observation based on its normalised score.

    Parameters
    ----------
    norm_scores : np.ndarray, shape (n_samples,)
        Normalised scores in [0, 1].  1 = most anomalous.
    thresholds : dict or None
        Absolute score bands: {"LOW": (lo, hi), ...}.  When supplied, this
        switches to fixed absolute banding and ``adaptive`` / ``anomaly_mask``
        are ignored.  Use this when a domain has real severity thresholds.
    adaptive : bool
        Default True.  Tier the flagged rows by rank (see module notes).
    anomaly_mask : np.ndarray or None
        Boolean mask selecting the flagged rows, normally ``predictions == -1``.
        Rows outside the mask are labelled "LOW": they sit below the detection
        threshold and are not on the triage queue.

    Returns
    -------
    severity : np.ndarray of str, shape (n_samples,)
        One of: "LOW", "MEDIUM", "HIGH", "CRITICAL"

    Notes
    -----
    Adaptive severity describes standing *within this run*.  It is stable
    under rescaling and under tied scores, but it is not an absolute severity
    and must not be compared across different datasets without recalibration.

    These bands are a demonstration tiering and are NOT domain ground truth —
    the right cut points depend on operational cost, not on statistics.
    """
    scores = np.asarray(norm_scores, dtype=np.float64).ravel()

    # ── Absolute banding mode ──────────────────────────────────────────────
    if thresholds is not None:
        severity = np.full(len(scores), "LOW", dtype=object)
        for label in SEVERITY_ORDER:
            if label not in thresholds:
                continue
            lo, hi = thresholds[label]
            severity[(scores >= lo) & (scores < hi)] = label
        # A score of exactly 1.0 belongs to the top band.
        severity[scores >= 1.0] = SEVERITY_ORDER[-1]
        return severity.astype(str)

    if not adaptive:
        return assign_severity(
            scores, thresholds=SEVERITY_THRESHOLDS, adaptive=False
        )

    # ── Rank-based tiering ─────────────────────────────────────────────────
    severity = np.full(len(scores), "LOW", dtype=object)

    if anomaly_mask is None:
        flagged = np.ones(len(scores), dtype=bool)
    else:
        flagged = np.asarray(anomaly_mask).ravel().astype(bool)
        if flagged.shape[0] != scores.shape[0]:
            raise ValueError(
                f"anomaly_mask has length {flagged.shape[0]} but norm_scores "
                f"has {scores.shape[0]}."
            )

    flagged_idx = np.flatnonzero(flagged)
    n_flagged = flagged_idx.size
    if n_flagged == 0:
        return severity.astype(str)

    if n_flagged < len(SEVERITY_ORDER):
        # Too few flagged rows for rank tiering to mean anything: a single
        # flagged row would come out "LOW" purely by arithmetic.  Fall back to
        # absolute bands so the output stays interpretable.
        return assign_severity(
            scores, thresholds=SEVERITY_THRESHOLDS, adaptive=False
        )

    # Rank the flagged rows by ascending score, so pct runs from 0 for the
    # LEAST anomalous flagged row up to ~1 for the MOST anomalous one — the
    # direction that puts the top-ranked anomaly in CRITICAL.
    #
    # Position in a stable sort is used rather than comparing score values, so
    # tied scores still receive distinct, evenly spread ranks.
    local_scores = scores[flagged_idx]
    order = np.argsort(local_scores, kind="stable")
    pct = np.empty(n_flagged, dtype=np.float64)
    pct[order] = np.arange(n_flagged, dtype=np.float64) / n_flagged

    levels = severity_rank_levels()
    top = SEVERITY_ORDER[-1]
    for label in SEVERITY_ORDER:
        lo, hi = levels[label]
        if label == top:
            local_mask = pct >= lo
        else:
            local_mask = (pct >= lo) & (pct < hi)
        severity[flagged_idx[local_mask]] = label

    return severity.astype(str)


def severity_legend(anomaly_mask_counts: dict[str, int]) -> list[dict[str, Any]]:
    """
    Build a legend for the severity donut so the UI can state what each band
    actually means instead of implying an absolute threshold.

    Parameters
    ----------
    anomaly_mask_counts : dict
        {severity_name: number_of_flagged_rows_in_that_band}

    Returns
    -------
    list of dicts, most severe first, each with name, count, share and a
    plain-English definition.
    """
    total = sum(v for v in anomaly_mask_counts.values() if v)
    levels = severity_rank_levels()

    definitions = {
        "CRITICAL": "Top 5% of flagged rows — review first",
        "HIGH": "Next 15% of flagged rows — high priority",
        "MEDIUM": "Next 30% of flagged rows — routine review",
        "LOW": "Remaining 50% of flagged rows — batch triage",
    }

    out = []
    for name in reversed(SEVERITY_ORDER):
        count = int(anomaly_mask_counts.get(name, 0))
        lo, hi = levels[name]
        out.append({
            "severity": name,
            "count": count,
            "share": round(count / total, 4) if total else 0.0,
            "rank_range": [round(lo, 4), round(hi, 4)],
            "definition": definitions[name],
        })
    return out


# ── Validation helper ──────────────────────────────────────────────────────────

def _validate_norm_scores(scores: np.ndarray) -> None:
    if scores.ndim != 1:
        raise ValueError(f"norm_scores must be 1-D, got shape {scores.shape}.")
    if len(scores) == 0:
        raise ValueError("norm_scores is empty.")

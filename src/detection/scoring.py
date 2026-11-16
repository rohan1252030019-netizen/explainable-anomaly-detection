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
    0.0 -> least anomalous (most normal)
    1.0 -> most anomalous

Why naive min-max is NOT good enough
------------------------------------
A plain min-max scale is defined by a SINGLE observation: the most extreme
point in the batch.  Anomaly scores are, by construction, heavy tailed, so one
outlier compresses the entire remaining population into a tiny sliver of the
[0, 1] range.  Measured on the bundled 880-row benchmark this produced:

    median = 0.156,  p95 = 0.527,  only 18 / 880 rows above 0.90

That distortion is not cosmetic.  It moves every fixed severity band
(see thresholding.SEVERITY_THRESHOLDS) to the wrong place, so almost every
detected anomaly gets labelled "LOW".

Normalisation strategies
------------------------
method="quantile"  (default)
    Robust linear rescale anchored on the [low_q, high_q] quantile interval
    (default 1% .. 99%).  Values outside the interval are clipped to 0 / 1.
    Preserves the linear "distance" reading of a score while ignoring the
    influence of a handful of extreme points.

method="rank"
    Empirical CDF: replaces each score by its quantile rank in the batch.
    The output is exactly uniform on [0, 1] by construction, which makes it
    the most stable choice for ranking or ensembling across models.  It is
    NOT linear in raw-score space, so the value loses its "how far from
    normal" interpretation.

method="minmax"
    Classic (x - min) / (max - min).  Kept for reproducibility of older runs
    and for tests; known to be outlier-sensitive, see above.

All methods preserve the ordering of the input scores, so thresholds and
ranks derived downstream remain monotone in the raw score.

IMPORTANT
---------
Quantiles are estimated from the batch being scored.  For a production
service the same low_q / high_q should be fitted once on a reference sample
and reused, otherwise the meaning of "0.8" drifts between batches.  Use
ScoreNormalizer.fit / .transform for that.
"""


from typing import Any

import numpy as np

DEFAULT_LOW_QUANTILE = 0.01
DEFAULT_HIGH_QUANTILE = 0.99

VALID_METHODS = ("quantile", "rank", "minmax")


# ── Fitted normaliser (reusable across batches) ─────────────────────────────────

class ScoreNormalizer:
    """
    Stateful score normaliser.

    Fit once on a reference batch, then call :meth:`transform` on any number of
    later batches.  This keeps the meaning of a normalised score stable over
    time, which a per-batch normalise_scores() call cannot guarantee.

    Examples
    --------
    >>> norm = ScoreNormalizer(method="quantile").fit(scores_batch_1)
    >>> later = norm.transform(scores_batch_2)
    """

    def __init__(
        self,
        method: str = "quantile",
        low_quantile: float = DEFAULT_LOW_QUANTILE,
        high_quantile: float = DEFAULT_HIGH_QUANTILE,
    ):
        if method not in VALID_METHODS:
            raise ValueError(
                f"Unknown normalisation method '{method}'. "
                f"Choose from: {list(VALID_METHODS)}."
            )
        if not 0.0 <= low_quantile < high_quantile <= 1.0:
            raise ValueError(
                f"Require 0 <= low_quantile < high_quantile <= 1, got "
                f"({low_quantile}, {high_quantile})."
            )
        self.method = method
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self._offset: float = 0.0
        self._scale: float = 1.0
        self._reference: np.ndarray | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, raw_scores: np.ndarray) -> "ScoreNormalizer":
        """Estimate the normalisation anchors from a reference batch."""
        clean = _prepare(raw_scores)

        if clean.size == 0:
            # Nothing to anchor on. Keep the empty input empty rather than
            # letting np.quantile raise deep inside the normaliser.
            self._offset = 0.0
            self._scale = 1.0
            self._reference = clean
            self._is_fitted = True
            return self

        if self.method == "rank":
            self._reference = clean
        elif self.method == "quantile":
            lo = float(np.quantile(clean, self.low_quantile))
            hi = float(np.quantile(clean, self.high_quantile))
            # Degenerate spread (e.g. near-constant scores): fall back to
            # full-range quantiles so we never divide by ~0.
            if np.isclose(lo, hi):
                lo = float(clean.min())
                hi = float(clean.max())
            self._offset = lo
            self._scale = (hi - lo) or 1.0
        else:  # minmax
            self._offset = float(clean.min())
            self._scale = (float(clean.max()) - float(clean.min())) or 1.0

        self._is_fitted = True
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        """Normalise a batch using the anchors learned by :meth:`fit`."""
        if not self._is_fitted:
            raise RuntimeError("ScoreNormalizer.fit() must be called first.")

        clean = _prepare(raw_scores)

        if self.method == "rank":
            assert self._reference is not None  # for type checkers
            return np.searchsorted(
                np.sort(self._reference), clean, side="right"
            ) / max(len(self._reference), 1)

        return np.clip((clean - self._offset) / self._scale, 0.0, 1.0)

    def fit_transform(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.fit(raw_scores).transform(raw_scores)

    def describe(self) -> dict[str, Any]:
        """Return the fitted anchors so a UI can display the mapping."""
        return {
            "method": self.method,
            "offset": self._offset,
            "scale": self._scale,
            "low_quantile": self.low_quantile,
            "high_quantile": self.high_quantile,
        }


# ── One-shot normalisation ─────────────────────────────────────────────────────

def normalize_scores(
    raw_scores: np.ndarray,
    method: str = "quantile",
    clip: bool = True,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
    high_quantile: float = DEFAULT_HIGH_QUANTILE,
) -> np.ndarray:
    """
    Normalise raw anomaly scores to [0, 1] in a single call.

    Convention: input scores follow "lower = more anomalous" (sklearn default).
    Output scores follow "higher = more anomalous".

    Parameters
    ----------
    raw_scores : np.ndarray, shape (n_samples,)
        Raw scores from any of the supported detectors.
    method : {"quantile", "rank", "minmax"}
        Normalisation strategy.  See the module docstring for trade-offs.
    clip : bool
        If True, clip the output to [0, 1].  Only meaningful for the linear
        methods; "rank" is already bounded.
    low_quantile, high_quantile : float
        Anchor interval for method="quantile".

    Returns
    -------
    norm_scores : np.ndarray, shape (n_samples,), dtype float64
        Values in [0, 1] where 1 = most anomalous.
    """
    normalizer = ScoreNormalizer(
        method=method,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    norm = normalizer.fit_transform(raw_scores)
    return norm if clip else np.asarray(norm, dtype=np.float64)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _prepare(raw_scores: np.ndarray) -> np.ndarray:
    """
    Validate, coerce to 1-D float64, and flip to "higher = more anomalous".

    Every detector in this framework follows the sklearn convention where a
    LOWER raw score means MORE anomalous.  Negating once here means every
    method below can be written as a plain ascending rescale, and forgetting
    the flip in one branch would silently reverse the whole ranking.
    """
    scores = np.asarray(raw_scores, dtype=np.float64)

    if scores.ndim != 1:
        raise ValueError(f"raw_scores must be 1-D, got shape {scores.shape}.")

    if len(scores) == 0:
        return scores.copy()

    return -_sanitize(scores)


def _sanitize(scores: np.ndarray) -> np.ndarray:
    """Replace NaN and +-inf with finite extremes."""
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
    weights: dict | None = None,
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
    arrays = [np.asarray(score_dict[n], dtype=np.float64) for n in names]

    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Score arrays have different lengths: "
            f"{dict(zip(names, lengths, strict=False))}"
        )

    if weights is None:
        w = np.ones(len(names)) / len(names)
    else:
        w = np.array([weights.get(n, 1.0) for n in names], dtype=np.float64)
        w_sum = w.sum()
        if np.isclose(w_sum, 0.0):
            w = np.ones(len(names)) / len(names)
        else:
            w /= w_sum  # normalise to sum to 1

    stacked = np.vstack(arrays).T   # shape: (n_samples, n_models)
    ensemble = (stacked * w).sum(axis=1)
    return np.clip(ensemble, 0.0, 1.0)

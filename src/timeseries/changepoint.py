"""
src/timeseries/changepoint.py

Change-Point Detection for Time-Series Anomaly Intelligence.

Provides:
1. PELT (Pruned Exact Linear Time): Optimal offline segmentation using pruned dynamic
   programming to find exact structural changepoints in linear time.
2. BOCPD (Bayesian Online Change-Point Detection): Online recursive run-length
   inference to identify regime shifts in streaming data.
"""

from typing import Any

import numpy as np

# ── Cost function for Gaussian mean/variance shift ─────────────────────────────

def _segment_cost_gaussian(y_seg: np.ndarray) -> float:
    """Least-squares cost for segment mean shift (sum of squared deviations)."""
    m = len(y_seg)
    if m <= 1:
        return 0.0
    return float(np.sum((y_seg - np.mean(y_seg)) ** 2))


def detect_changepoints_pelt(
    series: np.ndarray,
    penalty: float | None = None,
    min_size: int = 5,
) -> dict[str, Any]:
    """
    Pruned Exact Linear Time (PELT) change-point detection.

    Minimizes: sum_{i=1}^m C(y_{tau_{i-1}:tau_i}) + beta * m

    Parameters
    ----------
    series : np.ndarray, shape (N,)
    penalty : float or None
        Complexity penalty beta. If None, defaults to 2 * var(y) * log(N).
    min_size : int, default=5
        Minimum segment length between change-points.

    Returns
    -------
    dict containing:
        - changepoints: list of int indices where regime shifts occurred
        - n_changepoints: int
        - segments: list of dicts with segment statistics (start, end, mean, std)
    """
    y = np.asarray(series, dtype=np.float64).ravel()
    n = len(y)
    if n < min_size * 2:
        return {"changepoints": [], "n_changepoints": 0, "segments": [{"start": 0, "end": n, "mean": float(np.mean(y)), "std": float(np.std(y))}]}

    # Estimate within-segment noise variance using first differences
    diffs = np.diff(y)
    sigma_est = float(np.median(np.abs(diffs - np.median(diffs)))) / 1.4826
    noise_var = max(sigma_est ** 2, 0.1)
    beta = float(3.0 * noise_var * np.log(n)) if penalty is None else float(penalty)
    K = 0.0

    # F[t] = optimal cost of segmenting y[0:t]
    F = np.zeros(n + 1, dtype=np.float64)
    F[0] = -beta

    # Backpointer for optimal previous changepoint
    cp_pointers: dict[int, int] = {0: 0}

    # Candidate changepoint set R
    R = [0]

    for t in range(min_size, n + 1):
        candidates = [tau for tau in R if (t - tau) >= min_size]
        untested = [tau for tau in R if (t - tau) < min_size]
        if not candidates:
            candidates = [0]

        costs = []
        for tau in candidates:
            cost_seg = _segment_cost_gaussian(y[tau:t])
            costs.append(F[tau] + cost_seg + beta)

        best_idx = int(np.argmin(costs))
        best_tau = candidates[best_idx]
        best_val = costs[best_idx]

        F[t] = best_val
        cp_pointers[t] = best_tau

        # Prune candidates tau where F(tau) + C(y_{tau:t}) + K > F(t)
        survivors = [
            tau for tau in candidates
            if (F[tau] + _segment_cost_gaussian(y[tau:t]) + K) <= F[t]
        ]
        R = survivors + untested + [t]

    # Trace back changepoints
    changepoints = []
    curr = n
    while curr > 0:
        prev = cp_pointers.get(curr, 0)
        if prev > 0:
            changepoints.append(prev)
        curr = prev
    changepoints.reverse()

    # Build segment statistics
    segments = []
    bounds = [0] + changepoints + [n]
    for i in range(len(bounds) - 1):
        s_start, s_end = bounds[i], bounds[i + 1]
        s_data = y[s_start:s_end]
        segments.append({
            "start": s_start,
            "end": s_end,
            "length": s_end - s_start,
            "mean": round(float(np.mean(s_data)), 3),
            "std": round(float(np.std(s_data)), 3),
        })

    return {
        "changepoints": changepoints,
        "n_changepoints": len(changepoints),
        "segments": segments,
        "penalty_beta": beta,
        "total_cost": round(float(F[n]), 4),
    }


def detect_changepoints_bocpd(
    series: np.ndarray,
    hazard_rate: float = 0.02,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Bayesian Online Change-Point Detection (Adams & MacKay 2007).

    Sequentially updates run-length posterior distribution P(r_t | x_{1:t})
    with a constant hazard rate H = 1 / lambda.

    Parameters
    ----------
    series : np.ndarray, shape (N,)
    hazard_rate : float, default=0.02 (expected segment length 50)
    threshold : float, default=0.5 (cutoff on P(changepoint at t))

    Returns
    -------
    dict containing:
        - changepoint_probabilities: list of float
        - flagged_changepoints: list of int
        - max_prob: float
    """
    y = np.asarray(series, dtype=np.float64).ravel()
    n = len(y)
    if n == 0:
        return {"changepoint_probabilities": [], "flagged_changepoints": [], "max_prob": 0.0}

    # Recursive online run-length probability tracker
    # R[r] is the probability of current run length being r
    R = np.array([1.0], dtype=np.float64)
    cp_probs = np.zeros(n, dtype=np.float64)

    prior_mean = float(np.mean(y))
    prior_std = max(float(np.std(y)), 1.0)

    # Online running Gaussian sufficient statistics (mean & variance)
    running_mean = y[0]
    running_var = 1.0

    for t in range(n):
        x = y[t]

        # 1. Predictive probability under current segment model
        std = max(np.sqrt(running_var), 0.5)
        pred_prob = np.exp(-0.5 * ((x - running_mean) / std) ** 2) / (np.sqrt(2 * np.pi) * std)
        pred_prob = max(float(pred_prob), 1e-12)

        # Prior probability for reset r=0
        prior_prob = np.exp(-0.5 * ((x - prior_mean) / prior_std) ** 2) / (np.sqrt(2 * np.pi) * prior_std)
        prior_prob = max(float(prior_prob), 1e-6)

        # 2. Growth probabilities: P(r_t = r_{t-1} + 1)
        growth_probs = R * pred_prob * (1.0 - hazard_rate)

        # 3. Changepoint probability: P(r_t = 0) evaluated under unconditioned prior
        cp_prob = float(np.sum(R)) * prior_prob * hazard_rate

        # Update run length distribution for next step
        new_R = np.zeros(len(R) + 1, dtype=np.float64)
        new_R[0] = cp_prob
        new_R[1:] = growth_probs
        # Normalize
        total = np.sum(new_R)
        if total > 0:
            new_R /= total

        # Bound R size to keep online tracking O(1) in memory
        R = new_R[:150]
        R /= np.sum(R)
        cp_probs[t] = float(R[0])

        # If changepoint occurred, reset running sufficient stats to new observation
        if R[0] > 0.5:
            running_mean = x
            running_var = 1.0
        else:
            # Update running stats with EMA
            running_mean = 0.90 * running_mean + 0.10 * x
            running_var = 0.90 * running_var + 0.10 * ((x - running_mean) ** 2)

    flagged = np.where(cp_probs >= threshold)[0].tolist()

    return {
        "changepoint_probabilities": [round(float(p), 4) for p in cp_probs],
        "flagged_changepoints": flagged,
        "n_changepoints": len(flagged),
        "max_prob": round(float(np.max(cp_probs)), 4) if len(cp_probs) > 0 else 0.0,
    }

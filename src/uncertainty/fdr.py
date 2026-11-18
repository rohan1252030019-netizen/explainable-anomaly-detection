"""
src/uncertainty/fdr.py

False Discovery Rate (FDR) Control via Benjamini-Hochberg Procedure for SENTINEL.

Mathematical Foundation:
-------------------------
When an anomaly detection pipeline scores thousands of streaming or batch records,
evaluating each record at individual significance level alpha (e.g., alpha = 0.05)
leads to severe multiple-testing inflation:
    Expected False Alarms = m * alpha  (e.g., 50 false alarms for 1,000 nominal records).

The Benjamini-Hochberg (BH) procedure controls the False Discovery Rate (FDR):
    FDR = E[ V / max(R, 1) ] <= q*
where:
    V = Number of false positives (nominal records erroneously flagged).
    R = Total number of flagged anomalies (discoveries).
    q* = Target maximum acceptable false discovery rate (e.g., 0.05 or 0.10).

Procedure:
1. Sort m conformal p-values: p_(1) <= p_(2) <= ... <= p_(m).
2. Find k = max { i in 1..m : p_(i) <= (i / m) * q* }.
3. If such k exists, reject null for all records with p <= p_(k).
4. Compute monotonic adjusted p-values (q-values):
       q_(i) = min_{j >= i} { (m / j) * p_(j) }, clipped to 1.0.
"""

from typing import Any

import numpy as np


def benjamini_hochberg_fdr(
    p_values: np.ndarray,
    target_fdr: float = 0.05,
) -> dict[str, Any]:
    """
    Perform Benjamini-Hochberg procedure for multiple-testing FDR control.

    Parameters
    ----------
    p_values : np.ndarray of shape (m,)
        Raw conformal or statistical p-values.
    target_fdr : float, default=0.05
        Maximum acceptable false discovery rate in (0, 1).

    Returns
    -------
    dict containing:
        - 'q_values': np.ndarray of adjusted BH p-values matching input order
        - 'is_discovery': np.ndarray of boolean flags (True = rejected null / significant anomaly)
        - 'n_discoveries': int total discoveries
        - 'critical_p_cutoff': float maximum p-value rejected by BH
        - 'target_fdr': float
        - 'guarantee_statement': str
    """
    if not (0.0 < target_fdr < 1.0):
        raise ValueError(f"target_fdr must be in (0, 1), got {target_fdr}")

    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return {
            "q_values": [],
            "is_discovery": [],
            "n_discoveries": 0,
            "critical_p_cutoff": 0.0,
            "target_fdr": target_fdr,
            "guarantee_statement": "Empty input array.",
        }

    # Sort indices
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]

    # Critical thresholds: (i / m) * q*
    ranks = np.arange(1, m + 1, dtype=np.float64)
    thresholds = (ranks / float(m)) * target_fdr

    # Find k = max { i : sorted_p[i] <= threshold[i] }
    significant = sorted_p <= thresholds
    if np.any(significant):
        max_k_idx = int(np.where(significant)[0][-1])
        critical_p_cutoff = float(sorted_p[max_k_idx])
    else:
        max_k_idx = -1
        critical_p_cutoff = 0.0

    # Monotonic adjusted q-values:
    # q_(i) = min_{j >= i} (m / j) * p_(j)
    raw_q = (float(m) / ranks) * sorted_p
    # Backward cumulative minimum to enforce monotonicity
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    monotone_q = np.clip(monotone_q, 0.0, 1.0)

    # Restore original input ordering
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(m)
    final_q = monotone_q[inverse_order]
    is_discovery = final_q <= target_fdr

    n_discoveries = int(np.sum(is_discovery))

    guarantee = (
        f"Benjamini-Hochberg multiple testing correction controls False Discovery Rate "
        f"E[FP / Total Discoveries] <= {target_fdr:.1%} across {m} hypothesis tests. "
        f"Total certified discoveries: {n_discoveries}."
    )

    return {
        "q_values": [round(float(q), 6) for q in final_q],
        "is_discovery": [bool(d) for d in is_discovery],
        "n_discoveries": n_discoveries,
        "discovery_fraction": round(float(n_discoveries / max(1, m)), 4),
        "critical_p_cutoff": round(critical_p_cutoff, 6),
        "target_fdr": target_fdr,
        "guarantee_statement": guarantee,
    }

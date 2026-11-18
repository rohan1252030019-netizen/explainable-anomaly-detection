"""
src/timeseries/streaming_detector.py

Streaming & Online Anomaly Intelligence:
1. Streaming Half-Space Trees (HS-Trees): O(d * depth) incremental mass-based detector.
2. Streaming Drift Monitor: Live Population Stability Index (PSI) tracking.
3. Concept Drift Response Policy: Auto-recalibration proposal with mandatory human approval.
"""

import hashlib
import time
from typing import Any

import numpy as np

# ── Incremental Half-Space Tree Node ──────────────────────────────────────────

class _HSTNode:
    def __init__(self, depth: int, max_depth: int, min_val: np.ndarray, max_val: np.ndarray, rng: np.random.Generator):
        self.depth = depth
        self.r_mass: float = 0.0  # Reference window mass
        self.w_mass: float = 0.0  # Latest sliding window mass

        if depth < max_depth:
            # Pick a random feature to split
            d = len(min_val)
            self.split_feat = int(rng.integers(0, d))
            # Pick midpoint split
            f_min = min_val[self.split_feat]
            f_max = max_val[self.split_feat]
            self.split_val = float((f_min + f_max) / 2.0)

            # Left child bounding box
            left_max = max_val.copy()
            left_max[self.split_feat] = self.split_val
            self.left = _HSTNode(depth + 1, max_depth, min_val, left_max, rng)

            # Right child bounding box
            right_min = min_val.copy()
            right_min[self.split_feat] = self.split_val
            self.right = _HSTNode(depth + 1, max_depth, right_min, max_val, rng)
        else:
            self.split_feat = -1
            self.left = None
            self.right = None

    def is_leaf(self) -> bool:
        return self.left is None or self.right is None


class StreamingHalfSpaceTrees:
    """
    Streaming Half-Space Trees (HS-Trees) for online anomaly detection.

    Time complexity per sample: O(n_trees * max_depth), independent of N.
    Memory complexity: O(n_trees * 2^max_depth), fixed buffer.

    Parameters
    ----------
    n_features : int
    n_trees : int, default=25
    max_depth : int, default=8
    window_size : int, default=200
        Number of streaming observations in the active sliding window.
    random_state : int, default=42
    """

    def __init__(
        self,
        n_features: int,
        n_trees: int = 25,
        max_depth: int = 8,
        window_size: int = 200,
        random_state: int = 42,
    ):
        self.n_features = n_features
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.window_size = window_size
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)
        self.trees: list[_HSTNode] = []
        self.samples_in_current_window = 0
        self.total_samples_processed = 0
        self.max_seen_mass: float = 1.0

        # Initial bounding box [-3, 3] standard normal assumption, widened dynamically
        self.min_val = np.full(n_features, -4.0, dtype=np.float64)
        self.max_val = np.full(n_features, 4.0, dtype=np.float64)

        for _ in range(self.n_trees):
            self.trees.append(_HSTNode(0, self.max_depth, self.min_val, self.max_val, self.rng))

    def fit_reference(self, X_ref: np.ndarray) -> "StreamingHalfSpaceTrees":
        """Initialize reference node mass from a reference dataset."""
        X = np.asarray(X_ref, dtype=np.float64)
        for row in X:
            self._update_mass(row, is_reference=True)
        # Calibrate baseline maximum mass
        masses = [self._traverse_and_score(row) for row in X]
        if masses:
            self.max_seen_mass = max(1.0, float(np.percentile(masses, 95)))
        return self

    def _traverse_and_score(self, x: np.ndarray) -> float:
        """Accumulate inverse mass score across all trees."""
        total_score = 0.0
        for tree in self.trees:
            curr = tree
            while curr is not None:
                # Mass at current depth
                depth_weight = 2.0 ** curr.depth
                # Inverse reference mass
                total_score += curr.r_mass * depth_weight
                if curr.is_leaf():
                    break
                if x[curr.split_feat] <= curr.split_val:
                    curr = curr.left
                else:
                    curr = curr.right
        return total_score

    def _update_mass(self, x: np.ndarray, is_reference: bool = False) -> None:
        """Increment mass counters down the path of x."""
        for tree in self.trees:
            curr = tree
            while curr is not None:
                if is_reference:
                    curr.r_mass += 1.0
                else:
                    curr.w_mass += 1.0
                if curr.is_leaf():
                    break
                if x[curr.split_feat] <= curr.split_val:
                    curr = curr.left
                else:
                    curr = curr.right

    def _transfer_window_mass(self) -> None:
        """Slide window: transfer w_mass to r_mass and reset w_mass."""
        for tree in self.trees:
            stack = [tree]
            while stack:
                node = stack.pop()
                node.r_mass = node.w_mass
                node.w_mass = 0.0
                if not node.is_leaf():
                    if node.left:
                        stack.append(node.left)
                    if node.right:
                        stack.append(node.right)
        self.samples_in_current_window = 0

    def score_one(self, x: np.ndarray) -> dict[str, Any]:
        """
        Process a single streaming observation x.

        Returns
        -------
        dict containing:
            - raw_mass: float (low mass = high anomaly)
            - anomaly_score: float in [0, 1]
            - samples_seen: int
        """
        x_row = np.asarray(x, dtype=np.float64).ravel()
        if len(x_row) != self.n_features:
            raise ValueError(f"Feature count mismatch: expected {self.n_features}, got {len(x_row)}")

        # 1. Score observation against existing reference mass
        raw_mass = self._traverse_and_score(x_row)

        # 2. Update window mass
        self._update_mass(x_row, is_reference=False)
        self.samples_in_current_window += 1
        self.total_samples_processed += 1

        # Check if sliding window full
        if self.samples_in_current_window >= self.window_size:
            self._transfer_window_mass()

        # Update running max mass
        if raw_mass > self.max_seen_mass:
            self.max_seen_mass = raw_mass

        # Invert mass to bounded anomaly score: 0 mass = 1.0 (anomaly), max mass = 0.0 (normal)
        anomaly_score = float(np.clip(1.0 - (raw_mass / max(self.max_seen_mass, 1.0)), 0.0, 1.0))

        return {
            "raw_mass": round(raw_mass, 2),
            "anomaly_score": round(anomaly_score, 4),
            "total_samples_processed": self.total_samples_processed,
        }


# ── Streaming Population Stability Index (PSI) & Drift Monitor ─────────────────

def _compute_psi(ref: np.ndarray, curr: np.ndarray, bins: int = 5) -> float:
    """Compute Population Stability Index between reference and current distribution."""
    # Compute quantiles on reference
    q = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(ref, q)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    # Convert to fractions with Laplace smoothing
    ref_pct = (ref_counts + 1e-4) / (len(ref) + 1e-4 * bins)
    curr_pct = (curr_counts + 1e-4) / (len(curr) + 1e-4 * bins)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi))


class StreamingDriftMonitor:
    """
    Live sliding-window drift monitor tracking Population Stability Index (PSI).

    Drift Thresholds:
    - PSI < 0.10: Stable (no significant distribution change)
    - 0.10 <= PSI < 0.25: Moderate Drift Warning
    - PSI >= 0.25: Significant Drift Alarm (triggers recalibration policy)
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: list[str],
        window_size: int = 100,
        psi_threshold_warning: float = 0.10,
        psi_threshold_alarm: float = 0.25,
    ):
        self.ref_data = np.asarray(reference_data, dtype=np.float64)
        self.feature_names = feature_names
        self.window_size = window_size
        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_alarm = psi_threshold_alarm

        self.buffer: list[np.ndarray] = []
        self.last_psi_report: dict[str, Any] | None = None
        self.pending_recalibration_proposal: dict[str, Any] | None = None

    def add_observation(self, x: np.ndarray) -> dict[str, Any]:
        """
        Ingest a streaming observation and update PSI metrics.

        Returns
        -------
        dict containing status, overall PSI, max feature drift, and alarms.
        """
        x_row = np.asarray(x, dtype=np.float64).ravel()
        self.buffer.append(x_row)

        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # If buffer is not full enough yet, report calibrating
        if len(self.buffer) < min(20, self.window_size):
            return {
                "status": "Buffering",
                "buffer_size": len(self.buffer),
                "overall_psi": 0.0,
                "alarm": False,
            }

        curr_mat = np.vstack(self.buffer)
        feature_psis = {}
        for j, name in enumerate(self.feature_names):
            psi_val = _compute_psi(self.ref_data[:, j], curr_mat[:, j])
            feature_psis[name] = round(psi_val, 4)

        overall_psi = round(float(np.mean(list(feature_psis.values()))), 4)
        max_feat = max(feature_psis.items(), key=lambda item: item[1])

        # Evaluate drift tier
        if overall_psi >= self.psi_threshold_alarm:
            drift_status = "CRITICAL_DRIFT_ALARM"
            alarm = True
            # Create a formal recalibration proposal requiring human sign-off
            if self.pending_recalibration_proposal is None:
                self.pending_recalibration_proposal = {
                    "proposal_id": hashlib.sha256(f"{time.time()}_{overall_psi}".encode()).hexdigest()[:12],
                    "timestamp": time.time(),
                    "triggering_psi": overall_psi,
                    "primary_drift_feature": max_feat[0],
                    "proposed_new_baseline_samples": len(self.buffer),
                    "status": "Awaiting Human Approval",
                }
        elif overall_psi >= self.psi_threshold_warning:
            drift_status = "MODERATE_DRIFT_WARNING"
            alarm = False
        else:
            drift_status = "STABLE"
            alarm = False

        self.last_psi_report = {
            "status": drift_status,
            "overall_psi": overall_psi,
            "max_drifting_feature": max_feat[0],
            "max_feature_psi": max_feat[1],
            "feature_psis": feature_psis,
            "alarm": alarm,
            "buffer_size": len(self.buffer),
            "has_recalibration_proposal": self.pending_recalibration_proposal is not None,
            "recalibration_proposal": self.pending_recalibration_proposal,
        }
        return self.last_psi_report

    def approve_recalibration(self, approved: bool) -> dict[str, Any]:
        """Human-in-the-loop decision to recalibrate baseline."""
        if self.pending_recalibration_proposal is None:
            return {"success": False, "message": "No pending recalibration proposal to approve."}

        proposal = self.pending_recalibration_proposal
        if approved:
            # Update reference baseline to the recent streaming buffer
            self.ref_data = np.vstack(self.buffer)
            self.pending_recalibration_proposal = None
            return {
                "success": True,
                "action": "Baseline recalibrated to recent streaming distribution.",
                "proposal_id": proposal["proposal_id"],
                "new_reference_size": len(self.ref_data),
            }
        else:
            self.pending_recalibration_proposal = None
            return {
                "success": True,
                "action": "Recalibration proposal dismissed by analyst. Baseline preserved.",
                "proposal_id": proposal["proposal_id"],
            }

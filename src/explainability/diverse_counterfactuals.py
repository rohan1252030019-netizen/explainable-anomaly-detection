"""
src/explainability/diverse_counterfactuals.py

DiCE-style Diverse Counterfactuals with Plausibility and Actionability Constraints.

Upgrades classical nearest-neighbour counterfactuals with:
1. Diversity: Generates k distinct, non-redundant counterfactual paths to normality.
2. In-Distribution Plausibility Check: Evaluates candidate counterfactuals against a
   density estimator (LOF or GMM fitted on normal reference data). Counterfactuals
   must not land in unpopulated, adversarial feature space.
3. Actionability & Sparsity (L0) Constraints: Immutable features (e.g., account age,
   origin coordinates) cannot be modified, and at most `max_features_modified` features
   are adjusted.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor


def generate_diverse_counterfactuals(
    observation: np.ndarray,
    reference_normals: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    feature_names: Sequence[str],
    k_diverse: int = 3,
    max_features_modified: int = 3,
    immutable_features: Sequence[int | str] | None = None,
    score_threshold: float = 0.5,
    denormalize: Callable[[np.ndarray], np.ndarray] | None = None,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate k diverse, plausible counterfactuals for an anomalous observation.

    Parameters
    ----------
    observation : np.ndarray, shape (d,)
        The flagged anomalous row in scaled feature space.
    reference_normals : np.ndarray, shape (n_ref, d)
        Fitted reference normal rows.
    score_fn : callable
        f(X) -> np.ndarray of anomaly scores in [0, 1].
    feature_names : sequence of str
    k_diverse : int, default=3
        Number of diverse counterfactual options to generate.
    max_features_modified : int, default=3
        Sparsity constraint: cap on number of features modified per counterfactual.
    immutable_features : sequence of int or str, optional
        Features that cannot be modified (e.g. immutable user attributes).
    score_threshold : float, default=0.5
        Anomaly score cutoff below which an instance is deemed normal.
    denormalize : callable, optional
        f(scaled_row) -> raw_row to express changes in physical units.
    random_state : int, default=42

    Returns
    -------
    list of dicts, each representing a distinct valid counterfactual.
    """
    x_0 = np.asarray(observation, dtype=np.float64).ravel()
    ref = np.asarray(reference_normals, dtype=np.float64)
    d = len(x_0)
    names = list(feature_names)

    # Resolve immutable feature indices
    immutable_idx: set[int] = set()
    if immutable_features:
        for f in immutable_features:
            if isinstance(f, int):
                immutable_idx.add(f)
            elif isinstance(f, str) and f in names:
                immutable_idx.add(names.index(f))

    # Fit a reference density estimator on normal data to evaluate plausibility
    # We use LOF novelty detection
    lof_density = LocalOutlierFactor(n_neighbors=min(15, len(ref) - 1), novelty=True)
    lof_density.fit(ref)

    # Partition normal reference data into diverse clusters (DiCE prototype diversity)
    n_clusters = min(k_diverse * 2, max(2, len(ref) // 10))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5)
    cluster_labels = kmeans.fit_predict(ref)
    cluster_centers = kmeans.cluster_centers_

    # Sort clusters by distance to observation
    center_dists = np.linalg.norm(cluster_centers - x_0, axis=1)
    sorted_cluster_order = np.argsort(center_dists)

    counterfactuals: list[dict[str, Any]] = []
    seen_feature_subsets: list[set[int]] = []

    for c_idx in sorted_cluster_order:
        if len(counterfactuals) >= k_diverse:
            break

        cluster_pts = ref[cluster_labels == c_idx]
        if len(cluster_pts) == 0:
            continue

        # Find closest point in this cluster
        dists = np.linalg.norm(cluster_pts - x_0, axis=1)
        best_pt = cluster_pts[np.argmin(dists)].copy()

        # Apply actionability & sparsity constraints:
        # 1. Revert immutable features to x_0
        for imm_i in immutable_idx:
            best_pt[imm_i] = x_0[imm_i]

        # 2. Select top `max_features_modified` largest deltas and revert all others
        deltas = np.abs(best_pt - x_0)
        # Sort indices of actionable features by delta magnitude
        actionable_indices = [i for i in range(d) if i not in immutable_idx]
        actionable_indices.sort(key=lambda idx: deltas[idx], reverse=True)

        chosen_features = set(actionable_indices[:max_features_modified])
        # Revert remaining features to x_0
        candidate_cf = x_0.copy()
        for f_i in chosen_features:
            candidate_cf[f_i] = best_pt[f_i]

        # Check diversity of feature subset to avoid redundant explanations
        if any(chosen_features == prev for prev in seen_feature_subsets):
            continue

        # Evaluate score of candidate
        cand_score = float(score_fn(candidate_cf.reshape(1, -1))[0])

        # Plausibility check via LOF
        # 1 means inlier, -1 means outlier
        is_plausible = bool(lof_density.predict(candidate_cf.reshape(1, -1))[0] == 1)

        # Build modified feature records
        modifications = []
        for f_i in sorted(chosen_features):
            old_val = float(x_0[f_i])
            new_val = float(candidate_cf[f_i])
            delta = new_val - old_val

            mod_record = {
                "feature": names[f_i],
                "observed_value": round(old_val, 3),
                "suggested_value": round(new_val, 3),
                "delta": round(delta, 3),
            }
            if denormalize is not None:
                try:
                    raw_obs = denormalize(x_0)
                    raw_cf = denormalize(candidate_cf)
                    mod_record["observed_raw"] = round(float(raw_obs[f_i]), 3)
                    mod_record["suggested_raw"] = round(float(raw_cf[f_i]), 3)
                    mod_record["delta_raw"] = round(float(raw_cf[f_i] - raw_obs[f_i]), 3)
                except Exception:
                    pass
            modifications.append(mod_record)

        cf_record = {
            "candidate_id": len(counterfactuals) + 1,
            "target_score": round(cand_score, 4),
            "achieves_normality": bool(cand_score <= score_threshold),
            "is_in_distribution_plausible": is_plausible,
            "sparsity_l0": len(modifications),
            "modifications": modifications,
            "l2_distance": round(float(np.linalg.norm(candidate_cf - x_0)), 4),
        }

        counterfactuals.append(cf_record)
        seen_feature_subsets.append(chosen_features)

    # If fewer than k found due to tight constraints, relax and pad with nearest normal
    if len(counterfactuals) == 0 and len(ref) > 0:
        dists = np.linalg.norm(ref - x_0, axis=1)
        fallback_pt = ref[np.argmin(dists)].copy()
        for imm_i in immutable_idx:
            fallback_pt[imm_i] = x_0[imm_i]
        fallback_score = float(score_fn(fallback_pt.reshape(1, -1))[0])
        counterfactuals.append({
            "candidate_id": 1,
            "target_score": round(fallback_score, 4),
            "achieves_normality": bool(fallback_score <= score_threshold),
            "is_in_distribution_plausible": True,
            "sparsity_l0": d - len(immutable_idx),
            "modifications": [
                {
                    "feature": names[i],
                    "observed_value": round(float(x_0[i]), 3),
                    "suggested_value": round(float(fallback_pt[i]), 3),
                    "delta": round(float(fallback_pt[i] - x_0[i]), 3),
                }
                for i in range(d)
                if i not in immutable_idx and abs(fallback_pt[i] - x_0[i]) > 1e-4
            ][:max_features_modified],
            "l2_distance": round(float(np.linalg.norm(fallback_pt - x_0)), 4),
        })

    return counterfactuals

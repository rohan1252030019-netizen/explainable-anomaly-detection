"""
src/audit/root_cause.py

Root-Cause Clustering for Anomaly Intelligence.

Methodology:
Traditional clustering on raw features groups records by *what* they are.
SENTINEL Root-Cause Clustering runs HDBSCAN on feature attribution vectors (SHAP/LIME),
grouping anomalies by *why* they were flagged.

Each cluster represents a distinct operational failure mode, threat campaign, or bug archetype,
named automatically by its dominant attribution drivers.
"""

from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN


def cluster_root_causes(
    attribution_matrix: np.ndarray,
    anomaly_indices: list[int],
    feature_names: list[str],
    min_cluster_size: int = 3,
) -> dict[str, Any]:
    """
    Cluster anomalies based on feature attribution profiles.

    Parameters
    ----------
    attribution_matrix : np.ndarray, shape (n_anomalies, n_features)
        Attribution vectors for flagged anomalies.
    anomaly_indices : list of int
        Original dataset row indices corresponding to each attribution row.
    feature_names : list of str
    min_cluster_size : int, default=3

    Returns
    -------
    dict containing:
        - n_clusters: int
        - clusters: list of dicts describing each root-cause archetype
        - unclustered_anomalies: int
        - cluster_labels: list of int
    """
    attrs = np.asarray(attribution_matrix, dtype=np.float64)
    n_anom, n_feat = attrs.shape

    if n_anom < min_cluster_size:
        return {
            "n_clusters": 0,
            "clusters": [],
            "unclustered_anomalies": n_anom,
            "cluster_labels": [-1] * n_anom,
            "summary": "Too few anomalies to form statistical root-cause clusters.",
        }

    # Normalize attribution vectors to unit length so direction (root-cause signature) dominates magnitude
    norms = np.linalg.norm(attrs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    norm_attrs = attrs / norms

    # Fit HDBSCAN on normalized attribution vectors
    clusterer = HDBSCAN(
        min_cluster_size=min(min_cluster_size, max(2, n_anom // 3)),
        metric="euclidean",
    )
    labels = clusterer.fit_predict(norm_attrs)

    unique_labels = sorted(list(set(labels)))
    clusters = []

    for clus_id in unique_labels:
        if clus_id == -1:
            continue  # Noise / outlier points in HDBSCAN

        mask = labels == clus_id
        cluster_attrs = attrs[mask]
        cluster_orig_indices = [anomaly_indices[i] for i, m in enumerate(mask) if m]
        size = int(np.sum(mask))

        # Mean attribution vector for this cluster
        mean_attr = np.mean(cluster_attrs, axis=0)

        # Top 3 dominant features defining this root-cause cluster
        top_feat_indices = np.argsort(np.abs(mean_attr))[::-1][:3]
        dominant_features = [
            {
                "feature": feature_names[i],
                "mean_attribution": round(float(mean_attr[i]), 4),
                "direction": "Elevated Risk" if mean_attr[i] > 0 else "Suppressed Risk",
            }
            for i in top_feat_indices
        ]

        # Representative exemplar: point closest to centroid
        centroid = np.mean(norm_attrs[mask], axis=0)
        dists = np.linalg.norm(norm_attrs[mask] - centroid, axis=1)
        exemplar_idx = cluster_orig_indices[int(np.argmin(dists))]

        # Automatic cluster naming
        top_names = [f["feature"] for f in dominant_features[:2]]
        cluster_name = f"Archetype {clus_id + 1}: Driven by {' & '.join(top_names)}"

        clusters.append({
            "cluster_id": int(clus_id),
            "cluster_name": cluster_name,
            "size": size,
            "fraction_of_anomalies": round(float(size / n_anom), 4),
            "dominant_features": dominant_features,
            "exemplar_record_index": exemplar_idx,
            "member_record_indices": cluster_orig_indices,
        })

    # Sort clusters by size descending
    clusters.sort(key=lambda c: c["size"], reverse=True)
    n_noise = int(np.sum(labels == -1))

    return {
        "n_clusters": len(clusters),
        "clusters": clusters,
        "unclustered_anomalies": n_noise,
        "cluster_labels": [int(lbl) for lbl in labels],
        "summary": (
            f"Partitioned {n_anom} anomalies into {len(clusters)} distinct root-cause archetypes "
            f"({n_noise} isolated noise points)."
        ),
    }

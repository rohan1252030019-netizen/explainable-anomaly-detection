"""
src/explainability/feature_contributions.py

Feature-level contribution analysis for anomaly explanations.

Methodology
-----------
We use THREE complementary approaches and combine them:

A. Statistical deviation
   Compare the observation's raw feature values against the reference
   distribution of normal training samples (median, IQR, z-score).
   Produces a per-feature deviation score.

B. Perturbation-based importance (model-agnostic)
   For each feature, mask it (replace with the reference/median value) and
   measure how much the normalised anomaly score changes.
   Features that most reduce the score when masked are the most influential.

C. Scaled contribution
   Combine deviation and perturbation signals into a single contribution
   score, ranked so the top-k features can be shown to the user.

IMPORTANT:
- These are *model-based explanations*, not causal proofs.
- Language used: "feature contribution", "association", "deviation".
- We do NOT claim a feature *caused* the anomaly.
"""

from typing import Any

import numpy as np
import pandas as pd

# ── Statistical deviation ──────────────────────────────────────────────────────

def compute_statistical_deviations(
    observation: np.ndarray,
    feature_names: list[str],
    reference_stats: dict[str, dict[str, float]],
    raw_observation: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Compute statistical deviations for each feature of a single observation.

    Unit spaces
    -----------
    ``observation`` is in the model's SCALED space.  ``reference_stats`` (from
    ``PreprocessingPipeline.get_reference_stats()``) is deliberately in RAW
    dataset units so the UI can show real numbers.

    Comparing those two directly was silently wrong: it produced a robust
    z-score of, for example, ``(4.4 - 35.2) / (9.8 * 1.4826)`` for a feature
    whose standardised value was 4.4 — a large negative number for a value that
    is far ABOVE the median.  Pass ``raw_observation`` so the comparison is made
    in one consistent space.

    Parameters
    ----------
    observation : np.ndarray, shape (n_features,)
        The preprocessed (scaled) feature vector.
    feature_names : list of str
    reference_stats : dict
        Output of PreprocessingPipeline.get_reference_stats().
        Keys: "median", "mean", "std", "q25", "q75", "iqr", "min", "max".
        Only NUMERICAL features have entries; OHE features are handled
        separately.
    raw_observation : np.ndarray or None
        The same row in original units, aligned to ``feature_names``.  Supply
        this whenever it is available; without it the numerical deviations fall
        back to the scaled-vs-raw comparison described above and should be
        treated as untrustworthy.

    Returns
    -------
    list of dicts, one per feature, sorted by |deviation| descending.
    """
    obs = np.asarray(observation, dtype=np.float64).ravel()
    raw = (
        np.asarray(raw_observation, dtype=np.float64).ravel()
        if raw_observation is not None
        else None
    )
    if raw is not None and raw.shape[0] != obs.shape[0]:
        raw = None

    median_stats = reference_stats.get("median", {})
    std_stats = reference_stats.get("std", {})
    q25_stats = reference_stats.get("q25", {})
    q75_stats = reference_stats.get("q75", {})
    iqr_stats = reference_stats.get("iqr", {})

    results: list[dict[str, Any]] = []

    for i, feat in enumerate(feature_names):
        # For numerical features, compare like with like.  One-hot columns have
        # no raw counterpart, so their activation is used directly.
        is_categorical = feat not in median_stats
        val = float(raw[i]) if (raw is not None and not is_categorical) else float(obs[i])

        if not is_categorical:
            med = median_stats[feat]
            std = std_stats.get(feat, 1.0)
            q25 = q25_stats.get(feat, med)
            q75 = q75_stats.get(feat, med)
            iqr = iqr_stats.get(feat, 1.0)

            # Robust z-score (using IQR instead of std for robustness)
            if iqr > 1e-10:
                robust_z = (val - med) / (iqr * 1.4826)  # 1.4826 = consistency factor
            elif std > 1e-10:
                robust_z = (val - med) / std
            else:
                robust_z = 0.0

            deviation_magnitude = abs(robust_z)
            deviation_direction = "above" if val > med else "below"

            results.append({
                "feature": feat,
                "feature_type": "numerical",
                "observed_value": val,
                "reference_value": med,
                "q25": q25,
                "q75": q75,
                "deviation_magnitude": deviation_magnitude,
                "deviation_direction": deviation_direction,
                "robust_z_score": round(robust_z, 3),
                "units_match": raw is not None,
            })
        else:
            # OHE binary feature: contribution based on activation
            results.append({
                "feature": feat,
                "feature_type": "categorical_ohe",
                "observed_value": val,
                "reference_value": 0.0,
                "q25": 0.0,
                "q75": 1.0,
                "deviation_magnitude": abs(val),
                "deviation_direction": "active" if val > 0.5 else "inactive",
                "robust_z_score": float(val),
                "units_match": True,
            })

    results.sort(key=lambda x: x["deviation_magnitude"], reverse=True)
    return results


# ── Perturbation-based importance ──────────────────────────────────────────────

def _as_scalar(value: Any, context: str = "score") -> float:
    """
    Coerce whatever ``score_fn`` returned into a single float.

    Score functions in this project legitimately return either a scalar or a
    one-element array (normalisation pipelines return ``np.ndarray``). NumPy 2
    refuses ``float()`` on a one-element array, so the coercion has to be
    explicit. Getting this wrong used to raise inside
    ``perturbation_importance``, where the caller's broad ``except`` swallowed
    it and silently downgraded every explanation to statistics-only.
    """
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"The {context} function returned an empty result.")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError(f"The {context} function returned no finite values.")
    return float(finite[0])

def perturbation_importance(
    observation: np.ndarray,
    score_fn,
    feature_names: list[str],
    baseline: np.ndarray | None = None,
    n_repeats: int = 1,
) -> list[dict[str, Any]]:
    """
    Estimate feature importance by masking each feature one at a time.

    For each feature i:
    1. Replace observation[i] with baseline[i] (median/zero).
    2. Compute the new anomaly score.
    3. score_delta = original_score - new_score
       Positive delta → removing this feature DECREASED the anomaly score
       → the feature was CONTRIBUTING to the anomaly.

    Parameters
    ----------
    observation : np.ndarray, shape (n_features,)
        Preprocessed feature vector of the anomalous observation.
    score_fn : callable
        Function that takes np.ndarray (1, n_features) and returns float.
        Should return a NORMALISED anomaly score in [0, 1].
    feature_names : list of str
    baseline : np.ndarray or None
        Baseline values (zeros after scaling, meaning the mean).
        Defaults to zeros.
    n_repeats : int
        Number of repeats per feature (with slight noise) for stability.

    Returns
    -------
    list of dicts sorted by importance (descending), each with:
        feature, importance, score_delta, original_score, masked_score
    """
    observation = np.asarray(observation, dtype=np.float64)
    n_features = len(observation)

    if baseline is None:
        # After StandardScaler, 0 = mean of training data
        baseline = np.zeros(n_features, dtype=np.float64)

    # Original score
    original_score = _as_scalar(
        score_fn(observation.reshape(1, -1)),
        context="original score",
    )

    importances: list[dict[str, Any]] = []
    for i, feat in enumerate(feature_names):
        perturbed = observation.copy()
        perturbed[i] = baseline[i]

        masked_score = _as_scalar(
            score_fn(perturbed.reshape(1, -1)),
            context=f"masked score for '{feat}'",
        )
        delta = original_score - masked_score  # positive = feature contributed to anomaly

        importances.append({
            "feature": feat,
            "importance": abs(delta),       # unsigned importance
            "score_delta": round(delta, 4),
            "original_score": round(original_score, 4),
            "masked_score": round(masked_score, 4),
        })

    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances


# ── Combined contribution ──────────────────────────────────────────────────────

def combined_feature_contributions(
    stat_devs: list[dict[str, Any]],
    perturbation_imps: list[dict[str, Any]] | None,
    top_k: int = 10,
    stat_weight: float = 0.5,
    perturb_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Merge statistical deviation and perturbation importance into a single
    ranked contribution list.

    Parameters
    ----------
    stat_devs : list of dicts (from compute_statistical_deviations)
    perturbation_imps : list of dicts or None (from perturbation_importance)
    top_k : int
    stat_weight, perturb_weight : float

    Returns
    -------
    list of contribution dicts, sorted by combined_score descending.
    """
    # Index by feature name
    stat_map = {d["feature"]: d for d in stat_devs}

    # Normalise deviation magnitudes to [0, 1]
    stat_mags = np.array([d["deviation_magnitude"] for d in stat_devs])
    stat_max = stat_mags.max() if stat_mags.max() > 0 else 1.0
    stat_norm = {d["feature"]: d["deviation_magnitude"] / stat_max for d in stat_devs}

    if perturbation_imps:
        perturb_map = {d["feature"]: d for d in perturbation_imps}
        perturb_mags = np.array([d["importance"] for d in perturbation_imps])
        perturb_max = perturb_mags.max() if perturb_mags.max() > 0 else 1.0
        perturb_norm = {
            d["feature"]: d["importance"] / perturb_max
            for d in perturbation_imps
        }
    else:
        perturb_map = {}
        perturb_norm = {}
        stat_weight = 1.0
        perturb_weight = 0.0

    combined = []
    for feat, sd in stat_map.items():
        s_score = stat_norm.get(feat, 0.0)
        p_score = perturb_norm.get(feat, 0.0)
        combined_score = stat_weight * s_score + perturb_weight * p_score

        entry = {
            "feature": feat,
            "feature_type": sd["feature_type"],
            "observed_value": sd["observed_value"],
            "reference_value": sd["reference_value"],
            "deviation_direction": sd["deviation_direction"],
            "robust_z_score": sd.get("robust_z_score", 0.0),
            "stat_score": round(s_score, 4),
            "perturb_score": round(p_score, 4),
            "combined_score": round(combined_score, 4),
            "score_delta": perturb_map.get(feat, {}).get("score_delta", None),
        }
        combined.append(entry)

    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    return combined[:top_k]


# ── Global feature importance ──────────────────────────────────────────────────

def global_feature_importance(
    anomaly_contributions: list[list[dict[str, Any]]],
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Aggregate per-anomaly contributions into a global importance DataFrame.

    Parameters
    ----------
    anomaly_contributions : list of lists
        One inner list per anomaly, each from combined_feature_contributions().
    feature_names : list of str

    Returns
    -------
    pd.DataFrame with columns:
        feature, mean_combined_score, frequency, mean_z_score, rank
    Sorted by mean_combined_score descending.
    """
    score_accumulator: dict[str, list[float]] = {f: [] for f in feature_names}
    z_accumulator: dict[str, list[float]] = {f: [] for f in feature_names}
    freq_counter: dict[str, int] = {f: 0 for f in feature_names}

    for contribs in anomaly_contributions:
        if not contribs:
            continue
        contribs[0]["feature"] if contribs else None
        for entry in contribs:
            feat = entry["feature"]
            if feat in score_accumulator:
                score_accumulator[feat].append(entry["combined_score"])
                z_accumulator[feat].append(abs(entry.get("robust_z_score", 0.0)))
                if entry["combined_score"] > 0.1:
                    freq_counter[feat] += 1

    rows = []
    for feat in feature_names:
        scores = score_accumulator[feat]
        zs = z_accumulator[feat]
        rows.append({
            "feature": feat,
            "mean_combined_score": round(np.mean(scores), 4) if scores else 0.0,
            "frequency": freq_counter[feat],
            "mean_z_score": round(np.mean(zs), 3) if zs else 0.0,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_combined_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df

"""
src/explainability/counterfactuals.py

Actionable counterfactual recommendations for flagged anomalies.

Why this matters
----------------
Attribution answers "which features were unusual?".  That is diagnostic but it
does not tell an analyst what to DO.  A triage tool is far more useful when it
can also say: "this row would look normal if transaction_amount were 41.20
instead of 310.40 and geographic_distance_km were 50.10 instead of 640.00".

Method
------
Nearest-normal-counterfactual (also called a prototype-based counterfactual):

1. Take the set of rows the detector did NOT flag.  Those define "normal" in
   this model's terms — the same definition that produced the alert, so the
   recommendation is consistent with why it fired.
2. For the flagged row, search that reference set for the closest row, using a
   distance that respects the problem's structure.
3. Report the features that differ, in original (unscaled) units, with the
   required change and the anomaly score before and after.

Two refinements over plain nearest-neighbour:

- **Weighted distance.**  Raw Euclidean distance on standardised features
  treats a rare-but-inert feature and a decisive one as equally important.
  We weight each feature by its importance from the explanation, so the search
  looks for a normal row that differs mainly on the features the model
  actually used to flag the row.

- **Similarity inside the neighbourhood.**  Reaching the single closest normal
  row is usually arbitrary when many are equally close, so we require the
  counterfactual to be closer to the flagged row than the flagged row is to the
  *median* normal row.  This keeps the advice minimal rather than jumping to a
  distant prototype.

Honest limitations
------------------
- This is a nearest-neighbour suggestion, not a generative counterfactual.  It
  does not respect feature dependencies: changing one column of a real row to
  match another real row can imply an impossible combination.
- It is an association, not a causal claim.  "Moving this value would lower the
  score" describes the model's response; it does not assert that doing so in
  the real world resolves the underlying issue.
- Reported changes are the minimum needed to reach a neighbour, not an
  optimised intervention.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# ── Core counterfactual search ─────────────────────────────────────────────────

def nearest_normal_counterfactual(
    observation: np.ndarray,
    reference: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    feature_importance: Sequence[float] | None = None,
    feature_names: Sequence[str] | None = None,
    original_units: Sequence[float] | np.ndarray | None = None,
    denormalize: Callable[[np.ndarray], np.ndarray] | None = None,
    max_changes: int = 4,
    require_closer_than_median: bool = True,
    score_threshold: float | None = None,
    immutable_features: Sequence[int | str] | None = None,
) -> dict[str, Any]:
    """
    Find the closest non-flagged row and describe the changes it implies.

    Parameters
    ----------
    observation : np.ndarray, shape (n_features,)
        The flagged row, in the same scaled space as ``reference``.
    reference : np.ndarray, shape (n_reference, n_features)
        Rows the detector did NOT flag.
    score_fn : callable or None
        ``f(X) -> scores`` for X of shape (n, n_features).  Used to report the
        score before and after the suggested change.  Optional: without it the
        result still describes the nearest neighbour but cannot quantify the
        score reduction.
    feature_importance : sequence of float or None
        Non-negative importance per feature, e.g. the SHAP magnitudes for this
        row.  Used to weight the distance.  Defaults to uniform.
    feature_names : sequence of str or None
        Names for the changed features.  Defaults to ``f0..fn``.
    original_units : sequence of float or None
        Observed values in the dataset's ORIGINAL units, parallel to
        ``observation``.  Only valid when the transform is a pure offset (scale
        ``1``); prefer ``denormalize``, which is always exact.
    denormalize : callable or None
        ``f(scaled_row) -> raw_row`` for a single row.  Preferred over
        ``original_units``: it inverts the *whole* mapping, so the suggested
        value is the neighbour's real value rather than a scaled delta added to
        a raw one.
    max_changes : int
        Cap on how many features to list.
    require_closer_than_median : bool
        Reject the candidate if the flagged row is already closer to the median
        normal row than to the candidate, i.e. if the "counterfactual" is not
        actually an improvement.
    score_threshold : float or None
        The detection threshold on the normalised score.  Used to judge whether
        the counterfactual actually clears the bar, which a nearest-neighbour
        search does not guarantee.

    Returns
    -------
    dict with keys:
        found          whether a usable counterfactual was located
        reason         why not, when found is False
        neighbour_index   row index within `reference`
        distance       weighted distance to that row
        score_before   anomaly score of the original row (needs score_fn)
        score_after    anomaly score of the neighbour (needs score_fn)
        score_reduction score_before - score_after
        n_changes      number of features that differ
        changes        list of per-feature dicts
        text           one-sentence plain-English recommendation
        caveat         standing limitations
    """
    obs = np.asarray(observation, dtype=np.float64).ravel()
    ref = np.atleast_2d(np.asarray(reference, dtype=np.float64))

    if ref.shape[0] == 0:
        return _empty_result("No non-flagged reference rows were supplied.")
    if ref.shape[1] != obs.shape[0]:
        return _empty_result(
            f"Shape mismatch: observation has {obs.shape[0]} features but the "
            f"reference set has {ref.shape[1]}."
        )

    if feature_importance is None:
        weights = np.ones(obs.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(feature_importance, dtype=np.float64).ravel()
        if weights.shape[0] != obs.shape[0]:
            return _empty_result(
                "feature_importance length does not match the feature count."
            )
        # Guard against an all-zero importance vector (division by zero).
        if not np.any(weights > 0):
            weights = np.ones_like(weights)

    names = list(feature_names) if feature_names else [
        f"feature_{i}" for i in range(obs.shape[0])
    ]
    orig = (
        np.asarray(original_units, dtype=np.float64).ravel()
        if original_units is not None
        else None
    )

    # Weighted squared distance to every reference row, vectorised.
    diff = ref - obs
    d2 = (diff * diff * weights).sum(axis=1)
    best_idx = int(np.argmin(d2))
    best_dist = float(np.sqrt(d2[best_idx]))

    # The counterfactual should be a nearer neighbour than the typical normal
    # row, otherwise we are not actually proposing a small change.
    median_d2 = float(np.median(d2))
    if require_closer_than_median and best_dist >= float(np.sqrt(median_d2)):
        return _empty_result(
            "No normal row is meaningfully closer to this observation than the "
            "typical normal row is. This point is not close to any single "
            "prototype; it may belong to a multi-feature outlier region."
        )

    candidate = ref[best_idx].copy()

    # Immutable features: resolve indices and prevent alterations
    imm_indices = set()
    if immutable_features:
        for item in immutable_features:
            if isinstance(item, int) and 0 <= item < obs.shape[0]:
                imm_indices.add(item)
            elif isinstance(item, str) and item in names:
                imm_indices.add(names.index(item))

    # Reset candidate values on immutable features to the observed value
    for idx in imm_indices:
        candidate[idx] = obs[idx]

    # ── Scores before / after ──────────────────────────────────────────────
    score_before = score_after = score_reduction = None
    if score_fn is not None:
        try:
            scores = np.asarray(
                score_fn(np.vstack([obs, candidate])), dtype=np.float64
            ).ravel()
            if scores.size >= 2 and np.isfinite(scores[:2]).all():
                score_before = round(float(scores[0]), 4)
                score_after = round(float(scores[1]), 4)
                score_reduction = round(score_before - score_after, 4)
        except Exception:
            # A failed score call must not lose us the counterfactual itself.
            score_before = score_after = score_reduction = None

    # ── Per-feature changes, most important first ──────────────────────────
    delta = candidate - obs
    # Guard against float noise on features that are effectively identical.
    active = np.abs(delta) > 1e-9

    # Map BOTH endpoints back to original units.  A scaled delta is expressed
    # in standardised units, so ``orig_value + delta`` adds (for example) a
    # number of standard deviations to a number of currency units.  The two
    # endpoints must be inverted separately, then compared.
    obs_orig = cand_orig = None
    denorm_error: str | None = None
    if denormalize is not None:
        try:
            obs_orig = np.asarray(denormalize(obs), dtype=np.float64).ravel()
            cand_orig = np.asarray(denormalize(candidate), dtype=np.float64).ravel()
        except Exception as exc:  # recorded below, not swallowed
            denorm_error = f"{type(exc).__name__}: {exc}"
            obs_orig = cand_orig = None
    if obs_orig is None and orig is not None:
        obs_orig = orig

    order = np.argsort(-(np.abs(delta) * weights))
    changes: list[dict[str, Any]] = []
    for i in order:
        if not active[i]:
            continue
        if len(changes) >= max_changes:
            break

        entry: dict[str, Any] = {
            "feature": names[int(i)],
            "scaled_from": round(float(obs[i]), 4),
            "scaled_to": round(float(candidate[i]), 4),
            "direction": "increase" if delta[i] > 0 else "decrease",
            "magnitude_scaled": round(abs(float(delta[i])), 4),
            "importance_weight": round(float(weights[i]), 4),
        }

        if obs_orig is not None and i < obs_orig.shape[0]:
            entry["observed_value"] = round(float(obs_orig[i]), 4)
            if cand_orig is not None and i < cand_orig.shape[0]:
                target_orig = float(cand_orig[i])
            else:
                # Fallback only reached when no inverse mapping was supplied.
                target_orig = float(obs_orig[i] + delta[i])
            entry["suggested_value"] = round(target_orig, 4)
            entry["change_pct"] = (
                round(
                    float((target_orig - obs_orig[i]) / obs_orig[i]) * 100, 2
                )
                if abs(obs_orig[i]) > 1e-9
                else None
            )
        changes.append(entry)

    n_changes = int(active.sum())
    result: dict[str, Any] = {
        "found": True,
        "reason": None,
        "neighbour_index": best_idx,
        "distance": round(best_dist, 4),
        "score_before": score_before,
        "score_after": score_after,
        "score_reduction": score_reduction,
        "score_threshold": score_threshold,
        "quality": _quality_verdict(
            score_before, score_after, score_threshold, best_dist
        ),
        "n_changes": n_changes,
        "changes": changes,
        "intervention_cost": round(float(np.sum([c.get("magnitude_scaled", 0.0) for c in changes])), 4),
        "immutable_features_enforced": [names[i] for i in imm_indices],
        "denormalization_error": denorm_error,
        "text": _build_text(changes, score_before, score_after, _quality_verdict(
            score_before, score_after, score_threshold, best_dist
        )),
        "caveat": (
            "Nearest-neighbour suggestion, not a causal intervention. The "
            "changed values may not be jointly feasible in the real process, "
            "and lowering the model score does not by itself resolve the "
            "underlying issue."
        ),
    }
    return result


# ── Batch helper ───────────────────────────────────────────────────────────────

def counterfactuals_for_anomalies(
    anomaly_indices: Sequence[int],
    X: np.ndarray,
    normal_mask: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    feature_names: Sequence[str] | None = None,
    original_df: Any = None,
    shap_matrix: np.ndarray | None = None,
    denormalize: Callable[[np.ndarray], np.ndarray] | None = None,
    max_changes: int = 4,
    score_threshold: float | None = None,
    immutable_features: Sequence[int | str] | None = None,
) -> list[dict[str, Any]]:
    """
    Compute counterfactuals for several flagged rows.

    Parameters
    ----------
    anomaly_indices : sequence of int
        Row indices to explain.
    X : np.ndarray, shape (n_samples, n_features)
    normal_mask : np.ndarray, shape (n_samples,)
        Boolean mask of rows that were NOT flagged; these form the reference
        set.
    score_fn : callable or None
    feature_names : sequence of str or None
    original_df : pandas.DataFrame or None
        Original-scale data.  Used as a fallback when ``denormalize`` is absent.
    shap_matrix : np.ndarray or None
        shape (n_samples, n_features) of per-feature importance magnitudes.
        Used to weight the distance toward the features the model relied on.
    denormalize : callable or None
        ``f(scaled_row) -> raw_row``.  Preferred over ``original_df``: the
        scaled feature vector includes one-hot columns whose names will not
        appear in the raw DataFrame, so a name-based lookup silently fails for
        any categorical column and the suggestion falls back to z-units.
    max_changes : int

    Returns
    -------
    list of dicts, one per requested index, each also carrying "row_id".
    """
    X = np.asarray(X, dtype=np.float64)
    mask = np.asarray(normal_mask).ravel().astype(bool)

    if mask.shape[0] != X.shape[0]:
        raise ValueError(
            f"normal_mask has length {mask.shape[0]} but X has {X.shape[0]} rows."
        )

    reference = X[mask]
    out: list[dict[str, Any]] = []

    for row_id in anomaly_indices:
        if row_id < 0 or row_id >= X.shape[0]:
            out.append({
                "row_id": int(row_id),
                "found": False,
                "reason": f"Row {row_id} is outside the dataset.",
                "changes": [],
                "text": "",
            })
            continue

        importance = None
        if shap_matrix is not None:
            sm = np.asarray(shap_matrix, dtype=np.float64)
            if sm.shape[0] == X.shape[0]:
                importance = np.abs(sm[int(row_id)])

        original_units = None
        if denormalize is None and original_df is not None and feature_names is not None:
            original_units = _original_values_for(
                original_df, int(row_id), feature_names
            )

        res = nearest_normal_counterfactual(
            observation=X[int(row_id)],
            reference=reference,
            score_fn=score_fn,
            feature_importance=importance,
            feature_names=feature_names,
            original_units=original_units,
            denormalize=denormalize,
            max_changes=max_changes,
            score_threshold=score_threshold,
            immutable_features=immutable_features,
        )
        res["row_id"] = int(row_id)
        out.append(res)

    return out


# ── Internal helpers ───────────────────────────────────────────────────────────

def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "found": False,
        "reason": reason,
        "neighbour_index": None,
        "distance": None,
        "score_before": None,
        "score_after": None,
        "score_reduction": None,
        "score_threshold": None,
        "quality": {"label": "NONE", "detail": "No counterfactual found."},
        "n_changes": 0,
        "changes": [],
        "text": "No counterfactual recommendation could be generated.",
        "caveat": "",
    }


def _original_values_for(df: Any, row_id: int, feature_names: Sequence[str]) -> np.ndarray | None:
    """Pull one row's values for the named features, tolerating missing columns."""
    try:
        if row_id >= len(df):
            return None
        row = df.iloc[row_id]
        values = []
        for name in feature_names:
            if name in df.columns:
                values.append(float(row[name]))
            else:
                return None
        return np.asarray(values, dtype=np.float64)
    except Exception:
        return None


def _quality_verdict(
    score_before: float | None,
    score_after: float | None,
    score_threshold: float | None,
    distance: float,
) -> dict[str, Any]:
    """
    Judge how good the counterfactual actually is.

    A nearest-neighbour search returns the closest row it can find, which is
    not the same as a row that looks normal.  On the bundled benchmark the
    closest neighbour to a top-ranked anomaly often still scores ~0.35, well
    above the detection threshold, and presenting that as "set these values and
    the problem is solved" would be misleading.  This labels it honestly.
    """
    bar = score_threshold if score_threshold is not None else 0.5

    if score_after is None:
        return {
            "label": "UNVERIFIED",
            "detail": "No score function was supplied, so the effect of the "
                      "suggested change could not be measured.",
        }

    if score_after < bar:
        return {
            "label": "STRONG",
            "detail": (
                f"The nearest normal row scores {score_after:g}, below the "
                f"{bar:g} detection threshold. Adopting these values would "
                "clear the bar."
            ),
        }

    if score_after < bar + 0.25:
        return {
            "label": "PARTIAL",
            "detail": (
                f"The nearest normal row still scores {score_after:g}, above "
                f"the {bar:g} threshold. These changes move the row the right "
                "way but do not fully resolve it; a single-feature fix is "
                "probably not enough."
            ),
        }

    return {
        "label": "WEAK",
        "detail": (
            f"Even the closest normal row scores {score_after:g}, far above the "
            f"{bar:g} threshold. This point is not a near-miss of normal "
            "behaviour, so treat it as a genuinely novel case rather than a "
            "drifting record."
        ),
    }


def _build_text(
    changes: list[dict[str, Any]],
    score_before: float | None,
    score_after: float | None,
    quality: dict[str, Any] | None = None,
) -> str:
    """Compose a one-sentence recommendation."""
    if not changes:
        return (
            "The nearest non-flagged row is almost identical in every feature, "
            "so no single field explains this alert."
        )

    parts = []
    for c in changes[:3]:
        name = c["feature"]
        target = c.get("suggested_value")
        if target is None:
            parts.append(f"{name} {c['direction']} to ~{c['scaled_to']}")
        else:
            parts.append(f"{name} {c['direction']} to ~{target:g}")

    sentence = (
        "The closest non-flagged record differs in: "
        + "; ".join(parts)
        + "."
    )

    if score_before is not None and score_after is not None:
        sentence += (
            f" Moving to that record takes the anomaly score from "
            f"{score_before:g} to {score_after:g}."
        )

    if quality is not None:
        label = quality.get("label")
        if label == "WEAK":
            sentence += " This is a poor match, so it should not be read as a fix."
        elif label == "PARTIAL":
            sentence += " It helps but does not clear the bar on its own."

    return sentence

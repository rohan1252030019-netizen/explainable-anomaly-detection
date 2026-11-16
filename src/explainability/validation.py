"""
src/explainability/validation.py

Quantitative checks on explanation quality.

Motivation
----------
"SHAP says feature A mattered" is an assertion, not a result.  Without a
measure of fidelity, an explanation is unfalsifiable: if the attributions were
reversed or replaced with noise, the dashboard would look exactly the same.
This module supplies the three checks that turn that assertion into something
verifiable, all of which are cheap enough to run per-request.

1. Additivity  — do the attributions sum to the score shown to the user?
   Delegated to :meth:`SHAPExplainer.check_additivity`.

2. Faithfulness (deletion AUC) — if we remove features in the order the
   explanation ranks them, does the anomaly score actually collapse?  A
   faithful explanation should drive the score down fast, giving a LOW
   deletion AUC.  An unfaithful one leaves the score flat and scores near
   0.5, which is the value produced by random feature ordering.  We therefore
   report the deletion AUC of the explanation *relative to* the deletion AUC
   of a random ordering on the same row, which turns an absolute number into
   an interpretable improvement over chance.

3. Stability — how much does the explanation move when the model is refitted
   with a different seed?  Reported as the mean pairwise rank correlation of
   the attributions, plus the share of the top-3 features that survive a
   refit.  Low stability means a user who reruns the pipeline gets a different
   story, which is worth knowing before acting on one.

None of these measure truth.  They measure whether the explanation faithfully
describes the model, which is a different and achievable claim.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# ── Deletion-based faithfulness ────────────────────────────────────────────────

def deletion_curve(
    observation: np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    feature_order: Sequence[int],
    baseline: np.ndarray | None = None,
) -> list[float]:
    """
    Score the observation after masking features one at a time.

    Parameters
    ----------
    observation : np.ndarray, shape (n_features,)
        The row being explained.
    score_fn : callable
        Takes a single row (1-D or 2-D) and returns a scalar score.
    feature_order : sequence of int
        Feature indices in the order the explanation ranks them as most
        important first.
    baseline : np.ndarray or None
        Replacement value used when masking.  Defaults to zeros, which after
        standardisation is the training mean.

    Returns
    -------
    list of float
        ``curve[k]`` is the score after masking the first ``k + 1`` features of
        ``feature_order``.  Length is len(feature_order) + 1.
    """
    obs = np.asarray(observation, dtype=np.float64).ravel()
    base = (
        np.zeros_like(obs)
        if baseline is None
        else np.asarray(baseline, dtype=np.float64).ravel()
    )

    curve = [_as_score(score_fn(obs.copy()))]
    current = obs.copy()
    for idx in feature_order:
        if 0 <= idx < current.shape[0]:
            current[idx] = base[idx]
        curve.append(_as_score(score_fn(current.copy())))
    return curve


def _area_under_curve(values: np.ndarray, x: np.ndarray) -> float:
    """
    Trapezoidal area, across both the NumPy 1.x and 2.x spellings.

    ``np.trapz`` was removed in NumPy 2.0 (renamed ``np.trapezoid``), and this
    module's declared floor is numpy>=1.26.  ``getattr`` keeps the runtime working
    on both while telling the type checker the symbol is resolved dynamically.
    """
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if integrator is None:  # pragma: no cover - no supported NumPy omits both
        raise RuntimeError("NumPy provides neither trapezoid() nor trapz().")
    return float(integrator(values, x))


def deletion_auc(curve: Sequence[float]) -> float:
    """
    Area under the deletion curve, normalised to [0, 1].

    0.0 means the score drops to its floor as soon as anything is removed.
    1.0 means the score never moves.  0.5 is the no-information baseline.
    """
    values = np.asarray(curve, dtype=np.float64)
    if values.size < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, values.size)
    return _area_under_curve(values, x)


# ── Insertion-based faithfulness ───────────────────────────────────────────────

def insertion_curve(
    observation: np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    feature_order: Sequence[int],
    baseline: np.ndarray | None = None,
) -> list[float]:
    """
    Score the observation as features are inserted one by one starting from baseline.

    Parameters
    ----------
    observation : np.ndarray, shape (n_features,)
        The row being explained.
    score_fn : callable
        Takes a single row and returns a scalar score.
    feature_order : sequence of int
        Feature indices in the order the explanation ranks them as most
        important first.
    baseline : np.ndarray or None
        Starting replacement value (defaults to zeros / standardized mean).

    Returns
    -------
    list of float
        curve[k] is the score after inserting the first k features.
    """
    obs = np.asarray(observation, dtype=np.float64).ravel()
    base = (
        np.zeros_like(obs)
        if baseline is None
        else np.asarray(baseline, dtype=np.float64).ravel()
    )

    current = base.copy()
    curve = [_as_score(score_fn(current.copy()))]
    for idx in feature_order:
        if 0 <= idx < current.shape[0]:
            current[idx] = obs[idx]
        curve.append(_as_score(score_fn(current.copy())))
    return curve


def insertion_auc(curve: Sequence[float]) -> float:
    """
    Area under the insertion curve, normalised to [0, 1].

    1.0 means the score reaches its peak immediately upon introducing the first feature.
    0.0 means the score never rises above baseline.
    """
    values = np.asarray(curve, dtype=np.float64)
    if values.size < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, values.size)
    return _area_under_curve(values, x)


def faithfulness(
    observation: np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    feature_order: Sequence[int],
    random_state: int = 42,
    n_random: int = 5,
    baseline: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compare the explanation's deletion AUC against random feature orderings.

    Parameters
    ----------
    observation : np.ndarray, shape (n_features,)
    score_fn : callable
    feature_order : sequence of int
        Feature indices ranked most important first by the explanation.
    random_state, n_random : int
    baseline : np.ndarray or None

    Returns
    -------
    dict with keys:
        deletion_auc_explained   AUC when removing in the explained order
        deletion_auc_random      mean AUC over random orders
        deletion_auc_random_std  spread across those random orders
        improvement              explained AUC minus random AUC (negative is good)
        relative_improvement     improvement as a fraction of random AUC
        n_features               how many features were ablated
        curve                    the explained-order curve
        verdict                  plain-English reading of the result
    """
    obs = np.asarray(observation, dtype=np.float64).ravel()
    order = [int(i) for i in feature_order if 0 <= int(i) < obs.shape[0]]
    if not order:
        return {
            "deletion_auc_explained": None,
            "deletion_auc_random": None,
            "improvement": None,
            "relative_improvement": None,
            "n_features": 0,
            "curve": [],
            "verdict": "No ranked features to test.",
        }

    # Ablate every feature, not just the ranked ones.  Comparing an explained
    # curve of length len(order) + 1 against a random curve of length
    # n_features + 1 integrates two different x-grids, so the two AUCs are not
    # on the same scale and a perfect ranking can appear *worse* than random
    # (measured: 0.3018 vs 0.6078, labelled "counter-indicative").  Ranking the
    # unexplained features last keeps the explained ordering and makes both
    # curves cover the same feature set, which is the only comparison that
    # means anything.
    ranked = set(order)
    full_order = order + [i for i in range(obs.shape[0]) if i not in ranked]

    curve = deletion_curve(obs, score_fn, full_order, baseline=baseline)
    auc_explained = deletion_auc(curve)

    ins_curve = insertion_curve(obs, score_fn, full_order, baseline=baseline)
    auc_insertion = insertion_auc(ins_curve)

    rng = np.random.default_rng(random_state)
    random_aucs = []
    for _ in range(max(1, n_random)):
        shuffled = [int(i) for i in rng.permutation(obs.shape[0])]
        random_aucs.append(
            deletion_auc(deletion_curve(obs, score_fn, shuffled, baseline=baseline))
        )

    auc_random = float(np.mean(random_aucs))
    improvement = auc_explained - auc_random
    relative = improvement / auc_random if abs(auc_random) > 1e-12 else None

    if relative is None:
        verdict = "Could not compare against random ordering."
    elif relative <= -0.30:
        verdict = (
            "Strong. Removing the features this explanation names collapses "
            "the anomaly score far faster than removing random features."
        )
    elif relative <= -0.10:
        verdict = (
            "Reasonable. The named features move the score more than chance, "
            "but not dramatically."
        )
    elif relative < 0.10:
        verdict = (
            "Weak. The named features are barely more informative than random "
            "order; treat this explanation as unverified."
        )
    else:
        verdict = (
            "Counter-indicative. Removing the named features mattered LESS "
            "than random removal, so the ranking is likely wrong."
        )

    return {
        "deletion_auc_explained": round(auc_explained, 4),
        "deletion_auc_random": round(auc_random, 4),
        "deletion_auc_random_std": round(float(np.std(random_aucs)), 4),
        "improvement": round(improvement, 4),
        "relative_improvement": round(relative, 4) if relative is not None else None,
        "insertion_auc": round(auc_insertion, 4),
        "n_features": len(order),
        "n_features_ablated": len(full_order),
        "curve": [round(float(v), 4) for v in curve],
        "insertion_curve": [round(float(v), 4) for v in ins_curve],
        "verdict": verdict,
    }


# ── Stability across refits ────────────────────────────────────────────────────

def explanation_stability(
    attribute_matrix: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Summarise how consistent attributions are across repeated model fits.

    Parameters
    ----------
    attribute_matrix : np.ndarray, shape (n_fits, n_features)
        Row ``i`` holds the attributions from fit ``i``.  Fits should differ
        only in their random seed (or in a bootstrap resample).
    feature_names : sequence of str
    top_k : int
        How many leading features to track.

    Returns
    -------
    dict with keys:
        n_fits
        mean_rank_correlation   mean pairwise Spearman correlation
        min_rank_correlation
        top_k_features          the features that led most often
        top_k_retention         share of fits in which a top feature stayed top
        per_fit_leader          which feature led in each fit
        verdict
    """
    attrs = np.asarray(attribute_matrix, dtype=np.float64)
    if attrs.ndim != 2 or attrs.shape[0] < 2:
        return {
            "n_fits": int(attrs.shape[0]) if attrs.ndim == 2 else 0,
            "mean_rank_correlation": None,
            "min_rank_correlation": None,
            "top_k_features": [],
            "top_k_retention": None,
            "per_fit_leader": [],
            "verdict": "Needs at least two fits to measure stability.",
        }

    n_fits = attrs.shape[0]
    corrs: list[float] = []
    for i in range(n_fits):
        for j in range(i + 1, n_fits):
            a, b = attrs[i], attrs[j]
            if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                # A constant row carries no ranking information.
                corrs.append(0.0)
                continue
            ra = np.argsort(np.argsort(-a))
            rb = np.argsort(np.argsort(-b))
            if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
                corrs.append(0.0)
                continue
            corr = float(np.corrcoef(ra, rb)[0, 1])
            corrs.append(corr if np.isfinite(corr) else 0.0)

    order = np.argsort(-np.abs(attrs), axis=1)
    leaders = [feature_names[int(idx[0])] for idx in order]

    top_k = max(1, min(top_k, attrs.shape[1]))
    top_sets = [set(int(i) for i in idx[:top_k]) for idx in order]
    # Retention must be measured against a FIXED reference set.  The previous
    # definition intersected each top-k set with the union of all top-k sets,
    # which is a superset of every one of them, so the ratio was identically
    # 1.0 for any input and reported perfect stability for pure noise.  Use the
    # first fit's top-k as the reference and measure mean overlap with it.
    reference_top = top_sets[0] if top_sets else set()
    retention = (
        float(np.mean([len(s & reference_top) / len(reference_top)
                       for s in top_sets]))
        if reference_top else 0.0
    )
    top_counts: dict[str, int] = {}
    for s in top_sets:
        for idx in s:
            name = feature_names[int(idx)]
            top_counts[name] = top_counts.get(name, 0) + 1
    top_features = [
        {"feature": name, "appeared_in_top_k": count, "share": round(count / n_fits, 4)}
        for name, count in sorted(
            top_counts.items(), key=lambda kv: kv[1], reverse=True
        )
    ]

    mean_corr = float(np.mean(corrs)) if corrs else None
    if mean_corr is None:
        verdict = "Could not measure stability."
    elif mean_corr >= 0.9:
        verdict = "Very stable. Repeated fits agree on the feature ranking."
    elif mean_corr >= 0.7:
        verdict = "Reasonably stable, with some reordering between fits."
    elif mean_corr >= 0.4:
        verdict = (
            "Unstable. Fits disagree substantially on which features matter, "
            "so a single explanation should not be acted on alone."
        )
    else:
        verdict = "Very unstable. The feature ranking is fit-dependent noise."

    return {
        "n_fits": n_fits,
        "mean_rank_correlation": round(mean_corr, 4) if mean_corr is not None else None,
        "min_rank_correlation": round(float(np.min(corrs)), 4) if corrs else None,
        "top_k_features": top_features,
        "top_k_retention": round(retention, 4),
        "per_fit_leader": leaders,
        "verdict": verdict,
    }


# ── Combined per-observation report ────────────────────────────────────────────

def validate_explanation(
    observation: np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    ranked_feature_indices: Sequence[int],
    baseline: np.ndarray | None = None,
    shap_additivity: dict[str, Any] | None = None,
    n_random: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Assemble every available check into one report for a single observation.

    Parameters
    ----------
    observation : np.ndarray, shape (n_features,)
    score_fn : callable
    ranked_feature_indices : sequence of int
        Feature indices in the explanation's ranking order.
    baseline : np.ndarray or None
    shap_additivity : dict or None
        Output of ``SHAPExplainer.check_additivity`` for this row, if SHAP ran.
    n_random, random_state

    Returns
    -------
    dict with keys: faithfulness, additivity, overall, checks_run
    """
    faith = faithfulness(
        observation,
        score_fn,
        ranked_feature_indices,
        baseline=baseline,
        n_random=n_random,
        random_state=random_state,
    )

    add = shap_additivity if shap_additivity is not None else {
        "checked": False,
        "passed": False,
        "note": "SHAP was not run for this observation.",
    }

    checks_run = 1 + (1 if add.get("checked") else 0)

    # A single overall verdict is only meaningful when both checks passed.
    faith_ok = (
        faith.get("relative_improvement") is not None
        and faith["relative_improvement"] <= -0.10
    )
    add_ok = bool(add.get("passed"))

    if checks_run == 2:
        if faith_ok and add_ok:
            overall = "VERIFIED"
            summary = (
                "Additivity holds and the named features demonstrably drive the "
                "score."
            )
        elif add_ok:
            overall = "PARTIALLY VERIFIED"
            summary = (
                "The attributions are internally consistent but their ranking "
                "is close to what random ordering would produce."
            )
        elif faith_ok:
            overall = "PARTIALLY VERIFIED"
            summary = (
                "The feature ranking is behaviourally faithful but the SHAP "
                "values do not sum to the displayed score."
            )
        else:
            overall = "UNVERIFIED"
            summary = (
                "This explanation failed the checks that were run. Treat the "
                "attribution as a hypothesis, not a finding."
            )
    else:
        overall = "PARTIALLY VERIFIED" if faith_ok else "UNVERIFIED"
        summary = (
            "Only behavioural faithfulness was checked; SHAP additivity was "
            "not available for this row."
        )

    return {
        "overall": overall,
        "summary": summary,
        "checks_run": checks_run,
        "faithfulness": faith,
        "additivity": add,
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _as_score(value: Any) -> float:
    """Coerce whatever score_fn returned into a single float."""
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan")
    if not np.isfinite(arr).any():
        return float("nan")
    return float(arr[np.isfinite(arr)][0])

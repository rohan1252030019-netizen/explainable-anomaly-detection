"""
src/evaluation/holdout.py

Honest out-of-sample evaluation for unsupervised anomaly detectors.

The problem this module exists to solve
---------------------------------------
Fitting a detector and scoring the SAME rows produces optimistically biased
metrics.  Every detector in this framework (and sklearn's unsupervised
estimators generally) defines "normal" relative to the batch it saw, so a row
that helped shape the model is scored by a model that already knows it.  On
the bundled benchmark that turns a genuinely hard problem into ROC-AUC 0.99
with precision 1.00, which is a measurement artefact rather than a result.

What this module does
---------------------
1. Splits the rows into a fit set and a held-out set (stratified on the label
   when labels exist, so both sides contain anomalies).
2. Fits the detector on the fit set ONLY.
3. Scores the held-out rows with the fitted detector.
4. Reports metrics for both splits side by side, plus the optimism gap.

The gap between the two is the number that matters: a large gap means the
in-sample headline figure is not trustworthy and should not be quoted alone.

IMPORTANT
---------
- Labels are used to stratify the split and to score the result.  They are
  never passed to fit().
- For Local Outlier Factor, ``novelty=True`` is required to score unseen rows.
  The detector classes already default to it; see the note below.
- If the held-out split happens to contain no anomalies, ranking metrics such
  as ROC-AUC are undefined.  This is reported explicitly rather than being
  silently replaced with 0.5.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.labels import to_binary_labels
from src.detection.scoring import ScoreNormalizer
from src.detection.thresholding import get_threshold_value
from src.evaluation.metrics import supervised_metrics


def _to_binary(labels: np.ndarray) -> np.ndarray | None:
    """
    Map common label conventions to {0 = normal, 1 = anomaly}, or None.

    Thin alias kept for the module's historical call sites; the rule itself
    lives in :mod:`src.data.labels` so it cannot drift between callers.
    """
    return to_binary_labels(labels)


def holdout_split(
    X: np.ndarray,
    y: np.ndarray | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """
    Split a feature matrix into fit and held-out subsets.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray or None
        Optional labels, used only to stratify the split so that both sides
        contain anomalies.  Never returned for model fitting.
    test_size : float
        Fraction of rows held out for evaluation.
    random_state : int

    Returns
    -------
    dict with keys X_fit, X_test, y_fit, y_test, fit_idx, test_idx.
    ``y_fit`` / ``y_test`` are None when ``y`` is None.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    indices = np.arange(n)

    y_bin = _to_binary(y) if y is not None else None

    # Stratify only when both classes are present and the minority class has
    # enough members to survive the split; otherwise sklearn raises.
    stratify = None
    if y_bin is not None:
        counts = np.bincount(y_bin, minlength=2)
        n_test = int(round(n * test_size))
        if counts.min() >= 2 and n_test >= counts.min() and (n - n_test) >= counts.min():
            stratify = y_bin

    fit_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True,
    )
    fit_idx = np.sort(fit_idx)
    test_idx = np.sort(test_idx)

    return {
        "X_fit": X[fit_idx],
        "X_test": X[test_idx],
        "y_fit": None if y_bin is None else y_bin[fit_idx],
        "y_test": None if y_bin is None else y_bin[test_idx],
        "fit_idx": fit_idx,
        "test_idx": test_idx,
    }


def split_raw_frame(
    df: pd.DataFrame,
    label_col: str | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Split a RAW DataFrame into fit and held-out frames before preprocessing.

    Why this exists
    ---------------
    Splitting an already-preprocessed matrix is not a leakage-free holdout.
    The scaler, imputer and one-hot encoder were fitted on every row, so the
    held-out rows contributed their means, standard deviations, medians and
    category sets to the transformation applied to the training rows.  The
    resulting metrics are optimistic by an amount that depends on the data and
    cannot be corrected after the fact.

    The correct order is: split rows -> fit the preprocessor on the fit frame
    -> transform the held-out frame with the fitted preprocessor -> fit the
    detector on the transformed fit frame -> score the transformed held-out
    frame.

    Returns
    -------
    dict with keys ``fit_df``, ``test_df``, ``y_fit``, ``y_test``,
    ``fit_idx``, ``test_idx``.  Labels follow ``{0 = normal, 1 = anomaly}``
    and are ``None`` when ``label_col`` is absent or not interpretable.
    """
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
    if df is None or len(df) == 0:
        raise ValueError("Cannot split an empty frame.")

    n = len(df)
    indices = np.arange(n)

    y_bin = None
    if label_col and label_col in df.columns:
        y_bin = to_binary_labels(df[label_col])

    stratify = None
    if y_bin is not None:
        counts = np.bincount(y_bin, minlength=2)
        n_test = int(round(n * float(test_size)))
        if (
            counts.min() >= 2
            and n_test >= counts.min()
            and (n - n_test) >= counts.min()
        ):
            stratify = y_bin

    fit_idx, test_idx = train_test_split(
        indices,
        test_size=float(test_size),
        random_state=random_state,
        stratify=stratify,
        shuffle=True,
    )
    fit_idx = np.sort(fit_idx)
    test_idx = np.sort(test_idx)

    return {
        "fit_df": df.iloc[fit_idx].reset_index(drop=True),
        "test_df": df.iloc[test_idx].reset_index(drop=True),
        "y_fit": None if y_bin is None else y_bin[fit_idx],
        "y_test": None if y_bin is None else y_bin[test_idx],
        "fit_idx": fit_idx,
        "test_idx": test_idx,
    }


def evaluate_holdout(
    detector_factory,
    X: np.ndarray,
    y: np.ndarray | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
    threshold_method: str = "contamination",
    contamination: float = 0.05,
    quantile: float = 0.95,
    fixed_threshold: float = 0.5,
    normalization: str = "quantile",
) -> dict[str, Any]:
    """
    Fit on a subset, evaluate on the held-out remainder, and report both.

    Parameters
    ----------
    detector_factory : callable
        Zero-argument callable returning a fresh, unfitted detector exposing
        ``fit(X)`` and ``score_samples(X)``.  A factory (rather than an
        instance) is required so the fit set never leaks into the test set.
    X : np.ndarray, shape (n_samples, n_features)
        Preprocessed feature matrix.
    y : np.ndarray or None
        Ground-truth labels used for scoring only.
    test_size, random_state : float, int
    threshold_method : {"contamination", "quantile", "fixed"}
    contamination, quantile, fixed_threshold
        Threshold parameters, applied identically to both splits.
    normalization : str
        Score normalisation method, see ``src.detection.scoring``.

    Returns
    -------
    dict with keys:
        available         whether a meaningful comparison could be made
        reason            why it was skipped, when unavailable
        test_size, random_state
        n_fit, n_test
        in_sample         metrics on the fit rows
        out_of_sample     metrics on the held-out rows
        optimism_gap      dict of (in_sample - out_of_sample) per metric
        score_mean_in_sample / score_mean_out_of_sample
        flag_rate_in_sample / flag_rate_out_of_sample / flag_rate_drift
        note              human-readable caveat
    """
    split = holdout_split(
        X, y=y, test_size=test_size, random_state=random_state
    )
    report = evaluate_holdout_split(
        detector_factory,
        split["X_fit"],
        split["X_test"],
        y_fit=split["y_fit"],
        y_test=split["y_test"],
        threshold_method=threshold_method,
        contamination=contamination,
        quantile=quantile,
        fixed_threshold=fixed_threshold,
        normalization=normalization,
    )
    report["test_size"] = test_size
    report["random_state"] = random_state
    return report


def evaluate_holdout_split(
    detector_factory,
    X_fit: np.ndarray,
    X_test: np.ndarray,
    y_fit: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    threshold_method: str = "contamination",
    contamination: float = 0.05,
    quantile: float = 0.95,
    fixed_threshold: float = 0.5,
    normalization: str = "quantile",
) -> dict[str, Any]:
    """
    Score an explicit, already-built fit/test split.

    This is the primitive :func:`evaluate_holdout` uses after splitting the
    feature matrix.  It is public because a leakage-free pipeline must split
    the RAW frame first and fit the preprocessor on the fit rows only; callers
    that need that (see :func:`split_raw_frame`) build both matrices and hand
    them here, so the scaler, imputer and encoder never see a held-out row.

    ``y_fit`` / ``y_test`` must follow ``{0 = normal, 1 = anomaly}``; pass
    ``None`` when no trustworthy labels exist.
    """
    X_fit = np.asarray(X_fit, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    detector = detector_factory()
    detector.fit(X_fit)

    raw_fit = np.asarray(detector.score_samples(X_fit), dtype=np.float64)
    raw_test = np.asarray(detector.score_samples(X_test), dtype=np.float64)

    # Fit the normaliser on the fit scores only, then apply the SAME anchors to
    # both splits.  Normalising each split independently re-anchors the mapping
    # on the held-out batch, which (a) is leakage and (b) makes the optimistic
    # gap meaningless: every split is rescaled to look the same by construction.
    normalizer = ScoreNormalizer(method=normalization).fit(raw_fit)
    norm_fit = normalizer.transform(raw_fit)
    norm_test = normalizer.transform(raw_test)

    # Derive the operating threshold from the FIT scores once, then apply that
    # same number to both splits.  Recomputing a contamination/quantile cut on
    # the held-out batch forces its flag rate to ~contamination by construction,
    # so "flag_rate_drift" was structurally ~0 and hid every real shift.
    threshold = get_threshold_value(
        norm_fit,
        method=threshold_method,
        contamination=contamination,
        quantile=quantile,
        fixed_threshold=fixed_threshold,
    )
    pred_fit = np.where(norm_fit >= threshold, -1, 1)
    pred_test = np.where(norm_test >= threshold, -1, 1)

    result: dict[str, Any] = {
        "normalization": normalization,
        "threshold_method": threshold_method,
        "threshold_value": round(float(threshold), 6),
        "n_fit": int(X_fit.shape[0]),
        "n_test": int(X_test.shape[0]),
        "available": False,
        "reason": None,
        "note": (
            "In-sample metrics are optimistic because the detector scored the "
            "same rows it was fitted on. Read the out-of-sample column."
        ),
    }

    # The two score vectors cover different rows, so they cannot be correlated
    # directly.  What is meaningful is whether the score distribution and the
    # resulting flag rate transfer from the fit rows to unseen rows.
    result["score_mean_in_sample"] = round(float(np.mean(raw_fit)), 6)
    result["score_mean_out_of_sample"] = round(float(np.mean(raw_test)), 6)
    result["score_std_in_sample"] = round(float(np.std(raw_fit)), 6)
    result["score_std_out_of_sample"] = round(float(np.std(raw_test)), 6)
    result["flag_rate_in_sample"] = round(float((pred_fit == -1).mean()), 4)
    result["flag_rate_out_of_sample"] = round(float((pred_test == -1).mean()), 4)
    result["flag_rate_drift"] = round(
        float((pred_test == -1).mean() - (pred_fit == -1).mean()), 4
    )
    result["score_normalization"] = normalizer.describe()

    if y_fit is None or y_test is None:
        result["reason"] = (
            "Ground-truth labels are required to score precision / recall / "
            "AUC. Detection ran fine; only the accuracy metrics are skipped."
        )
        return result

    y_fit = np.asarray(y_fit).astype(int)
    y_test = np.asarray(y_test).astype(int)

    if y_test.sum() == 0:
        result["reason"] = (
            "The held-out split contains no anomalies, so ROC-AUC and PR-AUC "
            "are undefined for it. Increase test_size or lower the anomaly rate."
        )
        return result

    result["in_sample"] = supervised_metrics(y_fit, pred_fit, norm_fit)
    result["out_of_sample"] = supervised_metrics(y_test, pred_test, norm_test)
    result["available"] = True

    gaps: dict[str, float | None] = {}
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        a = result["in_sample"].get(key)
        b = result["out_of_sample"].get(key)
        gaps[key] = (
            round(float(a) - float(b), 4)
            if isinstance(a, (int, float)) and isinstance(b, (int, float))
            else None
        )
    result["optimism_gap"] = gaps

    return result


def compare_holdout_across_models(
    model_factories: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run :func:`evaluate_holdout` for several detectors and tabulate the result.

    Parameters
    ----------
    model_factories : dict
        {model_name: zero-argument factory returning an unfitted detector}
    X : np.ndarray
    y : np.ndarray or None
    **kwargs
        Forwarded to :func:`evaluate_holdout`.

    Returns
    -------
    pd.DataFrame with one row per model and columns:
        Model, ROC-AUC (in-sample), ROC-AUC (out-of-sample), AUC gap,
        PR-AUC (in-sample), PR-AUC (out-of-sample), F1 (out-of-sample),
        Recall (out-of-sample), Precision (out-of-sample),
        Anomalies (out-of-sample)
    """
    rows: list[dict[str, Any]] = []
    for name, factory in model_factories.items():
        try:
            res = evaluate_holdout(factory, X, y=y, **kwargs)
        except Exception as exc:  # a model failing must not kill the table
            rows.append({"Model": name, "Error": str(exc)})
            continue

        if not res.get("available"):
            rows.append({"Model": name, "Available": False,
                         "Reason": res.get("reason")})
            continue

        ins = res["in_sample"]
        oos = res["out_of_sample"]
        gaps = res.get("optimism_gap", {})
        rows.append({
            "Model": name,
            "ROC-AUC (in-sample)": _round(ins.get("roc_auc")),
            "ROC-AUC (out-of-sample)": _round(oos.get("roc_auc")),
            "AUC gap": _round(gaps.get("roc_auc")),
            "PR-AUC (in-sample)": _round(ins.get("pr_auc")),
            "PR-AUC (out-of-sample)": _round(oos.get("pr_auc")),
            "F1 (out-of-sample)": _round(oos.get("f1")),
            "Recall (out-of-sample)": _round(oos.get("recall")),
            "Precision (out-of-sample)": _round(oos.get("precision")),
            "Anomalies (out-of-sample)": oos.get("n_detected_anomalies"),
            "n_test": res.get("n_test"),
        })

    return pd.DataFrame(rows)


def _round(value: Any, digits: int = 4) -> Any:
    return round(float(value), digits) if isinstance(value, (int, float)) else None

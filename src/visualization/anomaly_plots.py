"""
src/visualization/anomaly_plots.py

Anomaly-specific visualizations.

Charts produced:
- Anomaly score distribution (histogram + KDE + threshold line)
- Normal vs anomaly scatter (2 features at a time)
- Anomaly severity distribution (pie / donut)
- Feature contribution bar chart (local explanation)
- Top global feature importance bar chart
- Model comparison bar chart
- Score gauge for a single anomaly
- Confusion matrix heatmap
- ROC curve
- Precision-Recall curve
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE_NORMAL  = "#4A90D9"
PALETTE_ANOMALY = "#E74C3C"
PALETTE_WARN    = "#F39C12"
PALETTE_OK      = "#27AE60"
FIG_DPI = 100
TITLE_FONTSIZE  = 13
LABEL_FONTSIZE  = 10


def _save_or_return(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    return fig


# ── 1. Anomaly score distribution ─────────────────────────────────────────────

def plot_score_distribution(
    norm_scores: np.ndarray,
    predictions: np.ndarray,
    threshold: Optional[float] = None,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram of normalised anomaly scores, coloured by normal/anomaly.

    Parameters
    ----------
    norm_scores  : np.ndarray [0, 1]
    predictions  : np.ndarray  (+1 normal, -1 anomaly)
    threshold    : float or None  — vertical line showing the cut-off
    model_name   : str
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    normal_scores  = norm_scores[predictions ==  1]
    anomaly_scores = norm_scores[predictions == -1]

    bins = np.linspace(0, 1, 40)

    ax.hist(normal_scores,  bins=bins, color=PALETTE_NORMAL,  alpha=0.65,
            label=f"Normal  (n={len(normal_scores)})")
    ax.hist(anomaly_scores, bins=bins, color=PALETTE_ANOMALY, alpha=0.65,
            label=f"Anomaly (n={len(anomaly_scores)})")

    if threshold is not None:
        ax.axvline(threshold, color="black", lw=1.8, ls="--",
                   label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel("Normalised Anomaly Score  (0 = normal, 1 = anomalous)",
                  fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Count", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"Anomaly Score Distribution — {model_name}", fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 2. Severity distribution ──────────────────────────────────────────────────

def plot_severity_distribution(
    severity: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Donut chart of anomaly severity breakdown."""
    anomaly_severity = severity[predictions == -1]
    order    = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    colors   = [PALETTE_OK, PALETTE_WARN, "#E67E22", PALETTE_ANOMALY]

    counts = {s: int((anomaly_severity == s).sum()) for s in order}
    labels = [s for s in order if counts[s] > 0]
    sizes  = [counts[s] for s in labels]
    clrs   = [colors[order.index(s)] for s in labels]

    fig, ax = plt.subplots(figsize=(5, 5))
    if sum(sizes) == 0:
        ax.text(0.5, 0.5, "No anomalies detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
    else:
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=clrs,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"width": 0.55},
        )
        for at in autotexts:
            at.set_fontsize(10)
        ax.set_title("Anomaly Severity Distribution", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 3. Normal vs anomaly scatter ──────────────────────────────────────────────

def plot_normal_vs_anomaly_scatter(
    df: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of two features coloured by anomaly decision."""
    fig, ax = plt.subplots(figsize=(7, 5))

    normal_mask  = predictions ==  1
    anomaly_mask = predictions == -1

    ax.scatter(df.loc[normal_mask,  feature_x], df.loc[normal_mask,  feature_y],
               c=PALETTE_NORMAL,  alpha=0.5, s=20, label="Normal",  zorder=2)
    ax.scatter(df.loc[anomaly_mask, feature_x], df.loc[anomaly_mask, feature_y],
               c=PALETTE_ANOMALY, alpha=0.9, s=50, marker="X", label="Anomaly", zorder=3)

    ax.set_xlabel(feature_x, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(feature_y, fontsize=LABEL_FONTSIZE)
    ax.set_title(f"Normal vs Anomaly: {feature_x} vs {feature_y}", fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 4. Local feature contribution bar chart ───────────────────────────────────

def plot_local_feature_contributions(
    feature_table: pd.DataFrame,
    norm_score: float,
    severity: str,
    row_id: Any = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of top feature contributions for a single anomaly.

    Parameters
    ----------
    feature_table : pd.DataFrame  — from ExplanationGenerator._build_feature_table()
    norm_score    : float
    severity      : str
    row_id        : any
    """
    if feature_table.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No explanation available", ha="center", va="center")
        return _save_or_return(fig, save_path)

    # Map contribution label to numeric for bar length
    contrib_map = {"Very High": 1.0, "High": 0.7, "Medium": 0.4, "Low": 0.15}
    bar_vals = feature_table["Contribution"].map(contrib_map).fillna(0.1)
    features = feature_table["Feature"]
    deviations = feature_table["Deviation"]

    # Colour by direction
    colors = []
    for _, row in feature_table.iterrows():
        d = str(row.get("Direction", ""))
        if "above" in d or "active" in d:
            colors.append(PALETTE_ANOMALY)
        elif "below" in d or "inactive" in d:
            colors.append(PALETTE_NORMAL)
        else:
            colors.append(PALETTE_WARN)

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.45)))
    bars = ax.barh(
        features[::-1], bar_vals[::-1].values,
        color=colors[::-1], alpha=0.8, edgecolor="white"
    )
    # Annotate with deviation
    for bar, dev in zip(bars, deviations[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                str(dev), va="center", fontsize=8)

    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Relative Contribution", fontsize=LABEL_FONTSIZE)
    ax.set_title(
        f"Feature Contributions — Row {row_id}\n"
        f"Score: {norm_score:.3f} | Severity: {severity}",
        fontsize=TITLE_FONTSIZE,
    )

    # Legend
    above_patch = mpatches.Patch(color=PALETTE_ANOMALY, label="Above typical")
    below_patch = mpatches.Patch(color=PALETTE_NORMAL, label="Below typical")
    ax.legend(handles=[above_patch, below_patch], fontsize=8, loc="lower right")

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 5. Global feature importance ─────────────────────────────────────────────

def plot_global_feature_importance(
    global_importance_df: pd.DataFrame,
    top_k: int = 10,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of global feature importance across anomalies."""
    if global_importance_df is None or global_importance_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Global importance not computed", ha="center", va="center")
        return _save_or_return(fig, save_path)

    df = global_importance_df.head(top_k).copy()

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.45)))
    bars = ax.barh(
        df["feature"][::-1],
        df["mean_combined_score"][::-1],
        color=PALETTE_ANOMALY, alpha=0.75, edgecolor="white",
    )
    # Frequency annotation
    if "frequency" in df.columns:
        for bar, freq in zip(bars, df["frequency"][::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"n={freq}", va="center", fontsize=8)

    ax.set_xlabel("Mean Contribution Score", fontsize=LABEL_FONTSIZE)
    title = f"Global Feature Importance (Top {top_k})"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 6. Model comparison bar chart ────────────────────────────────────────────

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "F1",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing models on a given metric.
    Falls back to Score Sep. when metric values are all 'N/A'.
    """
    df = comparison_df.copy()

    # Try to use the requested metric; fall back to Score Sep.
    def _try_numeric(col):
        try:
            return pd.to_numeric(df[col], errors="coerce")
        except Exception:
            return pd.Series([np.nan] * len(df))

    values = _try_numeric(metric)
    if values.isna().all():
        metric = "Score Sep."
        values = _try_numeric(metric)

    valid = ~values.isna()
    models = df["Model"][valid].tolist()
    vals   = values[valid].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 4))
    colors = [PALETTE_NORMAL, PALETTE_ANOMALY, PALETTE_WARN, "#8E44AD"][:len(models)]
    bars = ax.bar(models, vals, color=colors, alpha=0.8, edgecolor="white")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric, fontsize=LABEL_FONTSIZE)
    ax.set_title(f"Model Comparison — {metric}", fontsize=TITLE_FONTSIZE)
    ax.set_ylim(0, max(vals) * 1.2 if vals else 1)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 7. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: List[List[int]],
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of a 2x2 confusion matrix."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax, cbar=False,
    )
    ax.set_xlabel("Predicted", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Actual", fontsize=LABEL_FONTSIZE)
    title = "Confusion Matrix"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 8. ROC curve ─────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    norm_scores: np.ndarray,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """ROC curve with AUC annotation."""
    from sklearn.metrics import roc_curve, roc_auc_score

    # Convert labels
    y_bin = (np.asarray(y_true) == 1).astype(int)
    if y_bin.sum() == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No positive labels", ha="center", va="center")
        return _save_or_return(fig, save_path)

    fpr, tpr, _ = roc_curve(y_bin, norm_scores)
    auc = roc_auc_score(y_bin, norm_scores)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=PALETTE_ANOMALY, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=LABEL_FONTSIZE)
    title = "ROC Curve"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 9. Precision-Recall curve ─────────────────────────────────────────────────

def plot_pr_curve(
    y_true: np.ndarray,
    norm_scores: np.ndarray,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Precision-Recall curve with AP annotation."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    y_bin = (np.asarray(y_true) == 1).astype(int)
    if y_bin.sum() == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No positive labels", ha="center", va="center")
        return _save_or_return(fig, save_path)

    precision, recall, _ = precision_recall_curve(y_bin, norm_scores)
    ap = average_precision_score(y_bin, norm_scores)
    baseline = y_bin.mean()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, color=PALETTE_ANOMALY, lw=2,
            label=f"PR Curve (AP = {ap:.3f})")
    ax.axhline(baseline, color="gray", lw=1, ls="--",
               label=f"Baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Precision", fontsize=LABEL_FONTSIZE)
    title = "Precision-Recall Curve"
    if model_name:
        title += f" — {model_name}"
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 10. Score gauge ───────────────────────────────────────────────────────────

def plot_score_gauge(
    norm_score: float,
    severity: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Semi-circular gauge chart showing a single anomaly score.
    Colour transitions from green (normal) to red (critical).
    """
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"aspect": "equal"})

    # Draw coloured arc segments
    segments = [
        (0.00, 0.50, PALETTE_OK,   "Low"),
        (0.50, 0.75, PALETTE_WARN, "Medium"),
        (0.75, 0.90, "#E67E22",    "High"),
        (0.90, 1.00, PALETTE_ANOMALY, "Critical"),
    ]
    theta_start, theta_end = 180, 0   # degrees, semi-circle

    for lo, hi, color, _ in segments:
        t0 = theta_start - lo  * (theta_start - theta_end)
        t1 = theta_start - hi  * (theta_start - theta_end)
        theta = np.linspace(np.radians(t0), np.radians(t1), 50)
        r_outer, r_inner = 1.0, 0.6
        xs = np.concatenate([r_outer * np.cos(theta),
                              r_inner * np.cos(theta[::-1])])
        ys = np.concatenate([r_outer * np.sin(theta),
                              r_inner * np.sin(theta[::-1])])
        ax.fill(xs, ys, color=color, alpha=0.7)

    # Needle
    angle = np.radians(theta_start - norm_score * (theta_start - theta_end))
    ax.annotate("", xy=(0.75 * np.cos(angle), 0.75 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.plot(0, 0, "ko", ms=6)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.15, 1.1)
    ax.axis("off")
    ax.text(0, -0.1, f"Score: {norm_score:.3f}  |  {severity}",
            ha="center", va="top", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return _save_or_return(fig, save_path)

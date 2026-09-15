"""
src/visualization/distributions.py

Distribution plots for EDA and preprocessing inspection.

Charts produced:
- Feature histograms (with KDE)
- Box plots per feature
- Correlation heatmap
- Missing value bar chart
- Categorical frequency bar charts
- Skewness analysis bar chart
- Feature variance bar chart
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit / headless use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns


# ── Style defaults ─────────────────────────────────────────────────────────────

PALETTE_NORMAL  = "#4A90D9"
PALETTE_ANOMALY = "#E74C3C"
PALETTE_NEUTRAL = "#7F8C8D"
FIG_DPI = 100
TITLE_FONTSIZE  = 13
LABEL_FONTSIZE  = 10


def _save_or_return(fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    return fig


# ── 1. Feature distributions ──────────────────────────────────────────────────

def plot_feature_distributions(
    df: pd.DataFrame,
    numerical_cols: List[str],
    max_cols: int = 6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Histogram + KDE for each numerical feature.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
    max_cols : int   — cap number of subplots
    save_path : str or None

    Returns
    -------
    matplotlib Figure
    """
    cols_to_plot = numerical_cols[:max_cols]
    n = len(cols_to_plot)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numerical features", ha="center", va="center")
        return _save_or_return(fig, save_path)

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, color=PALETTE_NORMAL, alpha=0.7, density=True, label="Histogram")
        try:
            data.plot.kde(ax=ax, color="#1A5276", lw=2, label="KDE")
        except Exception:
            pass
        ax.set_title(col, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Value", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=TITLE_FONTSIZE + 2, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 2. Box plots ───────────────────────────────────────────────────────────────

def plot_boxplots(
    df: pd.DataFrame,
    numerical_cols: List[str],
    label_col: Optional[str] = None,
    max_cols: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Box plots optionally split by label (normal / anomaly).

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
    label_col : str or None   — if provided, hue by this column
    max_cols : int
    save_path : str or None

    Returns
    -------
    matplotlib Figure
    """
    cols = numerical_cols[:max_cols]
    n = len(cols)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numerical features", ha="center", va="center")
        return _save_or_return(fig, save_path)

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        if label_col and label_col in df.columns:
            groups = df.groupby(label_col)[col].apply(list)
            labels_list = [str(k) for k in groups.index]
            import matplotlib
            _mpl_ver = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
            _labels_kw = "tick_labels" if _mpl_ver >= (3, 9) else "labels"
            ax.boxplot(
                [g for g in groups],
                patch_artist=True,
                boxprops=dict(facecolor=PALETTE_NORMAL, alpha=0.6),
                medianprops=dict(color="black", lw=2),
                **{_labels_kw: labels_list},
            )
        else:
            ax.boxplot(
                df[col].dropna(),
                patch_artist=True,
                boxprops=dict(facecolor=PALETTE_NORMAL, alpha=0.6),
                medianprops=dict(color="black", lw=2),
            )
        ax.set_title(col, fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("Value", fontsize=LABEL_FONTSIZE)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Box Plots", fontsize=TITLE_FONTSIZE + 2, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 3. Correlation heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    numerical_cols: List[str],
    max_features: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Pearson correlation heatmap for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
    max_features : int   — cap to keep plot readable
    save_path : str or None

    Returns
    -------
    matplotlib Figure
    """
    cols = numerical_cols[:max_features]
    if len(cols) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need at least 2 numerical features", ha="center", va="center")
        return _save_or_return(fig, save_path)

    corr = df[cols].corr()
    size = max(6, len(cols) * 0.7)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        ax=ax,
        annot=len(cols) <= 12,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap (Pearson)", fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 4. Missing values bar chart ────────────────────────────────────────────────

def plot_missing_values(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of missing value percentages per column."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    fig, ax = plt.subplots(figsize=(max(6, len(missing_pct) * 0.6), 4))

    if len(missing_pct) == 0:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
    else:
        bars = ax.barh(missing_pct.index, missing_pct.values, color=PALETTE_ANOMALY, alpha=0.75)
        ax.set_xlabel("Missing (%)", fontsize=LABEL_FONTSIZE)
        ax.set_title("Missing Values per Feature", fontsize=TITLE_FONTSIZE)
        for bar, val in zip(bars, missing_pct.values):
            ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)

    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 5. Categorical frequency plots ────────────────────────────────────────────

def plot_categorical_frequencies(
    df: pd.DataFrame,
    categorical_cols: List[str],
    max_cols: int = 4,
    top_k: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of value counts for categorical features."""
    cols = categorical_cols[:max_cols]
    n = len(cols)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No categorical features", ha="center", va="center")
        return _save_or_return(fig, save_path)

    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        vc = df[col].value_counts().head(top_k)
        ax.barh(vc.index.astype(str), vc.values, color=PALETTE_NEUTRAL, alpha=0.8)
        ax.set_title(col, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Count", fontsize=LABEL_FONTSIZE)
        ax.invert_yaxis()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Feature Frequencies", fontsize=TITLE_FONTSIZE + 2, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── 6. Skewness bar chart ─────────────────────────────────────────────────────

def plot_skewness(
    df: pd.DataFrame,
    numerical_cols: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of feature skewness values."""
    skew = df[numerical_cols].skew().sort_values()
    colors = [PALETTE_ANOMALY if abs(v) > 1 else PALETTE_NORMAL for v in skew]

    fig, ax = plt.subplots(figsize=(7, max(4, len(skew) * 0.35)))
    bars = ax.barh(skew.index, skew.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(1, color=PALETTE_ANOMALY, lw=0.8, ls=":", alpha=0.6, label="|skew|>1 (high)")
    ax.axvline(-1, color=PALETTE_ANOMALY, lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Skewness", fontsize=LABEL_FONTSIZE)
    ax.set_title("Feature Skewness\n(|skew|>1 flagged in red)", fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_or_return(fig, save_path)

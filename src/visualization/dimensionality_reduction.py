"""
src/visualization/dimensionality_reduction.py

PCA and t-SNE visualizations for anomaly exploration.

PURPOSE
-------
Dimensionality reduction is used ONLY for visualization.
PCA / t-SNE do NOT perform anomaly detection.
The anomaly labels shown come from the unsupervised detector,
or optionally from ground-truth labels for evaluation.

IMPORTANT:
- t-SNE is non-deterministic and computationally expensive.
  Use only on subsampled data for academic demonstrations.
- PCA is deterministic and fast; preferred for dashboard use.
"""


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

PALETTE_NORMAL  = "#4A90D9"
PALETTE_ANOMALY = "#E74C3C"
PALETTE_GT_NORM = "#AAD4F5"
PALETTE_GT_ANOM = "#F5A9A9"
FIG_DPI = 100
TITLE_FONTSIZE  = 13
LABEL_FONTSIZE  = 10


def _save_or_return(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    return fig


# ── PCA reduction ─────────────────────────────────────────────────────────────

def reduce_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """
    Reduce X to ``n_components`` dimensions using PCA.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    n_components : int
    random_state : int

    Returns
    -------
    X_reduced : np.ndarray, shape (n_samples, n_components)
    pca : fitted PCA object (for variance explained)
    """
    n_comp = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


# ── t-SNE reduction ───────────────────────────────────────────────────────────

def reduce_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_state: int = 42,
    max_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce X to ``n_components`` dimensions using t-SNE.

    Subsamples to ``max_samples`` to keep computation tractable.

    Parameters
    ----------
    X : np.ndarray
    n_components : int
    perplexity : float
    max_iter : int
    random_state : int
    max_samples : int

    Returns
    -------
    X_reduced : np.ndarray, shape (min(n_samples, max_samples), n_components)
    idx : np.ndarray
        The row indices of the subsample, so the caller can align labels.
    """
    from sklearn.manifold import TSNE

    n = min(max_samples, X.shape[0])
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    X_sub = X[idx]

    perp = min(perplexity, max(2, n // 3))
    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        max_iter=max_iter,
        random_state=random_state,
    )
    X_reduced = tsne.fit_transform(X_sub)
    return X_reduced, idx


# ── PCA 2D plot ───────────────────────────────────────────────────────────────

def plot_pca_anomalies(
    X: np.ndarray,
    predictions: np.ndarray,
    norm_scores: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    model_name: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    2D PCA scatter plot coloured by model anomaly predictions.

    Optionally shows ground-truth labels in a side-by-side panel
    for evaluation purposes.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    predictions : np.ndarray  (+1 normal, -1 anomaly)
    norm_scores : np.ndarray or None  — used for marker size
    y_true : np.ndarray or None  — 0/1 ground-truth labels
    feature_names : list of str or None
    model_name : str
    save_path : str or None

    Returns
    -------
    matplotlib Figure
    """
    X_2d, pca = reduce_pca(X, n_components=2)
    var_exp = pca.explained_variance_ratio_

    has_gt = (y_true is not None)
    ncols = 2 if has_gt else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # ── Panel 1: Model predictions ────────────────────────────────────────────
    ax = axes[0]
    normal_mask  = predictions ==  1
    anomaly_mask = predictions == -1

    # Size proportional to anomaly score
    if norm_scores is not None:
        sizes_anomaly = 30 + 80 * norm_scores[anomaly_mask]
    else:
        sizes_anomaly = 60

    ax.scatter(X_2d[normal_mask, 0],  X_2d[normal_mask, 1],
               c=PALETTE_NORMAL, alpha=0.4, s=15, label="Normal", zorder=2)
    ax.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1],
               c=PALETTE_ANOMALY, alpha=0.9, s=sizes_anomaly,
               marker="X", label="Anomaly", zorder=3)

    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)" if len(var_exp) > 1 else "PC2",
                  fontsize=LABEL_FONTSIZE)
    title = "PCA — Model Predictions"
    if model_name:
        title += f" ({model_name})"
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=9)

    # ── Panel 2: Ground-truth labels (if available) ────────────────────────────
    if has_gt:
        ax2 = axes[1]
        y_bin = (np.asarray(y_true) == 1).astype(bool)

        ax2.scatter(X_2d[~y_bin, 0], X_2d[~y_bin, 1],
                    c=PALETTE_NORMAL, alpha=0.4, s=15,
                    label="Normal (GT)", zorder=2)
        ax2.scatter(X_2d[y_bin, 0],  X_2d[y_bin, 1],
                    c=PALETTE_ANOMALY, alpha=0.9, s=60,
                    marker="X", label="Anomaly (GT)", zorder=3)

        ax2.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)", fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)" if len(var_exp) > 1 else "PC2",
                       fontsize=LABEL_FONTSIZE)
        ax2.set_title("PCA — Ground-Truth Labels\n(for evaluation only — NOT used in training)",
                      fontsize=TITLE_FONTSIZE)
        ax2.legend(fontsize=9)

    # Variance explained summary
    total_var = sum(var_exp[:2]) * 100
    fig.suptitle(
        f"PCA 2D Projection  |  PC1+PC2 explain {total_var:.1f}% of variance",
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── PCA explained variance ────────────────────────────────────────────────────

def plot_pca_explained_variance(
    X: np.ndarray,
    max_components: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """Scree plot of PCA explained variance."""
    n_comp = min(max_components, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)
    var = pca.explained_variance_ratio_
    cum_var = np.cumsum(var)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(var) + 1), var * 100, color=PALETTE_NORMAL,
           alpha=0.7, label="Individual")
    ax.plot(range(1, len(cum_var) + 1), cum_var * 100, "ro-",
            ms=5, lw=1.5, label="Cumulative")
    ax.axhline(90, color="gray", ls="--", lw=1, alpha=0.6, label="90% threshold")
    ax.set_xlabel("Principal Component", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Explained Variance (%)", fontsize=LABEL_FONTSIZE)
    ax.set_title("PCA Explained Variance (Scree Plot)", fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ── t-SNE plot ────────────────────────────────────────────────────────────────

def plot_tsne_anomalies(
    X: np.ndarray,
    predictions: np.ndarray,
    y_true: np.ndarray | None = None,
    model_name: str = "",
    random_state: int = 42,
    max_samples: int = 500,
    save_path: str | None = None,
) -> plt.Figure:
    """
    2D t-SNE scatter plot coloured by model anomaly predictions.

    NOTE: t-SNE is non-deterministic (random_state fixes it here) and
    computationally expensive.  Results should be interpreted qualitatively,
    not as ground truth.
    """
    X_2d, idx = reduce_tsne(
        X, n_components=2,
        random_state=random_state,
        max_samples=max_samples,
    )
    pred_sub = predictions[idx]

    has_gt = (y_true is not None)
    ncols = 2 if has_gt else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # Panel 1: model predictions
    ax = axes[0]
    nm = pred_sub ==  1
    am = pred_sub == -1
    ax.scatter(X_2d[nm, 0], X_2d[nm, 1], c=PALETTE_NORMAL, alpha=0.4, s=15, label="Normal")
    ax.scatter(X_2d[am, 0], X_2d[am, 1], c=PALETTE_ANOMALY, alpha=0.9, s=60, marker="X", label="Anomaly")
    title = "t-SNE — Model Predictions"
    if model_name:
        title += f" ({model_name})"
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("t-SNE 1", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("t-SNE 2", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=9)

    # Panel 2: ground-truth
    if has_gt:
        ax2 = axes[1]
        y_sub = np.asarray(y_true)[idx]
        y_bin = (y_sub == 1).astype(bool)
        ax2.scatter(X_2d[~y_bin, 0], X_2d[~y_bin, 1], c=PALETTE_NORMAL, alpha=0.4, s=15, label="Normal (GT)")
        ax2.scatter(X_2d[y_bin, 0],  X_2d[y_bin, 1],  c=PALETTE_ANOMALY, alpha=0.9, s=60, marker="X", label="Anomaly (GT)")
        ax2.set_title("t-SNE — Ground-Truth Labels\n(evaluation only)", fontsize=TITLE_FONTSIZE)
        ax2.set_xlabel("t-SNE 1", fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel("t-SNE 2", fontsize=LABEL_FONTSIZE)
        ax2.legend(fontsize=9)

    n_shown = len(idx)
    fig.suptitle(
        f"t-SNE 2D Projection  |  "
        f"n={n_shown} (subsample)  |  random_state={random_state}",
        fontsize=10, y=1.02
    )
    fig.tight_layout()
    return _save_or_return(fig, save_path)

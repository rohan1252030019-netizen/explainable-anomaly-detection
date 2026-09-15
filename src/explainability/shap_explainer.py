"""
src/explainability/shap_explainer.py

SHAP-based explanations for anomaly detectors.

SHAP (SHapley Additive exPlanations) decomposes a model's prediction into
additive feature contributions.  For anomaly detectors, we explain the
normalised anomaly score.

Applicability notes
-------------------
SHAP works best with tree-based models (TreeExplainer — exact, fast) and
can be applied to any model via KernelExplainer (slower, approximate).

- Isolation Forest  : TreeExplainer (exact, recommended)
- LOF               : KernelExplainer (approximate; slow for large datasets)
- One-Class SVM     : KernelExplainer (approximate)
- Elliptic Envelope : KernelExplainer (approximate)

For large datasets, we subsample a background dataset for KernelExplainer
to keep computation tractable on a laptop.

IMPORTANT:
- SHAP values explain the *model's score function*, not a causal mechanism.
- Language: "SHAP contribution", "feature attribution", NOT "caused by".
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Lazy-import SHAP so the rest of the framework works even if SHAP has issues
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ── Background dataset sampler ─────────────────────────────────────────────────

def _get_background(
    X_train: np.ndarray,
    max_background: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """Sample a background dataset for KernelExplainer."""
    rng = np.random.default_rng(random_state)
    n = min(max_background, X_train.shape[0])
    idx = rng.choice(X_train.shape[0], size=n, replace=False)
    return X_train[idx]


# ── SHAP wrapper ───────────────────────────────────────────────────────────────

class SHAPExplainer:
    """
    Wraps SHAP explainers for anomaly detection models.

    Parameters
    ----------
    model_name : str
    sklearn_model : fitted sklearn model object
    X_train : np.ndarray
        Training data (background for KernelExplainer).
    feature_names : list of str
    score_fn : callable or None
        A function (X) -> array of normalised scores.
        Required for KernelExplainer; not needed for TreeExplainer.
    max_background : int
        Max background samples for KernelExplainer.
    """

    def __init__(
        self,
        model_name: str,
        sklearn_model: Any,
        X_train: np.ndarray,
        feature_names: List[str],
        score_fn: Optional[Callable] = None,
        max_background: int = 100,
    ):
        self.model_name = model_name
        self.sklearn_model = sklearn_model
        self.X_train = X_train
        self.feature_names = feature_names
        self.score_fn = score_fn
        self.max_background = max_background

        self._explainer = None
        self._explainer_type: Optional[str] = None
        self._init_error: Optional[str] = None

        if SHAP_AVAILABLE:
            self._build_explainer()

    def _build_explainer(self) -> None:
        """Try to build the most appropriate explainer."""
        try:
            if self.model_name == "Isolation Forest":
                self._explainer = shap.TreeExplainer(self.sklearn_model)
                self._explainer_type = "TreeExplainer"

            else:
                # KernelExplainer requires a prediction function
                if self.score_fn is None:
                    self._init_error = (
                        f"KernelExplainer for {self.model_name} requires "
                        "score_fn to be provided."
                    )
                    return

                background = _get_background(
                    self.X_train, self.max_background
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._explainer = shap.KernelExplainer(
                        self.score_fn,
                        background,
                    )
                self._explainer_type = "KernelExplainer"

        except Exception as exc:
            self._init_error = f"Could not build SHAP explainer: {exc}"

    def is_available(self) -> bool:
        return SHAP_AVAILABLE and self._explainer is not None

    def explain(
        self,
        X: np.ndarray,
        max_samples: int = 50,
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for the given observations.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        max_samples : int
            Max observations to explain (KernelExplainer is slow).

        Returns
        -------
        shap_values : np.ndarray, shape (n_samples, n_features) or None
        """
        if not self.is_available():
            return None

        X = X[:max_samples]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self._explainer_type == "TreeExplainer":
                    sv = self._explainer.shap_values(X)
                    # IsolationForest TreeExplainer may return list or array
                    if isinstance(sv, list):
                        sv = sv[0]
                else:
                    sv = self._explainer.shap_values(X, silent=True)

            return np.array(sv)

        except Exception as exc:
            self._init_error = f"SHAP explain failed: {exc}"
            return None

    def explain_single(
        self,
        observation: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        """
        Explain a single observation.

        Parameters
        ----------
        observation : np.ndarray, shape (n_features,)

        Returns
        -------
        dict mapping feature_name -> SHAP value, or None on failure.
        """
        sv = self.explain(observation.reshape(1, -1), max_samples=1)
        if sv is None:
            return None

        sv_1d = sv[0] if sv.ndim == 2 else sv
        return {
            feat: round(float(v), 6)
            for feat, v in zip(self.feature_names, sv_1d)
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "shap_available": SHAP_AVAILABLE,
            "explainer_type": self._explainer_type,
            "is_available": self.is_available(),
            "error": self._init_error,
        }

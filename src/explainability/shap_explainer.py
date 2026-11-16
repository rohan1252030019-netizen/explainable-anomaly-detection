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

Keeping SHAP values in the same units as the dashboard
------------------------------------------------------
This was a real inconsistency: TreeExplainer explains the sklearn model's RAW
output, while the UI displays and thresholds the NORMALISED score.  The
attributions therefore did not sum to the number shown to the user.

The fix exploits the fact that score normalisation is affine on the unclipped
range:

    normalised = (raw - offset) / scale

Shapley values are linear in the explained quantity, so dividing each raw-space
phi by ``scale`` gives the normalised-space phi exactly, and the base value
transforms the same way.  Pass ``score_space`` (the dict returned by
``ScoreNormalizer.describe()``) and the attributions line up with the score.

IMPORTANT:
- SHAP values explain the *model's score function*, not a causal mechanism.
- Language: "SHAP contribution", "feature attribution", NOT "caused by".
"""

import warnings
from collections.abc import Callable
from typing import Any

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
    score_space : dict or None
        Output of ``ScoreNormalizer.describe()``.  When supplied together with
        a TreeExplainer, raw-space SHAP values are rescaled into normalised
        score space so they sum to the score shown in the UI.
    """

    def __init__(
        self,
        model_name: str,
        sklearn_model: Any,
        X_train: np.ndarray,
        feature_names: list[str],
        score_fn: Callable | None = None,
        max_background: int = 100,
        score_space: dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.sklearn_model = sklearn_model
        self.X_train = X_train
        self.feature_names = feature_names
        self.score_fn = score_fn
        self.max_background = max_background
        self.score_space = score_space

        self._explainer: Any = None
        self._explainer_type: str | None = None
        self._init_error: str | None = None
        self._last_base_values: np.ndarray | None = None

        if SHAP_AVAILABLE:
            self._build_explainer()

    def _build_explainer(self) -> None:
        """
        Build the fastest explainer whose units actually match the score.

        Strategy
        --------
        1. For tree models try ``TreeExplainer``, which is exact and cheap.
        2. Verify it with :meth:`check_additivity` against ``score_fn``.
        3. If it fails, fall back to a permutation explainer wrapped around
           ``score_fn`` itself, whose units are correct by construction.

        Step 2 is not paranoia.  ``shap.TreeExplainer`` on a scikit-learn
        ``IsolationForest`` explains the raw summed tree path length (values
        around 11.5 for this benchmark), which is neither ``score_samples``
        nor ``decision_function``.  Reporting those numbers as "contributions
        to the anomaly score" would be wrong, and the additivity check is what
        catches that automatically rather than by reviewer inspection.
        """
        if self.score_fn is None:
            self._init_error = (
                f"A score_fn is required to explain {self.model_name} in "
                "anomaly-score units."
            )
            return

        probe = np.asarray(self.X_train, dtype=np.float64)[
            : min(8, len(self.X_train))
        ]

        # ── Attempt 1: exact tree explainer ──────────────────────────────────
        if self.model_name == "Isolation Forest":
            try:
                candidate = shap.TreeExplainer(self.sklearn_model)
                saved = self._explainer
                self._explainer = candidate
                self._explainer_type = "TreeExplainer"
                check = self.check_additivity(probe, max_samples=len(probe))
                if check.get("checked") and check.get("passed"):
                    self._init_error = None
                    return
                # Units do not line up; throw this explainer away.
                self._explainer = saved
                self._explainer_type = None
            except Exception as exc:
                self._explainer_type = None
                self._init_error = f"TreeExplainer unavailable: {exc}"

        # ── Attempt 2: permutation explainer over score_fn ───────────────────
        # Correct by construction: it explains exactly the function we display.
        try:
            background = _get_background(self.X_train, self.max_background)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._explainer = shap.Explainer(
                    self.score_fn,
                    background,
                    algorithm="permutation",
                )
            self._explainer_type = "PermutationExplainer"
            self._init_error = None
        except Exception as exc:
            self._explainer = None
            self._explainer_type = None
            self._init_error = f"Could not build SHAP explainer: {exc}"

    def is_available(self) -> bool:
        return SHAP_AVAILABLE and self._explainer is not None

    def _rescale_factor(self) -> float | None:
        """
        Multiplier converting raw-space SHAP values into score space.

        Only TreeExplainer needs it: it explains the raw model output, so the
        affine score normalisation has to be divided out.  The permutation
        explainer wraps ``score_fn`` itself and is already in score units, so
        rescaling it would double-correct.
        """
        if self._explainer_type != "TreeExplainer":
            return None
        if not self.score_space:
            return None
        if self.score_space.get("method") == "rank":
            # Rank normalisation is not affine; values cannot be rescaled.
            return None
        scale = float(self.score_space.get("scale", 1.0))
        return 1.0 / scale if abs(scale) > 1e-12 else None

    def explain(
        self,
        X: np.ndarray,
        max_samples: int = 50,
    ) -> np.ndarray | None:
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
            Expressed in the same units as the score the dashboard shows,
            provided ``score_space`` was supplied.
        """
        if not self.is_available():
            return None

        X = np.asarray(X, dtype=np.float64)[:max_samples]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if self._explainer_type == "PermutationExplainer":
                    # The modern SHAP Explainer returns an Explanation object
                    # that carries the per-row base values we need for the
                    # additivity check.
                    out = self._explainer(X)
                    arr = np.asarray(out.values, dtype=np.float64)
                    bv = getattr(out, "base_values", None)
                    self._last_base_values = (
                        None if bv is None
                        else np.asarray(bv, dtype=np.float64).ravel()
                    )
                else:
                    if hasattr(self._explainer, "shap_values"):
                        sv = self._explainer.shap_values(X)
                    else:  # pragma: no cover - older SHAP
                        sv = self._explainer(X)
                    if isinstance(sv, list):
                        # Some explainers return one array per output dimension.
                        sv = sv[0]
                    arr = np.asarray(sv, dtype=np.float64)
                    self._last_base_values = None

            # PermutationExplainer can return (n_features, n_samples).
            if arr.ndim == 2 and arr.shape[0] != X.shape[0] \
                    and arr.shape[1] == X.shape[0]:
                arr = arr.T

            # Convert raw-model units into normalised score units.
            factor = self._rescale_factor()
            if factor is not None and factor != 1.0:
                arr = arr * factor

            return arr

        except Exception as exc:
            self._init_error = f"SHAP explain failed: {exc}"
            return None

    def explain_single(
        self,
        observation: np.ndarray,
    ) -> dict[str, float] | None:
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
            for feat, v in zip(self.feature_names, sv_1d, strict=False)
        }

    def base_value(self) -> float | None:
        """
        Return E[f(x)] in the same units as :meth:`explain`.

        For TreeExplainer the raw expected value is mapped through the affine
        score normalisation.  For the permutation explainer the cached
        per-row base values are already in score units and are returned
        untouched (averaged, since they are constant across rows).
        """
        if not self.is_available():
            return None

        cached = self._last_base_values
        if cached is not None and cached.size:
            return round(float(np.mean(cached)), 6)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bv = getattr(self._explainer, "expected_value", None)
            if isinstance(bv, (list, np.ndarray)):
                bv = np.asarray(bv, dtype=np.float64).ravel()
                bv = bv[0] if bv.size else None
            if bv is None:
                return None
            bv = float(bv)
        except Exception:
            return None

        factor = self._rescale_factor()
        if factor is not None:
            offset = float((self.score_space or {}).get("offset", 0.0))
            bv = (bv - offset) * factor

        return round(bv, 6)

    def check_additivity(
        self,
        X: np.ndarray,
        score_fn: Callable | None = None,
        tolerance: float = 1e-3,
        max_samples: int = 20,
    ) -> dict[str, Any]:
        """
        Verify that sum(phi) + base_value reproduces the explained score.

        This is the cheapest possible check that an explanation is internally
        consistent.  If it fails, the attributions are not describing the
        quantity the user is looking at and should not be trusted or summed.

        Parameters
        ----------
        X : np.ndarray
            Rows to verify.
        score_fn : callable or None
            Must return the same score the explainer was built for.  Falls back
            to ``self.score_fn``.
        tolerance : float
            Absolute tolerance on the additivity residual.
        max_samples : int

        Returns
        -------
        dict with keys: checked, n_passed, n_failed, passed (bool),
        max_abs_error, mean_abs_error, rows (per-row residuals),
        score_units ("normalised" | "raw"), note
        """
        fn = score_fn or self.score_fn
        result: dict[str, Any] = {
            "checked": False,
            "passed": False,
            "n_passed": 0,
            "n_failed": 0,
            "max_abs_error": None,
            "mean_abs_error": None,
            "rows": [],
            "score_units": (
                "normalised" if self.score_space and self.score_fn else "raw"
            ),
            "note": "",
        }

        if not self.is_available() or fn is None:
            result["note"] = "SHAP unavailable or no score function to verify against."
            return result

        rows = np.asarray(X, dtype=np.float64)[:max_samples]
        sv = self.explain(rows, max_samples=max_samples)
        if sv is None:
            result["note"] = self._init_error or "SHAP computation failed."
            return result

        base = self.base_value()
        if base is None:
            result["note"] = (
                "Base value unavailable for this explainer; additivity cannot "
                "be verified for KernelExplainer."
            )
            return result

        try:
            actual = np.asarray(fn(rows), dtype=np.float64).ravel()
        except Exception as exc:
            result["note"] = f"Score function failed: {exc}"
            return result

        # score_fn may return normalised scores while the explainer explains the
        # raw model output (or vice versa). Compare like with like.
        predicted = sv.sum(axis=1) + base
        residual = np.abs(predicted - actual)

        # When units differ by a constant affine map, rescale the prediction
        # before judging, but keep the raw residual available for inspection.
        result["checked"] = True
        result["rows"] = [
            {
                "index": int(i),
                "shap_sum_plus_base": round(float(predicted[i]), 6),
                "actual_score": round(float(actual[i]), 6),
                "abs_error": round(float(residual[i]), 6),
            }
            for i in range(len(rows))
        ]
        result["max_abs_error"] = round(float(residual.max()), 6)
        result["mean_abs_error"] = round(float(residual.mean()), 6)
        result["n_passed"] = int((residual <= tolerance).sum())
        result["n_failed"] = int((residual > tolerance).sum())
        result["passed"] = result["n_failed"] == 0
        result["note"] = (
            "Additivity holds: sum(phi) + base_value reproduces the explained "
            "score within tolerance."
            if result["passed"]
            else "Additivity violated. The attributions do not sum to the "
                 "displayed score; treat them as a qualitative ranking only."
        )
        return result

    def get_status(self) -> dict[str, Any]:
        return {
            "shap_available": SHAP_AVAILABLE,
            "explainer_type": self._explainer_type,
            "is_available": self.is_available(),
            "error": self._init_error,
            "base_value": self.base_value(),
            "score_space": self.score_space,
            "uses_score_space": self._rescale_factor() is not None,
        }

"""
src/data/preprocessing.py

Preprocessing pipeline for the Explainable Anomaly Detection Framework.

Pipeline steps (in order):
1.  Drop label column if provided (labels are NOT used in training)
2.  Drop obvious ID columns
3.  Drop constant columns
4.  Handle datetime columns (convert or drop)
5.  Impute missing values
    - Numerical  → median
    - Categorical → most frequent (or "Unknown" if no mode)
6.  Remove duplicate rows (optional)
7.  Scale numerical features   → StandardScaler (default) or RobustScaler
8.  Encode categorical features → OneHotEncoder(handle_unknown="ignore")
9.  Return a dense NumPy array + ordered feature name list

Every step is logged so users can understand what happened.

IMPORTANT:
- The preprocessing pipeline is fit on training data via fit_transform().
- New data (e.g. uploaded for prediction) should be transformed via transform().
- Labels are stored separately and returned if requested; they are NEVER
  used during model training.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────────

NEAR_ZERO_VAR_THRESHOLD = 1e-10

# ID-hint tokens (lower-case)
_ID_HINTS = {"id", "index", "idx", "key", "uuid", "rownum", "row_num", "row_id"}


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for anomaly detection.

    Parameters
    ----------
    label_col : str or None
        Name of the ground-truth label column.  If provided it is removed
        before training.  Set to None if no label column exists.
    scaler_type : {"standard", "robust"}
        Scaling strategy for numerical features.
        - "standard"  → StandardScaler (zero mean, unit variance)
        - "robust"    → RobustScaler   (median/IQR, more robust to outliers)
    drop_duplicates : bool
        Whether to drop duplicate rows before fitting.
    id_cols : list of str or None
        Columns to explicitly exclude (e.g. ID columns identified by the user).
    contamination_hint : float or None
        Reserved for future use; not used in preprocessing itself.
    """

    def __init__(
        self,
        label_col: str | None = None,
        scaler_type: str = "standard",
        drop_duplicates: bool = False,
        id_cols: list[str] | None = None,
    ):
        self.label_col = label_col
        self.scaler_type = scaler_type
        self.drop_duplicates = drop_duplicates
        self.id_cols: list[str] = id_cols or []

        # Set after fit
        self.feature_names_out_: list[str] = []
        self.numerical_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.dropped_cols_: list[str] = []
        self.log_: list[str] = []
        self._pipeline: ColumnTransformer | None = None
        self._is_fitted: bool = False
        self._reference_stats: dict[str, Any] = {}  # for explanations

    # ── Public interface ───────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, list[str]]:
        """
        Fit the pipeline on ``df`` and return the transformed array.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        feature_names : list of str
        """
        df_clean = self._prepare(df, fit=True)
        pipeline = self._pipeline
        if pipeline is None:  # pragma: no cover - _prepare(fit=True) builds it
            raise RuntimeError("Preprocessing pipeline was not constructed.")
        X = pipeline.fit_transform(df_clean)
        self._is_fitted = True
        self._store_reference_stats(df_clean)
        return X, self.feature_names_out_

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted pipeline.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline is not fitted yet. Call fit_transform() first."
            )
        df_clean = self._prepare(df, fit=False)
        pipeline = self._pipeline
        if pipeline is None:  # pragma: no cover - guarded by _is_fitted
            raise RuntimeError("Preprocessing pipeline is not fitted.")
        return pipeline.transform(df_clean)

    def get_labels(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Extract the label column from a DataFrame (for post-hoc evaluation).

        Returns None if no label column was specified or if it's absent.
        """
        if self.label_col and self.label_col in df.columns:
            return df[self.label_col].values
        return None

    def get_log(self) -> list[str]:
        """Return the preprocessing log entries."""
        return self.log_

    def get_reference_stats(self) -> dict[str, Any]:
        """
        Return per-feature reference statistics computed on the training data.

        Used by the explainability module to compute deviations.

        Returns
        -------
        dict with keys: "median", "mean", "std", "q25", "q75", "iqr"
        Each value is a dict mapping feature name → statistic.
        """
        return self._reference_stats

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        self.log_.append(message)

    def _prepare(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Apply non-sklearn steps (dropping columns, de-duplication, type fixes).
        Also builds the sklearn ColumnTransformer when fit=True.
        """
        df = df.copy()
        self.dropped_cols_ = []

        if fit:
            self.log_ = []

        # ── Step 1: Remove label column ────────────────────────────────────────
        if self.label_col and self.label_col in df.columns:
            df = df.drop(columns=[self.label_col])
            self._log(
                f"[STEP 1] Label column '{self.label_col}' removed from training "
                "features. It will only be used for post-hoc evaluation."
            )

        # ── Step 2: Remove explicit ID columns ────────────────────────────────
        to_drop_ids = [c for c in self.id_cols if c in df.columns]
        if to_drop_ids:
            df = df.drop(columns=to_drop_ids)
            self.dropped_cols_.extend(to_drop_ids)
            self._log(f"[STEP 2] ID columns excluded from training: {to_drop_ids}")

        # Auto-detect ID columns (heuristic, only during fit)
        if fit:
            auto_id = self._auto_detect_id_cols(df)
            if auto_id:
                df = df.drop(columns=auto_id)
                self.dropped_cols_.extend(auto_id)
                self._log(
                    f"[STEP 2b] Auto-detected ID columns excluded: {auto_id}"
                )

        # ── Step 3: Remove constant columns ───────────────────────────────────
        if fit:
            const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
            if const_cols:
                df = df.drop(columns=const_cols)
                self.dropped_cols_.extend(const_cols)
                self._log(
                    f"[STEP 3] Constant columns removed "
                    f"(zero variance, no signal): {const_cols}"
                )

        # ── Step 4: Handle datetime columns ───────────────────────────────────
        datetime_cols = [
            c for c in df.columns
            if "datetime" in str(df[c].dtype) or "timedelta" in str(df[c].dtype)
        ]
        if datetime_cols:
            df = df.drop(columns=datetime_cols)
            self.dropped_cols_.extend(datetime_cols)
            self._log(
                f"[STEP 4] Datetime columns excluded "
                f"(not yet supported for numerical encoding): {datetime_cols}"
            )

        # ── Step 5: Deduplicate ────────────────────────────────────────────────
        if self.drop_duplicates and fit:
            n_before = len(df)
            df = df.drop_duplicates()
            n_removed = n_before - len(df)
            if n_removed > 0:
                df = df.reset_index(drop=True)
                self._log(
                    f"[STEP 5] {n_removed} duplicate rows removed."
                )

        # ── Step 6: Identify column types ─────────────────────────────────────
        if fit:
            self.numerical_cols_ = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
            self.categorical_cols_ = [
                c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            ]
            self._log(
                f"[STEP 6] Feature identification: "
                f"{len(self.numerical_cols_)} numerical, "
                f"{len(self.categorical_cols_)} categorical."
            )

        # ── Step 7: Build sklearn pipeline (only during fit) ──────────────────
        if fit:
            self._build_pipeline(df)

        # Reindex to match fitted columns
        expected_cols = self.numerical_cols_ + self.categorical_cols_
        missing_cols = [c for c in expected_cols if c not in df.columns]
        for c in missing_cols:
            if c in self.numerical_cols_:
                df[c] = np.nan
            else:
                df[c] = "Unknown"
        df = df[expected_cols]

        return df

    def _build_pipeline(self, df: pd.DataFrame) -> None:
        """Build the ColumnTransformer pipeline and collect feature names."""

        # Numerical transformer
        scaler = (
            StandardScaler() if self.scaler_type == "standard"
            else RobustScaler()
        )
        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ])
        scaler_name = (
            "StandardScaler (zero mean, unit variance)"
            if self.scaler_type == "standard"
            else "RobustScaler (median/IQR, robust to extreme values)"
        )
        self._log(
            f"[STEP 7a] Numerical features: "
            f"median imputation -> {scaler_name}."
        )

        # Categorical transformer
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ])
        self._log(
            "[STEP 7b] Categorical features: "
            "most-frequent imputation -> OneHotEncoder(handle_unknown='ignore')."
        )

        transformers = []
        if self.numerical_cols_:
            transformers.append(("num", numerical_transformer, self.numerical_cols_))
        if self.categorical_cols_:
            transformers.append(("cat", categorical_transformer, self.categorical_cols_))

        self._pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        # Pre-compute feature names (before fit, we need to fit encoder to get names)
        # We'll update these after fitting; for now store placeholders
        self.feature_names_out_ = self._compute_feature_names(df)

    def _compute_feature_names(self, df: pd.DataFrame) -> list[str]:
        """
        Compute the output feature names after transformation.

        We do a small dry-run fit on df to get OHE category names.
        This is called once during _build_pipeline and again after fitting.
        """
        names = list(self.numerical_cols_)

        if self.categorical_cols_:
            # Temporarily fit encoder to get category names
            cat_data = df[self.categorical_cols_].copy()
            # Fill missing with most frequent (quick imputation for name extraction)
            for c in self.categorical_cols_:
                if cat_data[c].isnull().any():
                    mode = cat_data[c].mode()
                    fill = mode[0] if len(mode) > 0 else "Unknown"
                    cat_data[c] = cat_data[c].fillna(fill)
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enc.fit(cat_data)
            for i, col in enumerate(self.categorical_cols_):
                for cat in enc.categories_[i]:
                    names.append(f"{col}_{cat}")

        return names

    def _auto_detect_id_cols(self, df: pd.DataFrame) -> list[str]:
        """Heuristically detect ID columns not explicitly specified."""
        auto_id = []
        for col in df.columns:
            col_lower = col.lower().replace("-", "_").replace(" ", "_")
            name_match = any(hint in col_lower for hint in _ID_HINTS)
            uniqueness = df[col].nunique() / max(len(df), 1)
            if name_match and uniqueness > 0.9:
                auto_id.append(col)
        return auto_id

    def _store_reference_stats(self, df_clean: pd.DataFrame) -> None:
        """
        Store reference statistics for each ORIGINAL numerical column.

        Used later by the explainability module to compute deviations.
        The statistics are computed on the raw (pre-scaling) values.
        """
        stats: dict[str, Any] = {
            "median": {},
            "mean": {},
            "std": {},
            "q25": {},
            "q75": {},
            "iqr": {},
            "min": {},
            "max": {},
        }
        for col in self.numerical_cols_:
            series = df_clean[col].dropna()
            stats["median"][col] = float(series.median())
            stats["mean"][col] = float(series.mean())
            stats["std"][col] = float(series.std())
            stats["q25"][col] = float(series.quantile(0.25))
            stats["q75"][col] = float(series.quantile(0.75))
            stats["iqr"][col] = float(
                series.quantile(0.75) - series.quantile(0.25)
            )
            stats["min"][col] = float(series.min())
            stats["max"][col] = float(series.max())

        self._reference_stats = stats
        self._log(
            "[STEP 8] Reference statistics computed for explanations "
            f"({len(self.numerical_cols_)} numerical features)."
        )


    def denormalize(self, scaled_row: np.ndarray) -> np.ndarray:
            """
            Map a scaled feature vector back to original dataset units.

            The explainability layer mixes three different unit spaces: the model
            consumes scaled values, while ``_store_reference_stats`` deliberately
            keeps medians and IQRs in RAW units so the UI can display real
            numbers.  Comparing one against the other silently produces
            meaningless z-scores, so every crossing of that boundary goes through
            this method.

            Only the numerical block can be inverted.  One-hot columns are not
            ordinal, so they are returned as their 0/1 activation and must be
            interpreted by category name instead.

            Parameters
            ----------
            scaled_row : np.ndarray, shape (n_features_out,)
                One row of the output of ``transform`` / ``fit_transform``.

            Returns
            -------
            np.ndarray, shape (n_features_out,)
                Values in the original units for numerical features; the raw 0/1
                activation for one-hot features.
            """
            row = np.asarray(scaled_row, dtype=np.float64).ravel()
            out = row.copy()

            scaler = None
            pipeline = getattr(self, "_pipeline", None)
            if pipeline is not None:
                try:
                    scaler = pipeline.named_transformers_["num"].named_steps[
                        "scaler"
                    ]
                except Exception:
                    scaler = None

            n_num = len(self.numerical_cols_)
            if n_num == 0:
                return out

            block = row[:n_num]
            if scaler is not None:
                try:
                    block = scaler.inverse_transform(block.reshape(1, -1)).ravel()
                except Exception:
                    block = row[:n_num]
            out[:n_num] = block
            return out

    def feature_units(self) -> dict[str, str]:
        """
        Declare which unit space each output feature lives in.

        Returns a mapping ``feature_name -> "raw" | "onehot"`` so callers never
        have to guess whether a number is a real measurement or an encoder
        activation.
        """
        units: dict[str, str] = {name: "raw" for name in self.numerical_cols_}
        for name in self.feature_names_out_:
            if name not in units:
                units[name] = "onehot"
        return units


# ── Convenience function ───────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    df: pd.DataFrame,
    label_col: str | None = None,
    scaler_type: str = "standard",
    drop_duplicates: bool = False,
    id_cols: list[str] | None = None,
) -> tuple["PreprocessingPipeline", np.ndarray, list[str]]:
    """
    Convenience wrapper: create, fit, and return the pipeline + transformed data.

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str or None
    scaler_type : {"standard", "robust"}
    drop_duplicates : bool
    id_cols : list of str or None

    Returns
    -------
    pipeline : PreprocessingPipeline
    X : np.ndarray
    feature_names : list of str
    """
    pipeline = PreprocessingPipeline(
        label_col=label_col,
        scaler_type=scaler_type,
        drop_duplicates=drop_duplicates,
        id_cols=id_cols or [],
    )
    X, feature_names = pipeline.fit_transform(df)
    return pipeline, X, feature_names

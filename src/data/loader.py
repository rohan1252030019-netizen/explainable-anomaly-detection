"""
src/data/loader.py

Dataset loading utilities for the Explainable Anomaly Detection Framework.

Responsibilities:
- Load CSV datasets
- Identify column types (numerical, categorical)
- Detect ID columns, label columns, constant columns
- Compute basic dataset statistics
- Generate controlled synthetic datasets for demonstration

NOTE: Labels, if present, are identified here but are NEVER passed to
      unsupervised model training. They are stored separately for post-hoc
      evaluation only.
"""

import os
import io
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

# Heuristic: a column whose name contains any of these tokens is likely an ID.
_ID_HINTS = {"id", "index", "idx", "key", "uuid", "rownum", "row_num", "row_id"}

# Heuristic: a column whose name or unique values suggest it is a label.
_LABEL_HINTS = {
    "label", "target", "class", "anomaly", "fraud", "attack",
    "malicious", "normal", "is_anomaly", "is_fraud", "y",
}

# Cardinality threshold: categorical columns with more unique values than this
# are considered high-cardinality and a warning is raised.
HIGH_CARDINALITY_THRESHOLD = 50


# ── Dataset Summary dataclass (plain dict for simplicity) ──────────────────────

def _empty_summary() -> Dict[str, Any]:
    return {
        "n_rows": 0,
        "n_cols": 0,
        "numerical_cols": [],
        "categorical_cols": [],
        "total_missing": 0,
        "missing_by_col": {},
        "duplicate_rows": 0,
        "constant_cols": [],
        "potential_id_cols": [],
        "potential_label_col": None,
        "high_cardinality_cols": [],
        "dtypes": {},
        "memory_mb": 0.0,
    }


# ── Core loader ────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads a CSV dataset and produces a rich dataset summary.

    Parameters
    ----------
    filepath : str or None
        Path to a CSV file.  Either ``filepath`` or ``dataframe`` must be
        provided when calling ``load()``.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.summary: Dict[str, Any] = _empty_summary()
        self.filepath: Optional[str] = None

    # ── Public interface ───────────────────────────────────────────────────────

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load a CSV file from disk.

        Parameters
        ----------
        filepath : str

        Returns
        -------
        pd.DataFrame
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        self.filepath = filepath
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV: {exc}") from exc

        return self._process(df)

    def load_from_bytes(self, data: bytes, filename: str = "upload.csv") -> pd.DataFrame:
        """
        Load a CSV from raw bytes (Streamlit file uploader).

        Parameters
        ----------
        data : bytes
        filename : str

        Returns
        -------
        pd.DataFrame
        """
        try:
            df = pd.read_csv(io.BytesIO(data), low_memory=False)
        except Exception as exc:
            raise ValueError(f"Failed to parse CSV bytes: {exc}") from exc

        self.filepath = filename
        return self._process(df)

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accept a pre-loaded DataFrame and compute the summary.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        return self._process(df.copy())

    # ── Internal processing ────────────────────────────────────────────────────

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all inspection steps and populate ``self.summary``."""
        if df.empty:
            raise ValueError("The loaded dataset is empty.")

        if df.shape[0] < 10:
            raise ValueError(
                f"Dataset has only {df.shape[0]} rows. "
                "At least 10 rows are required."
            )

        self.df = df
        summary = _empty_summary()

        # Basic dimensions
        summary["n_rows"] = int(df.shape[0])
        summary["n_cols"] = int(df.shape[1])

        # Column types
        summary["numerical_cols"] = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        summary["categorical_cols"] = [
            c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
        ]

        # Missing values
        missing = df.isnull().sum()
        summary["missing_by_col"] = {
            col: int(cnt) for col, cnt in missing.items() if cnt > 0
        }
        summary["total_missing"] = int(missing.sum())

        # Duplicates
        summary["duplicate_rows"] = int(df.duplicated().sum())

        # Constant columns (zero variance / single unique value)
        summary["constant_cols"] = [
            c for c in df.columns if df[c].nunique(dropna=False) <= 1
        ]

        # Potential ID columns
        summary["potential_id_cols"] = self._detect_id_cols(df)

        # Potential label column
        summary["potential_label_col"] = self._detect_label_col(df)

        # High-cardinality categorical columns
        summary["high_cardinality_cols"] = [
            c for c in summary["categorical_cols"]
            if df[c].nunique() > HIGH_CARDINALITY_THRESHOLD
        ]

        # Data types
        summary["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Memory
        summary["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3)

        self.summary = summary
        return df

    @staticmethod
    def _detect_id_cols(df: pd.DataFrame) -> List[str]:
        """
        Heuristically identify columns that look like row identifiers.

        A column is flagged if:
        - Its name (lower-cased) matches or contains a known ID token, AND
        - Its values are nearly unique (>90% unique).
        """
        id_cols = []
        for col in df.columns:
            col_lower = col.lower().replace("-", "_").replace(" ", "_")
            name_match = any(hint in col_lower for hint in _ID_HINTS)
            uniqueness = df[col].nunique() / max(len(df), 1)
            if name_match and uniqueness > 0.9:
                id_cols.append(col)
        return id_cols

    @staticmethod
    def _detect_label_col(df: pd.DataFrame) -> Optional[str]:
        """
        Heuristically identify a potential ground-truth label column.

        Checks:
        1. Column name matches known label hints AND has low cardinality (≤ 10).
        2. Binary integer columns {0, 1} with a short generic name.

        Categorical columns are excluded from detection because a name match
        there is almost certainly a content column (e.g. 'transaction_type').
        """
        for col in df.columns:
            col_lower = col.lower().replace("-", "_").replace(" ", "_")
            n_unique = df[col].nunique(dropna=True)

            # Only consider numeric columns OR columns whose name is an exact
            # match to a label hint (not just "contains")
            is_numeric = pd.api.types.is_numeric_dtype(df[col])

            if is_numeric and n_unique <= 10:
                # Exact or suffix/prefix match on a label hint token
                if any(
                    col_lower == hint
                    or col_lower.startswith(hint + "_")
                    or col_lower.endswith("_" + hint)
                    for hint in _LABEL_HINTS
                ):
                    return col

        # Fallback: binary numeric column {0, 1} with a short name
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = set(df[col].dropna().unique())
                # Cast to int for comparison
                try:
                    unique_int = {int(v) for v in unique_vals}
                except (ValueError, TypeError):
                    continue
                if unique_int <= {0, 1}:
                    col_lower = col.lower()
                    if len(col_lower) <= 6:
                        return col

        return None

    def get_summary(self) -> Dict[str, Any]:
        """Return the dataset summary dictionary."""
        return self.summary

    def get_descriptive_stats(self) -> pd.DataFrame:
        """Return descriptive statistics for numerical columns."""
        if self.df is None:
            raise RuntimeError("No dataset loaded.")
        return self.df[self.summary["numerical_cols"]].describe().T

    def get_column_info(self) -> pd.DataFrame:
        """
        Return a per-column information table.

        Columns: feature, dtype, n_missing, pct_missing, n_unique, type_tag
        """
        if self.df is None:
            raise RuntimeError("No dataset loaded.")
        rows = []
        for col in self.df.columns:
            n_miss = int(self.df[col].isnull().sum())
            pct_miss = round(n_miss / len(self.df) * 100, 2)
            n_unique = int(self.df[col].nunique(dropna=False))
            if col in self.summary["potential_id_cols"]:
                tag = "ID"
            elif col == self.summary["potential_label_col"]:
                tag = "Label"
            elif col in self.summary["constant_cols"]:
                tag = "Constant"
            elif col in self.summary["categorical_cols"]:
                tag = "Categorical"
            else:
                tag = "Numerical"
            rows.append({
                "Feature": col,
                "DType": str(self.df[col].dtype),
                "Missing": n_miss,
                "Missing %": pct_miss,
                "Unique Values": n_unique,
                "Type": tag,
            })
        return pd.DataFrame(rows)


# ── Synthetic dataset generator ────────────────────────────────────────────────

def generate_sample_dataset(
    n_normal: int = 800,
    n_anomalies: int = 80,
    n_features: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic tabular dataset with clearly labelled anomalies.

    Anomaly types included:
    - Global outliers  : extreme values far from normal cluster
    - Local outliers   : points in sparse region near normal boundary
    - Feature outliers : anomalies in a single feature only
    - Contextual       : unusual combination of individually normal values

    The ``label`` column uses:
        0 → normal
        1 → anomaly

    IMPORTANT: This label is provided for post-hoc evaluation ONLY.
    The unsupervised models must NOT receive this column during training.

    Parameters
    ----------
    n_normal : int
        Number of normal observations.
    n_anomalies : int
        Total number of anomalous observations.
    n_features : int
        Number of continuous features (minimum 4).
    random_state : int

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(random_state)
    n_features = max(n_features, 4)

    # ── Normal observations ────────────────────────────────────────────────────
    means = rng.uniform(10, 50, n_features)
    stds = rng.uniform(2, 8, n_features)
    X_normal = rng.normal(loc=means, scale=stds, size=(n_normal, n_features))

    # ── Anomaly breakdown ──────────────────────────────────────────────────────
    n_global = n_anomalies // 3
    n_local = n_anomalies // 3
    n_feature = n_anomalies // 6
    n_contextual = n_anomalies - n_global - n_local - n_feature

    # Global outliers — far from the normal cluster
    X_global = rng.normal(
        loc=means + rng.choice([-1, 1], n_features) * rng.uniform(6, 10, n_features) * stds,
        scale=stds * 0.5,
        size=(n_global, n_features),
    )

    # Local outliers — near the normal boundary but in sparse regions
    X_local = rng.normal(
        loc=means + rng.choice([-1, 1], n_features) * 3 * stds,
        scale=stds * 0.3,
        size=(n_local, n_features),
    )

    # Feature-level outliers — anomalous in exactly one feature
    X_feature = rng.normal(loc=means, scale=stds, size=(n_feature, n_features))
    anomaly_feature_idx = rng.integers(0, n_features, n_feature)
    for i, fi in enumerate(anomaly_feature_idx):
        X_feature[i, fi] = means[fi] + rng.choice([-1, 1]) * rng.uniform(7, 12) * stds[fi]

    # Contextual anomalies — unusual value combinations
    X_context = rng.normal(loc=means, scale=stds, size=(n_contextual, n_features))
    for i in range(n_contextual):
        fi = rng.integers(0, n_features - 1)
        X_context[i, fi] = means[fi] + rng.uniform(4, 6) * stds[fi]
        X_context[i, fi + 1] = means[fi + 1] - rng.uniform(4, 6) * stds[fi + 1]

    # ── Assemble ───────────────────────────────────────────────────────────────
    X_anomalies = np.vstack([X_global, X_local, X_feature, X_context])

    labels_normal = np.zeros(n_normal, dtype=int)
    labels_anomaly = np.ones(len(X_anomalies), dtype=int)

    X_all = np.vstack([X_normal, X_anomalies])
    y_all = np.concatenate([labels_normal, labels_anomaly])

    # Shuffle
    shuffle_idx = rng.permutation(len(X_all))
    X_all = X_all[shuffle_idx]
    y_all = y_all[shuffle_idx]

    # Build DataFrame
    feature_names = [
        "transaction_amount", "account_age_days", "daily_transaction_count",
        "avg_transaction_value", "device_count", "geographic_distance_km",
    ]
    # Pad if more features were requested
    while len(feature_names) < n_features:
        feature_names.append(f"feature_{len(feature_names) + 1}")
    feature_names = feature_names[:n_features]

    df = pd.DataFrame(X_all, columns=feature_names)

    # Add a categorical feature
    categories = ["online", "in-store", "mobile", "atm"]
    cat_probs = [0.40, 0.30, 0.20, 0.10]
    df["transaction_type"] = rng.choice(categories, size=len(df), p=cat_probs)

    # Add a second categorical
    regions = ["north", "south", "east", "west", "central"]
    df["region"] = rng.choice(regions, size=len(df))

    # Add a synthetic ID column (should be excluded from training)
    df.insert(0, "transaction_id", range(1000, 1000 + len(df)))

    # Add label (post-hoc evaluation only — NOT for training)
    df["label"] = y_all

    # Round continuous features to 2 decimal places
    for col in feature_names:
        df[col] = df[col].round(2)

    return df

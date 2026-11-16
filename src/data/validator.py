"""
src/data/validator.py

Dataset validation for the Explainable Anomaly Detection Framework.

Validates that a loaded DataFrame meets minimum requirements for anomaly
detection and produces a structured validation report with warnings.
"""

from typing import Any

import pandas as pd

# ── Validation thresholds ──────────────────────────────────────────────────────

MIN_ROWS = 20
MIN_FEATURES_AFTER_PREPROCESSING = 1
MAX_MISSING_FRACTION = 0.80        # Warn if a column has >80% missing
HIGH_CARDINALITY_THRESHOLD = 50    # Warn if categorical col has >50 unique values
NEAR_CONSTANT_VARIANCE_THRESHOLD = 1e-10  # Flag near-zero variance columns


class ValidationError(ValueError):
    """
    Raised when a dataset fails a hard validation check.

    Subclasses ``ValueError`` deliberately: a dataset with 12 rows is a bad
    *value supplied by the caller*, not an internal failure.  The API routers
    translate ``ValueError`` into HTTP 400 and let everything else become a 500,
    so deriving from the builtin is what makes "your CSV is too small" a client
    error instead of a server error.  Code that already catches
    ``ValidationError`` is unaffected.
    """


class DataValidator:
    """
    Validates a DataFrame before it enters the preprocessing pipeline.

    Produces a validation report containing:
    - Basic statistics
    - Errors (blocking issues)
    - Warnings (non-blocking notices)
    - Feature type breakdown
    """

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Run all validation checks on ``df``.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Validation report.  Always contains keys:
            ``valid``, ``errors``, ``warnings``, and all summary fields.

        Raises
        ------
        ValidationError
            If a hard check fails (e.g. empty DataFrame, too few rows).
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── Hard checks (raise immediately) ───────────────────────────────────

        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input is not a pandas DataFrame.")

        if df.empty:
            raise ValidationError("The dataset is empty (0 rows or 0 columns).")

        if df.shape[0] < MIN_ROWS:
            raise ValidationError(
                f"Dataset has only {df.shape[0]} rows. "
                f"Minimum required: {MIN_ROWS}."
            )

        # ── Column-level checks ────────────────────────────────────────────────

        numerical_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        categorical_cols = [
            c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
        ]

        # All-constant columns
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            warnings.append(
                f"Constant columns detected (will be removed during preprocessing): "
                f"{constant_cols}"
            )

        # Near-zero variance numerical columns
        low_var_cols = []
        for c in numerical_cols:
            col_data = df[c].dropna()
            if len(col_data) > 0 and col_data.var() < NEAR_CONSTANT_VARIANCE_THRESHOLD:
                low_var_cols.append(c)
        if low_var_cols:
            warnings.append(
                f"Near-zero variance numerical columns (may not contribute to detection): "
                f"{low_var_cols}"
            )

        # Missing value checks
        missing = df.isnull().sum()
        total_missing = int(missing.sum())
        high_missing_cols = [
            c for c in df.columns
            if df[c].isnull().mean() > MAX_MISSING_FRACTION
        ]
        if high_missing_cols:
            warnings.append(
                f"Columns with >{MAX_MISSING_FRACTION*100:.0f}% missing values "
                f"(imputation may be unreliable): {high_missing_cols}"
            )

        # Duplicate rows
        n_dupes = int(df.duplicated().sum())
        if n_dupes > 0:
            warnings.append(
                f"{n_dupes} duplicate rows detected. "
                "Consider removing them before training."
            )

        # High-cardinality categorical columns
        high_card_cols = [
            c for c in categorical_cols
            if df[c].nunique() > HIGH_CARDINALITY_THRESHOLD
        ]
        if high_card_cols:
            warnings.append(
                f"High-cardinality categorical columns "
                f"(>{HIGH_CARDINALITY_THRESHOLD} unique values, "
                f"may inflate feature space): {high_card_cols}"
            )

        # No numerical columns at all
        if len(numerical_cols) == 0:
            errors.append(
                "No numerical columns found. "
                "After one-hot encoding, there should be numerical features for anomaly detection."
            )

        # Check for unsupported dtypes (e.g. datetime stored as object)
        unsupported = []
        for c in df.columns:
            dtype_str = str(df[c].dtype)
            if "datetime" in dtype_str or "timedelta" in dtype_str:
                unsupported.append(c)
        if unsupported:
            warnings.append(
                f"Datetime columns detected: {unsupported}. "
                "These will be excluded unless converted to numerical first."
            )

        # Minimum usable features after removing constants and IDs
        usable_numerical = [
            c for c in numerical_cols
            if c not in constant_cols
        ]
        usable_categorical = [
            c for c in categorical_cols
            if c not in constant_cols
        ]
        if len(usable_numerical) + len(usable_categorical) < MIN_FEATURES_AFTER_PREPROCESSING:
            errors.append(
                "No usable features remain after removing constant columns. "
                "Cannot run anomaly detection."
            )

        # Very wide datasets
        if df.shape[1] > 200:
            warnings.append(
                f"Dataset has {df.shape[1]} columns. "
                "Very high-dimensional data may reduce interpretability of explanations."
            )

        # ── Compile report ─────────────────────────────────────────────────────

        valid = len(errors) == 0

        report: dict[str, Any] = {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "constant_cols": constant_cols,
            "high_missing_cols": high_missing_cols,
            "high_cardinality_cols": high_card_cols,
            "n_duplicates": n_dupes,
            "total_missing": total_missing,
            "missing_pct": round(total_missing / max(df.size, 1) * 100, 2),
        }

        return report

    @staticmethod
    def format_report(report: dict[str, Any]) -> str:
        """
        Return a human-readable string representation of the validation report.

        Parameters
        ----------
        report : dict

        Returns
        -------
        str
        """
        lines = [
            "=" * 60,
            "  Dataset Validation Report",
            "=" * 60,
            f"  Status  : {'PASSED' if report['valid'] else 'FAILED'}",
            f"  Rows    : {report['n_rows']}",
            f"  Columns : {report['n_cols']}",
            f"  Numerical features   : {len(report['numerical_cols'])}",
            f"  Categorical features : {len(report['categorical_cols'])}",
            f"  Total missing values : {report['total_missing']} "
            f"({report['missing_pct']}%)",
            f"  Duplicate rows       : {report['n_duplicates']}",
        ]

        if report["errors"]:
            lines.append("\n  ERRORS (blocking):")
            for e in report["errors"]:
                lines.append(f"    [X] {e}")

        if report["warnings"]:
            lines.append("\n  WARNINGS (non-blocking):")
            for w in report["warnings"]:
                lines.append(f"    [!] {w}")

        lines.append("=" * 60)
        return "\n".join(lines)

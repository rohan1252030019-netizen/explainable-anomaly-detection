"""
src/data/profiler.py

Advanced Data Intelligence and Dataset Profiling Engine for SENTINEL.

Provides:
- Automatic schema discovery (numeric, categorical, binary, datetime, id candidates).
- Missing value, duplicate, and cardinality analysis.
- Multicollinearity diagnostics (correlation condition number, collinear pairs).
- Transparent, evidence-based Dataset Health Score (0-100) with itemized deductions.
- Preprocessing and modeling recommendations.
"""

import re
from typing import Any

import numpy as np
import pandas as pd

ID_REGEX = re.compile(r"(^id$|_id$|id_|uuid|guid|key|index|pk)", re.IGNORECASE)
NEAR_CONSTANT_VARIANCE = 1e-8
HIGH_CARDINALITY_THRESHOLD = 50
COLLINEARITY_CORR_THRESHOLD = 0.90


class DataProfiler:
    """
    Profiles a pandas DataFrame to discover schema, quality issues,
    and computational characteristics before training anomaly detectors.
    """

    def __init__(
        self,
        high_cardinality_limit: int = HIGH_CARDINALITY_THRESHOLD,
        collinear_threshold: float = COLLINEARITY_CORR_THRESHOLD,
    ):
        self.high_cardinality_limit = high_cardinality_limit
        self.collinear_threshold = collinear_threshold

    def profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Run complete data intelligence audit on ``df``.

        Returns
        -------
        dict[str, Any]
            Structured profiling report containing schema, quality,
            multicollinearity, health score, and warnings.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}.")

        n_rows, n_cols = df.shape
        if n_rows == 0 or n_cols == 0:
            return {
                "n_rows": n_rows,
                "n_cols": n_cols,
                "health_score": 0.0,
                "warnings": ["Dataset is empty."],
                "schema": {},
                "quality": {},
                "collinearity": {},
            }

        schema = self._discover_schema(df)
        quality = self._audit_quality(df, schema)
        collinearity = self._audit_collinearity(df, schema["numerical_cols"])
        health = self._compute_health_score(df, quality, collinearity, schema)

        # Generate actionable recommendations
        recommendations = self._generate_recommendations(schema, quality, collinearity)

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
            "schema": schema,
            "quality": quality,
            "collinearity": collinearity,
            "health": health,
            "recommendations": recommendations,
        }

    def _discover_schema(self, df: pd.DataFrame) -> dict[str, Any]:
        """Classify each column into a semantic data type."""
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        binary_cols: list[str] = []
        datetime_cols: list[str] = []
        id_candidates: list[str] = []
        constant_cols: list[str] = []
        high_cardinality_cols: list[str] = []

        n_rows = len(df)

        for col in df.columns:
            series = df[col]
            n_unique = int(series.nunique(dropna=True))

            # Constant check
            if n_unique <= 1:
                constant_cols.append(col)
                continue

            # Binary check
            if n_unique == 2:
                binary_cols.append(col)
                continue

            # ID candidate check:
            # 1) Name matches ID pattern and has high uniqueness
            # 2) OR Integer/String/Object type with 100% uniqueness (floating point continuous variables excluded)
            is_name_id = bool(ID_REGEX.search(str(col)))
            uniqueness_ratio = n_unique / max(n_rows, 1)
            is_discrete = (
                pd.api.types.is_integer_dtype(series)
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            )

            if (is_name_id and (uniqueness_ratio > 0.8 or n_unique >= 20)) or (
                is_discrete and uniqueness_ratio == 1.0 and n_rows >= 20
            ):
                id_candidates.append(col)
                continue

            # Datetime check
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(col)
                continue
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                sample = series.dropna().head(10)
                if not sample.empty:
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                        if parsed.notnull().all():
                            datetime_cols.append(col)
                            continue
                    except Exception:
                        pass

            # Numeric vs Categorical
            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
                if n_unique > self.high_cardinality_limit:
                    high_cardinality_cols.append(col)

        return {
            "numerical_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "binary_cols": binary_cols,
            "datetime_cols": datetime_cols,
            "id_candidates": id_candidates,
            "constant_cols": constant_cols,
            "high_cardinality_cols": high_cardinality_cols,
            "feature_count": len(numeric_cols) + len(categorical_cols) + len(binary_cols),
        }

    def _audit_quality(self, df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
        """Compute detailed quality metrics across all features."""
        n_rows = len(df)
        total_cells = df.size
        null_counts = df.isnull().sum()
        total_missing = int(null_counts.sum())
        missing_rate = float(total_missing / total_cells) if total_cells > 0 else 0.0

        cols_with_missing = {
            col: {
                "count": int(count),
                "fraction": round(float(count / n_rows), 4),
            }
            for col, count in null_counts.items()
            if count > 0
        }

        # Duplicate row analysis
        duplicate_rows_count = int(df.duplicated().sum())
        duplicate_rate = float(duplicate_rows_count / n_rows) if n_rows > 0 else 0.0

        # Near-constant numeric columns (variance < threshold)
        near_constant_cols = []
        for col in schema["numerical_cols"]:
            var = float(df[col].var(skipna=True)) if len(df[col].dropna()) > 1 else 0.0
            if 0.0 <= var < NEAR_CONSTANT_VARIANCE:
                near_constant_cols.append(col)

        # Mode frequency check (if top value occurs in >95% of rows)
        dominant_feature_cols = []
        for col in df.columns:
            mode_counts = df[col].value_counts(dropna=True)
            if not mode_counts.empty and (mode_counts.iloc[0] / n_rows) > 0.95:
                if col not in schema["constant_cols"] and col not in near_constant_cols:
                    dominant_feature_cols.append(col)

        return {
            "total_missing_cells": total_missing,
            "missing_rate": round(missing_rate, 4),
            "columns_with_missing": cols_with_missing,
            "duplicate_rows_count": duplicate_rows_count,
            "duplicate_rate": round(duplicate_rate, 4),
            "near_constant_cols": near_constant_cols,
            "dominant_feature_cols": dominant_feature_cols,
        }

    def _audit_collinearity(self, df: pd.DataFrame, numerical_cols: list[str]) -> dict[str, Any]:
        """Analyze multicollinearity among numerical features."""
        if len(numerical_cols) < 2:
            return {
                "collinear_pairs": [],
                "condition_number": 1.0,
                "has_severe_multicollinearity": False,
            }

        clean_sub = df[numerical_cols].dropna()
        if len(clean_sub) < 5:
            return {
                "collinear_pairs": [],
                "condition_number": 1.0,
                "has_severe_multicollinearity": False,
            }

        # Pairwise Pearson correlation
        corr_matrix = clean_sub.corr(method="pearson").abs()
        collinear_pairs = []

        cols = numerical_cols
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_i, col_j = cols[i], cols[j]
                val = corr_matrix.loc[col_i, col_j]
                if not np.isnan(val) and val >= self.collinear_threshold:
                    collinear_pairs.append({
                        "feature_a": col_i,
                        "feature_b": col_j,
                        "correlation": round(float(val), 4),
                    })

        # Condition number of standardized feature matrix
        try:
            X_std = (clean_sub - clean_sub.mean()) / (clean_sub.std().replace(0, 1.0))
            singular_values = np.linalg.svd(X_std.values, compute_uv=False)
            cond_num = float(singular_values[0] / max(singular_values[-1], 1e-12))
        except Exception:
            cond_num = 1.0

        has_severe = bool(cond_num > 50.0 or len(collinear_pairs) > 3)

        return {
            "collinear_pairs": collinear_pairs,
            "condition_number": round(cond_num, 2),
            "has_severe_multicollinearity": has_severe,
        }

    def _compute_health_score(
        self,
        df: pd.DataFrame,
        quality: dict[str, Any],
        collinearity: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compute an itemized Dataset Health Score between 0.0 and 100.0.
        """
        score = 100.0
        deductions: list[dict[str, Any]] = []

        # 1. Missingness penalty
        missing_rate = quality["missing_rate"]
        if missing_rate > 0.0:
            pen = min(25.0, missing_rate * 50.0)
            score -= pen
            deductions.append({
                "category": "Missing Values",
                "points": round(pen, 1),
                "reason": f"{round(missing_rate * 100, 2)}% missing cell rate across dataset.",
            })

        # 2. Duplicate rows penalty
        dup_rate = quality["duplicate_rate"]
        if dup_rate > 0.0:
            pen = min(15.0, dup_rate * 30.0)
            score -= pen
            deductions.append({
                "category": "Duplicate Rows",
                "points": round(pen, 1),
                "reason": f"{quality['duplicate_rows_count']} duplicate rows ({round(dup_rate * 100, 2)}%).",
            })

        # 3. Constant and near-constant features
        bad_cols = len(schema["constant_cols"]) + len(quality["near_constant_cols"])
        total_features = max(schema["feature_count"] + bad_cols, 1)
        if bad_cols > 0:
            frac = bad_cols / total_features
            pen = min(20.0, frac * 40.0)
            score -= pen
            deductions.append({
                "category": "Zero / Near-Zero Variance Features",
                "points": round(pen, 1),
                "reason": f"{bad_cols} features have zero or near-zero variance.",
            })

        # 4. Multicollinearity penalty
        if collinearity["has_severe_multicollinearity"]:
            pen = 15.0 if len(collinearity["collinear_pairs"]) <= 2 else 20.0
            score -= pen
            deductions.append({
                "category": "Multicollinearity",
                "points": pen,
                "reason": f"High condition number ({collinearity['condition_number']}) and {len(collinearity['collinear_pairs'])} collinear pairs (|r| >= {self.collinear_threshold}).",
            })

        # 5. High cardinality categoricals
        n_high_card = len(schema["high_cardinality_cols"])
        if n_high_card > 0:
            pen = min(10.0, n_high_card * 3.0)
            score -= pen
            deductions.append({
                "category": "High Cardinality Categoricals",
                "points": round(pen, 1),
                "reason": f"{n_high_card} categorical columns exceed {self.high_cardinality_limit} unique categories.",
            })

        # 6. Sample size check
        n_rows = len(df)
        if n_rows < 50:
            pen = 10.0 if n_rows < 30 else 5.0
            score -= pen
            deductions.append({
                "category": "Sample Size",
                "points": pen,
                "reason": f"Low row count ({n_rows} rows). Unsupervised density estimators require adequate support.",
            })

        final_score = max(0.0, min(100.0, round(score, 1)))

        if final_score >= 85:
            grade = "A (Excellent)"
            status = "Ready for unsupervised anomaly modeling."
        elif final_score >= 70:
            grade = "B (Good)"
            status = "Minor data quality issues; standard preprocessing will resolve."
        elif final_score >= 50:
            grade = "C (Fair)"
            status = "Moderate data issues present; review collinearity and missingness."
        else:
            grade = "D (Poor)"
            status = "Significant data hygiene issues; recommend cleaning prior to modeling."

        return {
            "score": final_score,
            "grade": grade,
            "status": status,
            "deductions": deductions,
        }

    def _generate_recommendations(
        self,
        schema: dict[str, Any],
        quality: dict[str, Any],
        collinearity: dict[str, Any],
    ) -> list[str]:
        """Produce concrete preprocessing and algorithmic advice."""
        recs: list[str] = []

        if schema["id_candidates"]:
            recs.append(
                f"Exclude candidate ID columns {schema['id_candidates']} from model features to prevent memorization."
            )

        if schema["constant_cols"]:
            recs.append(
                f"Drop constant columns {schema['constant_cols']} as they convey zero anomaly information."
            )

        if quality["missing_rate"] > 0:
            recs.append(
                f"Missing values detected ({round(quality['missing_rate']*100, 2)}%). Use median imputation for numericals and mode for categoricals with missingness indicators."
            )

        if collinearity["collinear_pairs"]:
            pair_strs = [f"({p['feature_a']}, {p['feature_b']})" for p in collinearity["collinear_pairs"][:3]]
            recs.append(
                f"Collinear feature pairs detected {pair_strs}. Tree-based models (Isolation Forest) are robust to collinearity, while Elliptic Envelope or distance methods may become ill-conditioned."
            )

        if schema["high_cardinality_cols"]:
            recs.append(
                f"High-cardinality categorical features {schema['high_cardinality_cols']} should use frequency encoding rather than one-hot encoding to avoid dimensional explosion."
            )

        if not recs:
            recs.append("Dataset features look well-conditioned. Standard scaling and Isolation Forest / LOF are recommended.")

        return recs

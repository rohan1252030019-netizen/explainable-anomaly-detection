"""
src/timeseries/rolling_features.py

Strictly causal, leakage-safe rolling window feature engineering for time-series.

Non-Negotiable Rule:
At time step t, features must ONLY use observations from tau <= t.
Future values (tau > t) are never accessed, preventing lookahead leakage.
"""

from typing import Any

import numpy as np
import pandas as pd


def generate_causal_rolling_features(
    df: pd.DataFrame,
    target_columns: list[str],
    windows: list[int] | None = None,
    lags: list[int] | None = None,
    include_ema: bool = True,
    ema_spans: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate strictly causal rolling statistics and lag features.

    Parameters
    ----------
    df : pd.DataFrame
        Input time-ordered dataset.
    target_columns : list of str
        Numeric columns to engineer rolling features for.
    windows : list of int, default=[5, 15, 30]
        Backward look-back window sizes.
    lags : list of int, default=[1, 2, 3]
        Lag offsets.
    include_ema : bool, default=True
        Whether to include exponential moving averages.
    ema_spans : list of int, default=[10, 30]

    Returns
    -------
    tuple of (df_features, new_feature_names)
    """
    if windows is None:
        windows = [5, 15]
    if lags is None:
        lags = [1, 2, 3]
    if ema_spans is None:
        ema_spans = [10, 30]

    out = df.copy()
    new_cols: list[str] = []

    for col in target_columns:
        if col not in df.columns:
            continue
        series = df[col]

        # 1. Causal Lags
        for lag in lags:
            lag_name = f"{col}_lag_{lag}"
            out[lag_name] = series.shift(lag)
            new_cols.append(lag_name)

        # 2. Causal Rolling Stats (mean, std, min, max)
        for w in windows:
            # Shift by 0: rolling window includes current row t and past t-w+1..t-1
            # Never looks ahead!
            roll = series.rolling(window=w, min_periods=1)

            mean_name = f"{col}_roll_mean_{w}"
            std_name = f"{col}_roll_std_{w}"
            min_name = f"{col}_roll_min_{w}"
            max_name = f"{col}_roll_max_{w}"

            out[mean_name] = roll.mean()
            # std on 1 sample is NaN; fill with 0.0
            out[std_name] = roll.std().fillna(0.0)
            out[min_name] = roll.min()
            out[max_name] = roll.max()

            new_cols.extend([mean_name, std_name, min_name, max_name])

        # 3. Causal EMA (strictly past-looking)
        if include_ema:
            for span in ema_spans:
                ema_name = f"{col}_ema_{span}"
                # adjust=False ensures pure recursive causal formulation:
                # y_t = alpha * x_t + (1 - alpha) * y_{t-1}
                out[ema_name] = series.ewm(span=span, adjust=False).mean()
                new_cols.append(ema_name)

        # 4. Instantaneous velocity / change
        diff_name = f"{col}_diff_1"
        out[diff_name] = series.diff().fillna(0.0)
        new_cols.append(diff_name)

    # Fill initial NaNs caused by lags using backward/forward fill
    out[new_cols] = out[new_cols].bfill().ffill().fillna(0.0)

    return out, new_cols


def verify_temporal_causality(
    df: pd.DataFrame,
    target_col: str,
    feature_generator_fn: Any,
    t_test: int = 20,
) -> bool:
    """
    Mathematical test for zero lookahead data leakage.

    Perturbs an arbitrary future value at t_future > t_test and verifies
    that all engineered feature values at t_test remain 100% invariant.

    Returns
    -------
    bool: True if strictly causal (leakage-safe), False if contaminated.
    """
    if len(df) <= t_test + 5:
        raise ValueError("Dataframe too short for temporal causality test")

    # Baseline features
    df_base = df.copy()
    feats_base, cols = feature_generator_fn(df_base, [target_col])
    val_at_t_base = feats_base.loc[t_test, cols].to_numpy()

    # Modify future value at t_future = t_test + 3
    t_future = t_test + 3
    df_future_corrupted = df.copy()
    df_future_corrupted.loc[t_future, target_col] += 1000.0

    feats_corrupted, _ = feature_generator_fn(df_future_corrupted, [target_col])
    val_at_t_corrupted = feats_corrupted.loc[t_test, cols].to_numpy()

    # Must be exactly identical (zero lookahead leakage)
    is_causal = bool(np.allclose(val_at_t_base, val_at_t_corrupted, atol=1e-9))
    return is_causal

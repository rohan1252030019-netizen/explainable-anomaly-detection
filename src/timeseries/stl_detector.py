"""
src/timeseries/stl_detector.py

Seasonal-Trend Decomposition (STL) and Residual Anomaly Detection.

Decomposes a time series y_t into three additive components:
    y_t = Trend_t + Seasonal_t + Residual_t

Detection Principle:
Anomalies that occur during expected seasonal peaks (e.g. Black Friday or diurnal cycles)
are normal in absolute value, while small anomalies during expected quiet troughs
are anomalous. Evaluating raw values causes false positives/negatives.
By subtracting Trend and Seasonality, detection operates strictly on the Residual:
    Score_t = |Residual_t - Median(Residual)| / (MAD(Residual) * 1.4826)

This robust Z-score isolates true structural irregularities without lookahead leakage.
"""

from typing import Any

import numpy as np


class STLAnomalyDetector:
    """
    Robust Seasonal-Trend Decomposition for residual anomaly detection.

    Parameters
    ----------
    period : int, default=7
        Seasonal frequency/period (e.g. 7 for weekly, 24 for hourly daily).
    robust_z_threshold : float, default=3.0
        Threshold on the residual robust Z-score for flagging anomalies.
    trend_window : int or None, default=None
        Size of the centered moving average window for trend extraction.
        Defaults to next odd integer >= 2 * period + 1.
    """

    def __init__(
        self,
        period: int = 7,
        robust_z_threshold: float = 3.0,
        trend_window: int | None = None,
    ):
        if period < 2:
            raise ValueError(f"Period must be >= 2, got {period}")
        self.period = period
        self.robust_z_threshold = robust_z_threshold

        if trend_window is None:
            self.trend_window = 2 * period + 1
        else:
            self.trend_window = trend_window if trend_window % 2 == 1 else trend_window + 1

    def fit_transform(
        self,
        series: np.ndarray,
    ) -> dict[str, Any]:
        """
        Decompose 1D series and flag residual anomalies.

        Parameters
        ----------
        series : np.ndarray of shape (N,)

        Returns
        -------
        dict containing:
            - trend: np.ndarray
            - seasonal: np.ndarray
            - residual: np.ndarray
            - anomaly_scores: np.ndarray in [0, 1]
            - robust_z_scores: np.ndarray
            - is_anomaly: np.ndarray of bool
            - n_anomalies: int
            - anomaly_indices: list of int
        """
        y = np.asarray(series, dtype=np.float64).ravel()
        n = len(y)
        if n < self.period * 2:
            raise ValueError(f"Series length {n} is too short for period {self.period} (requires >= {self.period * 2})")

        # 1. Estimate Trend via Moving Average with edge extrapolation
        half_w = self.trend_window // 2
        trend = np.empty(n, dtype=np.float64)

        # Centered moving average for interior
        padded = np.pad(y, half_w, mode="edge")
        weights = np.ones(self.trend_window) / self.trend_window
        conv = np.convolve(padded, weights, mode="valid")
        trend[:n] = conv[:n]

        # 2. Detrend series
        detrended = y - trend

        # 3. Estimate Seasonal component by averaging across each phase of the period
        seasonal_cycle = np.zeros(self.period, dtype=np.float64)
        for p in range(self.period):
            phase_indices = np.arange(p, n, self.period)
            seasonal_cycle[p] = np.nanmedian(detrended[phase_indices])

        # Center seasonal cycle around 0
        seasonal_cycle -= np.mean(seasonal_cycle)

        # Tile seasonal cycle over entire length
        seasonal = np.tile(seasonal_cycle, int(np.ceil(n / self.period)))[:n]

        # 4. Compute Residual
        residual = y - trend - seasonal

        # 5. Robust Z-score on residual using Median Absolute Deviation (MAD)
        med_res = float(np.median(residual))
        mad_res = float(np.median(np.abs(residual - med_res)))
        # Normal consistency scale factor 1.4826
        sigma_res = max(mad_res * 1.4826, 1e-6)

        robust_z = np.abs(residual - med_res) / sigma_res

        # Map robust Z to bounded [0, 1] anomaly score
        # Using sigmoid-style mapping centered at threshold
        anomaly_scores = 1.0 / (1.0 + np.exp(-(robust_z - self.robust_z_threshold)))

        is_anomaly = robust_z >= self.robust_z_threshold
        anomaly_indices = np.where(is_anomaly)[0].tolist()

        return {
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "robust_z_scores": robust_z,
            "anomaly_scores": anomaly_scores,
            "is_anomaly": is_anomaly,
            "n_anomalies": int(np.sum(is_anomaly)),
            "anomaly_indices": anomaly_indices,
            "period": self.period,
            "residual_mad": sigma_res,
        }

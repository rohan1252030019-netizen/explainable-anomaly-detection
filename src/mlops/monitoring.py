"""
src/mlops/monitoring.py

SENTINEL Production Observability & Hardening:
1. Prometheus Metrics Exporter (/metrics endpoint format)
2. Token Bucket Sliding Rate Limiter
3. Structured Request Logging Middleware
"""

import time
from collections import defaultdict


class MetricsTracker:
    """In-memory telemetry counters for Prometheus exposition."""

    def __init__(self):
        self.request_counts = defaultdict(int)
        self.anomalies_detected_count = 0
        self.last_psi_value = 0.0

    def record_request(self, endpoint: str, method: str, status_code: int):
        key = f'{endpoint}|{method}|{status_code}'
        self.request_counts[key] += 1

    def record_anomalies(self, count: int):
        self.anomalies_detected_count += count

    def set_psi(self, psi: float):
        self.last_psi_value = float(psi)

    def generate_prometheus_metrics(self) -> str:
        """Produce standard Prometheus exposition format text."""
        lines = [
            "# HELP sentinel_http_requests_total Total HTTP requests handled by SENTINEL.",
            "# TYPE sentinel_http_requests_total counter",
        ]
        for key, count in self.request_counts.items():
            endpoint, method, status = key.split("|")
            lines.append(
                f'sentinel_http_requests_total{{endpoint="{endpoint}",method="{method}",status="{status}"}} {count}'
            )

        lines.extend([
            "# HELP sentinel_anomalies_detected_total Cumulative count of detected anomalies.",
            "# TYPE sentinel_anomalies_detected_total counter",
            f"sentinel_anomalies_detected_total {self.anomalies_detected_count}",
            "# HELP sentinel_streaming_drift_psi Latest Population Stability Index (PSI).",
            "# TYPE sentinel_streaming_drift_psi gauge",
            f"sentinel_streaming_drift_psi {self.last_psi_value:.4f}",
        ])
        return "\n".join(lines) + "\n"


metrics_tracker = MetricsTracker()


class RateLimiter:
    """
    Sliding window rate limiter per client IP.

    Parameters
    ----------
    max_requests_per_minute : int, default=120
    """

    def __init__(self, max_requests_per_minute: int = 120):
        self.max_rpm = max_requests_per_minute
        # Map client_ip -> list of timestamps
        self.requests = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - 60.0

        # Filter out timestamps older than 60 seconds
        timestamps = [t for t in self.requests[client_ip] if t >= window_start]
        self.requests[client_ip] = timestamps

        if len(timestamps) >= self.max_rpm:
            return False

        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter(max_requests_per_minute=200)

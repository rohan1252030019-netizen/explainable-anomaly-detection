"""
scripts/smoke_ulb_api.py

Exercise the full SENTINEL HTTP surface against the real ULB credit-card fraud
dataset, in-process, with FastAPI's TestClient.

This is an integration smoke test, not a unit test: it asserts that the app's
real routes accept the real file and produce labelled, leakage-safe metrics.
It prints a compact report and exits non-zero if a required step fails.

Run
---
    python scripts/smoke_ulb_api.py
"""

from __future__ import annotations

import io
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

SAMPLE = os.path.join(PROJECT_ROOT, "data", "sample", "creditcard_fraud_sample.csv")

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def timed(label: str, fn):
    started = time.perf_counter()
    result = fn()
    print(f"    {label}: {time.perf_counter() - started:.2f}s")
    return result


def main() -> int:
    if not os.path.exists(SAMPLE):
        raise SystemExit(
            f"Missing {SAMPLE}. Run scripts/prepare_ulb_dataset.py first."
        )

    client = TestClient(app)

    print("\n1. Health")
    health = client.get("/api/health")
    check("GET /api/health", health.status_code == 200, str(health.status_code))

    print("\n2. Upload the real ULB sample")
    with open(SAMPLE, "rb") as handle:
        payload = handle.read()
    upload = client.post(
        "/api/dataset/upload",
        files={"file": ("creditcard_fraud_sample.csv", io.BytesIO(payload), "text/csv")},
    )
    check("POST /api/dataset/upload", upload.status_code == 200,
          f"{upload.status_code} {upload.text[:200]}")
    if upload.status_code != 200:
        return 1
    dataset = upload.json()["dataset"]
    check("row count preserved", dataset["n_rows"] == 20_000, str(dataset["n_rows"]))
    check("31 columns detected", dataset["n_cols"] == 31, str(dataset["n_cols"]))
    check("label auto-detected as Class", dataset["label_col"] == "Class",
          str(dataset["label_col"]))
    check("feature matrix ready", dataset["feature_matrix_ready"])
    check("no missing values", dataset["total_missing"] == 0,
          str(dataset["total_missing"]))

    print("\n3. Deep profiling")
    profile = timed("GET /api/dataset/profile", lambda: client.get("/api/dataset/profile"))
    check("POST profile", profile.status_code == 200,
          f"{profile.status_code} {profile.text[:160]}")

    print("\n4. Isolation Forest detection")
    detection = timed(
        "POST /api/detection/run",
        lambda: client.post(
            "/api/detection/run",
            json={
                "model_name": "Isolation Forest",
                "contamination": 0.02,
                "threshold_method": "contamination",
            },
        ),
    )
    check("detection ran", detection.status_code == 200,
          f"{detection.status_code} {detection.text[:200]}")
    if detection.status_code == 200:
        overview = detection.json()["overview"]
        check("anomalies flagged", overview.get("n_anomalies", 0) > 0,
              str(overview.get("n_anomalies")))

    print("\n5. Held-out evaluation (leakage-safe)")
    holdout = timed(
        "GET /api/evaluation/holdout",
        lambda: client.get("/api/evaluation/holdout", params={"test_size": 0.3}),
    )
    check("holdout available", holdout.status_code == 200,
          f"{holdout.status_code} {holdout.text[:200]}")
    if holdout.status_code == 200:
        report = holdout.json()
        check("held-out metrics available", bool(report.get("available")))
        oos = report.get("out_of_sample") or {}
        check("PR-AUC reported", isinstance(oos.get("pr_auc"), (int, float)),
              f"{oos.get('pr_auc')}")
        check("ROC-AUC reported", isinstance(oos.get("roc_auc"), (int, float)),
              f"{oos.get('roc_auc')}")

    print("\n6. Explanations")
    explanations = timed(
        "POST /api/explainability/generate",
        lambda: client.post(
            "/api/explainability/generate", json={"use_shap": False, "top_k": 20}
        ),
    )
    check("explanations generated", explanations.status_code == 200,
          f"{explanations.status_code} {explanations.text[:200]}")

    print("\n7. Consensus ensemble")
    ensemble = timed(
        "POST /api/detection/ensemble",
        lambda: client.post(
            "/api/detection/ensemble",
            json={
                "models": [
                    "Isolation Forest",
                    "Histogram-Based Outlier Score",
                    "PCA Reconstruction",
                ],
                "aggregation_method": "rank_borda",
            },
        ),
    )
    check("ensemble ran", ensemble.status_code == 200,
          f"{ensemble.status_code} {ensemble.text[:200]}")

    print("\n8. Export results")
    export = client.get("/api/export/predictions")
    check("CSV export", export.status_code == 200,
          f"{export.status_code} {len(export.content)} bytes")

    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("Failed checks:")
        for name in failed:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

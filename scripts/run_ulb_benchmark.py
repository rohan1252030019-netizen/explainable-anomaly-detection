"""
scripts/run_ulb_benchmark.py

End-to-end benchmark of the XAUD detector registry on the real ULB
credit-card fraud dataset (OpenML 1597).

Design
------
* **Time-ordered split.**  The file is already sorted by ``Time``; the first
  70 % of seconds is the fit window and the last 30 % is the held-out window.
  A random split on this dataset leaks future transactions into training.
* **Preprocessing is fitted on the fit window only.**  The scaler and imputer
  never see a held-out row.
* **Detectors are unsupervised.**  ``Class`` is removed before fitting and is
  used only after scoring, for evaluation.
* **Threshold is derived from fit scores**, then applied unchanged to the
  held-out window.  PR-AUC / ROC-AUC are threshold-free and are reported
  alongside precision, recall and precision@top-1%.

Run
---
    python scripts/run_ulb_benchmark.py
    python scripts/run_ulb_benchmark.py --data data/raw/creditcard_ulb.csv
    python scripts/run_ulb_benchmark.py --json-out outputs/reports/ulb_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import DataLoader  # noqa: E402
from src.data.preprocessing import PreprocessingPipeline  # noqa: E402
from src.detection.scoring import ScoreNormalizer  # noqa: E402
from src.evaluation.holdout import evaluate_holdout_split  # noqa: E402
from src.models.model_manager import (  # noqa: E402
    DETECTOR_REGISTRY,
    MODEL_ROW_LIMITS,
    ModelManager,
)

DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "sample", "creditcard_fraud_sample.csv")
LABEL_COL = "Class"


def load_dataset(path: str) -> pd.DataFrame:
    loader = DataLoader()
    df = loader.load_csv(path)
    if LABEL_COL not in df.columns:
        raise SystemExit(f"{path} has no '{LABEL_COL}' column; is this the ULB file?")
    summary = loader.get_summary()
    print(f"Loaded {path}")
    print(
        f"  rows={summary['n_rows']:,}  cols={summary['n_cols']}  "
        f"missing={summary['total_missing']:,}  "
        f"duplicates={summary['duplicate_rows']:,}"
    )
    print(
        f"  detected label column: {summary['potential_label_col']!r}  "
        f"memory={summary['memory_mb']:.1f} MB"
    )
    return df


def time_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    cut = int(round(len(ordered) * (1.0 - test_size)))
    return ordered.iloc[:cut].reset_index(drop=True), ordered.iloc[cut:].reset_index(drop=True)


def top_k_precision(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    return float(np.mean(y_true[order]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DETECTOR_REGISTRY),
        help="Subset of registered detectors to run.",
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    fit_df, test_df = time_split(df, args.test_size)

    y_fit = fit_df[LABEL_COL].to_numpy(dtype=int)
    y_test = test_df[LABEL_COL].to_numpy(dtype=int)
    print(
        f"\nTime split: fit={len(fit_df):,} rows "
        f"({y_fit.sum()} frauds, {y_fit.mean() * 100:.2f}%) | "
        f"test={len(test_df):,} rows ({y_test.sum()} frauds, "
        f"{y_test.mean() * 100:.2f}%)"
    )

    # Fit preprocessing on the fit window only, then transform the holdout.
    pipeline = PreprocessingPipeline(
        label_col=LABEL_COL,
        scaler_type="robust",
        drop_duplicates=False,
    )
    X_fit, feature_names = pipeline.fit_transform(fit_df)
    X_test = pipeline.transform(test_df)
    print(f"Prepared matrices: fit={X_fit.shape} test={X_test.shape}")

    # Flag budget: the fit-window fraud prevalence.  Threshold-free metrics are
    # unaffected by this choice; it only sets the operating point.
    contamination = float(np.clip(y_fit.mean(), 1e-4, 0.5))

    manager = ModelManager()
    results = []
    ensemble_inputs: dict[str, dict[str, np.ndarray]] = {}

    print("\n" + "=" * 100)
    print(f"{'model':<34}{'PR-AUC':>8}{'ROC-AUC':>9}{'precision':>10}{'recall':>8}"
          f"{'P@1%':>7}{'fit s':>8}{'score s':>9}")
    print("=" * 100)

    for name in args.models:
        if name not in DETECTOR_REGISTRY:
            print(f"skipping unknown model {name!r}")
            continue
        limit = MODEL_ROW_LIMITS.get(name)
        if limit is not None and len(X_fit) > limit:
            print(f"{name:<34} skipped: fit rows {len(X_fit):,} > limit {limit:,}")
            continue

        params: dict = {}
        if name == "Local Outlier Factor":
            params["novelty"] = True
        if name == "One-Class SVM":
            params["nu"] = contamination
        if "contamination" in manager.create_detector(name, params).get_params():
            params["contamination"] = contamination

        def factory(n: str = name, p: dict = params):
            return manager.create_detector(n, p)

        started = time.perf_counter()
        try:
            report = evaluate_holdout_split(
                factory,
                X_fit,
                X_test,
                y_fit=y_fit,
                y_test=y_test,
                threshold_method="contamination",
                contamination=contamination,
            )
        except Exception as exc:  # a single detector failure must not abort the run
            print(f"{name:<34} FAILED: {type(exc).__name__}: {exc}")
            results.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        elapsed = time.perf_counter() - started

        if not report.get("available"):
            print(f"{name:<34} unavailable: {report.get('reason')}")
            results.append({"model": name, "error": report.get("reason")})
            continue

        out = report["out_of_sample"]

        # Refit to capture test scores for the ensemble and top-k metric.
        detector = factory()
        t0 = time.perf_counter()
        detector.fit(X_fit)
        fit_seconds = time.perf_counter() - t0
        raw_fit = np.asarray(detector.score_samples(X_fit), dtype=float)
        t1 = time.perf_counter()
        raw_test = np.asarray(detector.score_samples(X_test), dtype=float)
        score_seconds = time.perf_counter() - t1
        normalizer = ScoreNormalizer(method="quantile").fit(raw_fit)
        norm_test = normalizer.transform(raw_test)
        ensemble_inputs[name] = {
            "scores": norm_test,
            "predictions": np.where(norm_test >= report["threshold_value"], -1, 1),
        }

        k = max(1, int(round(len(y_test) * 0.01)))
        row = {
            "model": name,
            "pr_auc": out["pr_auc"],
            "roc_auc": out["roc_auc"],
            "precision": out["precision"],
            "recall": out["recall"],
            "f1": out["f1"],
            "precision_at_top_1pct": top_k_precision(y_test, norm_test, k),
            "flag_rate": report["flag_rate_out_of_sample"],
            "fit_seconds": round(fit_seconds, 3),
            "score_seconds": round(score_seconds, 3),
            "total_seconds": round(elapsed, 3),
            "threshold": report["threshold_value"],
            "in_sample_pr_auc": report["in_sample"]["pr_auc"],
            "in_sample_roc_auc": report["in_sample"]["roc_auc"],
        }
        results.append(row)
        print(
            f"{name:<34}{row['pr_auc']:>8.4f}{row['roc_auc']:>9.4f}"
            f"{row['precision']:>10.4f}{row['recall']:>8.4f}"
            f"{row['precision_at_top_1pct']:>7.4f}"
            f"{fit_seconds:>8.2f}{score_seconds:>9.2f}"
        )

    # ── Consensus ensemble over the detectors that completed ────────────────
    ensemble_summary = None
    if len(ensemble_inputs) >= 2:
        from src.detection.ensemble import IntelligentEnsembleEngine

        engine = IntelligentEnsembleEngine(aggregation_method="rank_borda")
        ens = engine.run_ensemble(ensemble_inputs)
        ens_scores = np.asarray(ens["ensemble_scores"], dtype=float)
        k = max(1, int(round(len(y_test) * 0.01)))
        ensemble_summary = {
            "n_models": len(ensemble_inputs),
            "models": list(ensemble_inputs),
            "pr_auc": round(float(average_precision_score(y_test, ens_scores)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, ens_scores)), 4),
            "precision_at_top_1pct": round(top_k_precision(y_test, ens_scores, k), 4),
            "flagged": int((np.asarray(ens["consensus_predictions"]) == -1).sum()),
        }
        print("\nConsensus ensemble (rank Borda):")
        print(
            f"  models={ensemble_summary['n_models']}  "
            f"PR-AUC={ensemble_summary['pr_auc']:.4f}  "
            f"ROC-AUC={ensemble_summary['roc_auc']:.4f}  "
            f"P@1%={ensemble_summary['precision_at_top_1pct']:.4f}"
        )

    best = None
    completed = [r for r in results if "pr_auc" in r]
    if completed:
        best = max(completed, key=lambda r: r["pr_auc"])
        print(
            f"\nBest detector by PR-AUC: {best['model']} "
            f"({best['pr_auc']:.4f})"
        )

    payload = {
        "dataset": {
            "path": os.path.relpath(args.data, PROJECT_ROOT),
            "rows": int(len(df)),
            "frauds": int(df[LABEL_COL].sum()),
            "features_after_preprocessing": int(X_fit.shape[1]),
            "split": "time-ordered",
            "fit_rows": int(len(fit_df)),
            "test_rows": int(len(test_df)),
            "fit_frauds": int(y_fit.sum()),
            "test_frauds": int(y_test.sum()),
            "contamination_used": round(contamination, 6),
        },
        "results": results,
        "ensemble": ensemble_summary,
        "best_model": best["model"] if best else None,
    }

    if args.json_out:
        out_path = args.json_out
        if not os.path.isabs(out_path):
            out_path = os.path.join(PROJECT_ROOT, out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

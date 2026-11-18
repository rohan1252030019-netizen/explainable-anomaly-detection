"""
run.py — Entry point for the Explainable Anomaly Detection Framework.

Architecture:
    Frontend: Next.js + TypeScript + Tailwind CSS (Port 3000)
    Backend:  FastAPI + Uvicorn (Port 8001)
    ML Core:  src/

Usage:
    python run.py                      # Launch FastAPI backend
    python run.py --all                # Launch BOTH backend and frontend concurrently
    python run.py --frontend           # Launch Next.js frontend dev server
    python run.py --backend            # Launch FastAPI backend server
    python run.py --demo               # Run pipeline demo on sample data
    python run.py --test               # Run test suite
    python run.py --generate-sample    # Generate and save sample dataset
    python run.py --lint               # Run ruff over the Python sources
    python run.py --check              # Verify requirements.txt matches pyproject
"""

import argparse
import os
import subprocess
import sys

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def launch_backend(port: int | None = None):
    """Launch the FastAPI backend server."""
    # Imported here so `--help` and the other subcommands work without uvicorn
    # (and its dependencies) being installed.
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "\n  ERROR: uvicorn is not installed.\n"
            "  Run:  pip install -r requirements.txt\n"
        )
        sys.exit(1)

    from backend.config import settings

    server_port = port or settings.PORT

    print("=" * 60)
    print("  SENTINEL — Autonomous Anomaly Intelligence Platform")
    print("  FastAPI Backend Service")
    print("=" * 60)
    print(f"\n  Starting FastAPI server on http://localhost:{server_port} ...")
    print(f"  API Documentation available at: http://localhost:{server_port}/docs")
    print("  Next.js Frontend available at:  http://localhost:3000\n")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn", "backend.main:app",
            "--host", settings.HOST, "--port", str(server_port),
        ],
        cwd=PROJECT_ROOT,
    )


def launch_frontend(port: int = 3000):
    """Launch the Next.js frontend server."""
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    print("=" * 60)
    print("  SENTINEL Next.js Frontend")
    print("=" * 60)
    print(f"\n  Starting Next.js dev server on http://localhost:{port} ...\n")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    subprocess.run([npm_cmd, "run", "dev", "--", "-p", str(port)], cwd=frontend_dir)


def launch_all(backend_port: int | None = None, frontend_port: int = 3000):
    """Launch both FastAPI backend and Next.js frontend concurrently."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "\n  ERROR: uvicorn is not installed.\n"
            "  Run:  pip install -r requirements.txt\n"
        )
        sys.exit(1)

    from backend.config import settings

    b_port = backend_port or settings.PORT
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    print("=" * 60)
    print("  SENTINEL 3.0 — Launching Full-Stack Platform")
    print("=" * 60)
    print(f"\n  Backend:  http://127.0.0.1:{b_port} (Docs: http://127.0.0.1:{b_port}/docs)")
    print(f"  Frontend: http://localhost:{frontend_port}\n")
    print("  Press Ctrl+C to stop both services.\n")

    p_backend = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "backend.main:app",
            "--host", settings.HOST, "--port", str(b_port),
        ],
        cwd=PROJECT_ROOT,
    )
    p_frontend = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "-p", str(frontend_port)],
        cwd=frontend_dir,
    )

    try:
        p_backend.wait()
    except KeyboardInterrupt:
        print("\n  Stopping services...")
    finally:
        for p in (p_backend, p_frontend):
            if p.poll() is None:
                p.terminate()


def run_demo():
    """Run the detection pipeline on the sample dataset."""
    print("=" * 60)
    print("  Running Demo Pipeline")
    print("=" * 60)

    # Generate sample data if not already present
    from src.data.loader import generate_sample_dataset
    sample_path = os.path.join(PROJECT_ROOT, "data", "sample", "sample_anomaly_data.csv")
    if not os.path.exists(sample_path):
        print("\n[1/8] Generating sample dataset...")
        df = generate_sample_dataset(n_normal=800, n_anomalies=80, random_state=42)
        df.to_csv(sample_path, index=False)
        print(f"      Saved: {sample_path}")
    else:
        import pandas as pd
        df = pd.read_csv(sample_path)
        print(f"\n[1/8] Loaded existing sample dataset: {sample_path}")

    print(f"      Shape: {df.shape}")

    import numpy as np

    # Validate
    print("\n[2/8] Validating dataset...")
    from src.data.validator import DataValidator
    validator = DataValidator()
    report = validator.validate(df)
    print(f"      Rows: {report['n_rows']}, Columns: {report['n_cols']}")
    print(f"      Numerical: {len(report['numerical_cols'])}, Categorical: {len(report['categorical_cols'])}")
    print(f"      Missing values: {report['total_missing']}")

    # Preprocess
    print("\n[3/8] Preprocessing...")
    from src.data.preprocessing import PreprocessingPipeline
    label_col = "label" if "label" in df.columns else None
    pipeline = PreprocessingPipeline(label_col=label_col)
    X, feature_names = pipeline.fit_transform(df)
    y = (df[label_col].values == 1).astype(int) if label_col else None
    print(f"      Feature matrix shape: {X.shape}")

    # Train Isolation Forest
    print("\n[4/8] Training Isolation Forest...")
    from src.models.isolation_forest import IsolationForestDetector
    model = IsolationForestDetector(contamination=0.05, random_state=42)
    model.fit(X)
    raw_scores = model.score_samples(X)
    print(f"      Score range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")

    # Score and rank
    print("\n[5/8] Scoring, thresholding, ranking...")
    from src.detection.ranking import rank_anomalies
    from src.detection.scoring import ScoreNormalizer
    from src.detection.thresholding import apply_threshold, assign_severity

    normalizer = ScoreNormalizer().fit(raw_scores)
    norm_scores = normalizer.transform(raw_scores)
    labels_pred = apply_threshold(norm_scores, method="contamination", contamination=0.05)
    ranked = rank_anomalies(norm_scores, labels_pred)
    severity = assign_severity(norm_scores, anomaly_mask=labels_pred == -1)

    n_anomalies = int((labels_pred == -1).sum())
    print(f"      Anomalies detected: {n_anomalies} ({n_anomalies/len(df)*100:.2f}%)")
    spread = {s: int(((labels_pred == -1) & (severity == s)).sum())
              for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}
    print(f"      Severity split: {spread}")

    # Out-of-sample check
    print("\n[6/8] Held-out validation (model refitted on a training split)...")
    from src.evaluation.holdout import evaluate_holdout
    holdout = evaluate_holdout(
        lambda: IsolationForestDetector(contamination=0.05, random_state=42),
        X, y=y,
    )
    if holdout["available"]:
        ins, oos = holdout["in_sample"], holdout["out_of_sample"]
        print(f"      ROC-AUC  in-sample {ins['roc_auc']:.4f}  |  held-out {oos['roc_auc']:.4f}")
        print(f"      PR-AUC   in-sample {ins['pr_auc']:.4f}  |  held-out {oos['pr_auc']:.4f}")
        print(f"      Recall   in-sample {ins['recall']:.4f}  |  held-out {oos['recall']:.4f}")
        print(f"      Optimism gap (ROC-AUC): {holdout['optimism_gap']['roc_auc']:+.4f}")
    else:
        print(f"      Skipped: {holdout['reason']}")

    # Counterfactual for the top-ranked anomaly
    print("\n[7/8] Counterfactual for the top-ranked anomaly...")
    from src.explainability.counterfactuals import counterfactuals_for_anomalies
    if n_anomalies > 0:
        top_id = int(np.argmax(norm_scores))
        cfs = counterfactuals_for_anomalies(
            [top_id], X, labels_pred == 1,
            score_fn=lambda a: normalizer.transform(
                model.score_samples(np.atleast_2d(np.asarray(a, dtype=np.float64)))
            ),
            feature_names=feature_names,
            denormalize=pipeline.denormalize,
            score_threshold=float(np.percentile(norm_scores, 95)),
        )
        cf = cfs[0]
        if cf["found"]:
            print(f"      {cf['text']}")
            print(f"      Quality: {cf['quality']['label']}")
        else:
            print(f"      None found: {cf['reason']}")
    else:
        print("      No anomalies flagged.")

    # Export
    print("\n[8/8] Exporting results...")
    import pandas as pd
    results_df = df.copy()
    results_df["original_index"] = range(len(results_df))
    results_df["anomaly_score"] = raw_scores
    results_df["normalized_score"] = norm_scores
    results_df["prediction"] = labels_pred
    results_df["severity"] = severity
    results_df["rank"] = ranked
    results_df["is_anomaly"] = labels_pred == -1
    results_df["model"] = "Isolation Forest"

    out_path = os.path.join(PROJECT_ROOT, "outputs", "predictions", "demo_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"      Results saved: {out_path}")

    print("\n  Demo completed successfully.")
    print("  Run `python run.py --backend` to start the API and explore the dashboard.")
    print("=" * 60)





def run_tests():
    """Run the pytest test suite."""
    print("=" * 60)
    print("  Running Test Suite")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
    )
    sys.exit(result.returncode)


def run_lint():
    """Run ruff over the Python sources."""
    print("=" * 60)
    print("  Linting")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "backend", "tests",
         "run.py", "scripts"],
        cwd=PROJECT_ROOT,
    )
    sys.exit(result.returncode)


def run_check():
    """Verify requirements.txt and pyproject.toml agree."""
    print("=" * 60)
    print("  Checking dependency declarations")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, os.path.join("scripts", "check_requirements.py")],
        cwd=PROJECT_ROOT,
    )
    sys.exit(result.returncode)


def generate_sample():
    """Generate and save the sample dataset."""
    from src.data.loader import generate_sample_dataset
    sample_path = os.path.join(PROJECT_ROOT, "data", "sample", "sample_anomaly_data.csv")
    print("Generating sample dataset...")
    df = generate_sample_dataset(n_normal=800, n_anomalies=80, random_state=42)
    df.to_csv(sample_path, index=False)
    print(f"Saved: {sample_path}")
    print(f"Shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explainable Framework for Unsupervised Anomaly Detection"
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Launch both backend and frontend"
    )
    parser.add_argument(
        "--backend", "-b", action="store_true", help="Launch FastAPI backend"
    )
    parser.add_argument(
        "--frontend", "-f", action="store_true", help="Launch Next.js frontend"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Backend port (default: 8001)"
    )
    parser.add_argument(
        "--frontend-port", type=int, default=3000, help="Frontend port (default: 3000)"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run pipeline demo on sample data"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run pytest test suite"
    )
    parser.add_argument(
        "--generate-sample", action="store_true", help="Generate sample dataset"
    )
    parser.add_argument(
        "--lint", action="store_true", help="Run ruff over the Python sources"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify requirements.txt matches pyproject.toml",
    )
    args = parser.parse_args()

    if args.all:
        launch_all(backend_port=args.port, frontend_port=args.frontend_port)
    elif args.frontend:
        launch_frontend(port=args.frontend_port)
    elif args.demo:
        run_demo()
    elif args.test:
        run_tests()
    elif args.lint:
        run_lint()
    elif args.check:
        run_check()
    elif args.generate_sample:
        generate_sample()
    else:
        launch_backend(port=args.port)

"""
run.py — Entry point for the Explainable Anomaly Detection Framework.

Architecture:
    Frontend: Next.js + TypeScript + Tailwind CSS (Port 3000)
    Backend:  FastAPI + Uvicorn (Port 8000)
    ML Core:  src/ & models/ (100% untouched)

Usage:
    python run.py                      # Launch FastAPI backend + instructions
    python run.py --backend            # Launch FastAPI backend server
    python run.py --demo               # Run pipeline demo on sample data
    python run.py --test               # Run test suite
    python run.py --generate-sample    # Generate and save sample dataset
"""

import argparse
import subprocess
import sys
import os

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def launch_backend():
    """Launch the FastAPI backend server."""
    print("=" * 60)
    print("  Explainable Framework for Unsupervised Anomaly Detection")
    print("  FastAPI Backend Service")
    print("=" * 60)
    print("\n  Starting FastAPI server on http://localhost:8000 ...")
    print("  API Documentation available at: http://localhost:8000/docs")
    print("  Next.js Frontend available at:  http://localhost:3000\n")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=PROJECT_ROOT,
    )


def run_demo():
    """Run the detection pipeline on the sample dataset."""
    print("=" * 60)
    print("  Running Demo Pipeline")
    print("=" * 60)

    # Generate sample data if not already present
    from src.data.loader import generate_sample_dataset
    sample_path = os.path.join(PROJECT_ROOT, "data", "sample", "sample_anomaly_data.csv")
    if not os.path.exists(sample_path):
        print("\n[1/6] Generating sample dataset...")
        df = generate_sample_dataset(n_normal=800, n_anomalies=80, random_state=42)
        df.to_csv(sample_path, index=False)
        print(f"      Saved: {sample_path}")
    else:
        import pandas as pd
        df = pd.read_csv(sample_path)
        print(f"\n[1/6] Loaded existing sample dataset: {sample_path}")

    print(f"      Shape: {df.shape}")

    # Validate
    print("\n[2/6] Validating dataset...")
    from src.data.validator import DataValidator
    validator = DataValidator()
    report = validator.validate(df)
    print(f"      Rows: {report['n_rows']}, Columns: {report['n_cols']}")
    print(f"      Numerical: {len(report['numerical_cols'])}, Categorical: {len(report['categorical_cols'])}")
    print(f"      Missing values: {report['total_missing']}")

    # Preprocess
    print("\n[3/6] Preprocessing...")
    from src.data.preprocessing import PreprocessingPipeline
    label_col = "label" if "label" in df.columns else None
    pipeline = PreprocessingPipeline(label_col=label_col)
    X, feature_names = pipeline.fit_transform(df)
    print(f"      Feature matrix shape: {X.shape}")

    # Train Isolation Forest
    print("\n[4/6] Training Isolation Forest...")
    from src.models.isolation_forest import IsolationForestDetector
    model = IsolationForestDetector(contamination=0.05, random_state=42)
    model.fit(X)
    scores = model.score_samples(X)
    print(f"      Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Score and rank
    print("\n[5/6] Scoring, thresholding, ranking...")
    from src.detection.scoring import normalize_scores
    from src.detection.thresholding import apply_threshold
    from src.detection.ranking import rank_anomalies

    norm_scores = normalize_scores(scores)
    labels_pred = apply_threshold(norm_scores, method="quantile", quantile=0.95)
    ranked = rank_anomalies(norm_scores, labels_pred)

    n_anomalies = int((labels_pred == -1).sum())
    print(f"      Anomalies detected: {n_anomalies} ({n_anomalies/len(df)*100:.2f}%)")

    # Export
    print("\n[6/6] Exporting results...")
    import pandas as pd
    results_df = df.copy()
    results_df["anomaly_score"] = scores
    results_df["normalized_score"] = norm_scores
    results_df["prediction"] = labels_pred
    results_df["rank"] = ranked

    out_path = os.path.join(PROJECT_ROOT, "outputs", "predictions", "demo_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"      Results saved: {out_path}")

    print("\n  Demo completed successfully.")
    print("  Run `python run.py` to start the backend API.")
    print("=" * 60)


def run_tests():
    """Run the pytest test suite."""
    print("=" * 60)
    print("  Running Test Suite")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
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
        "--backend", action="store_true", help="Launch FastAPI backend"
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
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.test:
        run_tests()
    elif args.generate_sample:
        generate_sample()
    else:
        launch_backend()

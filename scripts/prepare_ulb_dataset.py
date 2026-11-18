"""
scripts/prepare_ulb_dataset.py

Download, verify and stage the ULB credit-card fraud dataset (OpenML id 1597).

Source
------
Machine Learning Group, Université Libre de Bruxelles (MLG-ULB) with
Worldline.  284,807 real card transactions by European cardholders over two
days in September 2013; 492 labelled frauds (0.172%).

    Dal Pozzolo, Caelen, Johnson, Bontempi (2015),
    "Calibrating Probability with Undersampling for Unbalanced Classification",
    IEEE SSCI.  DOI: 10.1109/SSCI.2015.33

What this script does
---------------------
1. Uses a cached ``data/raw/creditcard_ulb.pq`` if present, otherwise
   downloads the OpenML parquet.
2. Asserts the published shape (284,807 x 31) and fraud count (492) so a
   corrupt or substituted file fails loudly instead of silently.
3. Writes the full frame to ``data/raw/creditcard_ulb.csv`` (git-ignored) for
   research-scale runs.
4. Writes a stratified, deterministic sample to
   ``data/sample/creditcard_fraud_sample.csv`` that keeps *every* fraud and
   fits the platform's upload (25 MB) and detector row budgets (20k for LOF).

Run
---
    python scripts/prepare_ulb_dataset.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "sample")

CACHE_PARQUET = os.path.join(RAW_DIR, "creditcard_ulb.pq")
FULL_CSV = os.path.join(RAW_DIR, "creditcard_ulb.csv")
SAMPLE_CSV = os.path.join(SAMPLE_DIR, "creditcard_fraud_sample.csv")

OPENML_PARQUET = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"

EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492
EXPECTED_COLS = 31

# 492 frauds + 19,508 normals.  Exactly at the LOF row ceiling so all eight
# detectors in the registry can be exercised on the same frame.
SAMPLE_NORMAL = 19_508
RANDOM_STATE = 42


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as response:
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    print(f"Saved {destination} ({os.path.getsize(destination) / 1e6:.1f} MB)")


def load_source() -> pd.DataFrame:
    if not os.path.exists(CACHE_PARQUET):
        download(OPENML_PARQUET, CACHE_PARQUET)

    df = pd.read_parquet(CACHE_PARQUET)
    # OpenML stores Class as a categorical of strings; the published dataset
    # uses integer 0/1.  Normalise without changing any values.
    df["Class"] = df["Class"].astype(int)
    return df


def verify(df: pd.DataFrame) -> None:
    problems: list[str] = []
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLS):
        problems.append(f"shape is {df.shape}, expected {(EXPECTED_ROWS, EXPECTED_COLS)}")
    if int(df["Class"].sum()) != EXPECTED_FRAUDS:
        problems.append(f"fraud count is {int(df['Class'].sum())}, expected {EXPECTED_FRAUDS}")
    if not set(df["Class"].unique()) <= {0, 1}:
        problems.append(f"Class has unexpected values {sorted(df['Class'].unique())}")
    if df.isna().any().any():
        problems.append("dataset contains missing values, which the published file does not")

    expected_pca = {f"V{i}" for i in range(1, 29)}
    missing_pca = expected_pca - set(df.columns)
    if missing_pca:
        problems.append(f"missing PCA columns: {sorted(missing_pca)}")
    for required in ("Time", "Amount"):
        if required not in df.columns:
            problems.append(f"missing required column {required}")

    if problems:
        raise SystemExit("Dataset verification FAILED:\n  - " + "\n  - ".join(problems))


def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    frauds = df[df["Class"] == 1]
    normals = df[df["Class"] == 0].sample(
        n=SAMPLE_NORMAL, random_state=RANDOM_STATE
    )
    sample = pd.concat([normals, frauds]).sort_values("Time").reset_index(drop=True)
    return sample


def main() -> int:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    df = load_source()
    verify(df)
    print(
        f"Verified source: {df.shape[0]:,} rows x {df.shape[1]} cols, "
        f"{int(df['Class'].sum())} frauds "
        f"({df['Class'].mean() * 100:.3f}%)"
    )
    print(f"SHA-256 (parquet): {sha256(CACHE_PARQUET)}")

    df.to_csv(FULL_CSV, index=False)
    print(f"Wrote full CSV: {FULL_CSV} ({os.path.getsize(FULL_CSV) / 1e6:.1f} MB)")

    sample = build_sample(df)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(
        f"Wrote stratified sample: {SAMPLE_CSV} "
        f"({sample.shape[0]:,} rows, {int(sample['Class'].sum())} frauds, "
        f"{os.path.getsize(SAMPLE_CSV) / 1e6:.2f} MB)"
    )
    print(f"SHA-256 (sample):  {sha256(SAMPLE_CSV)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

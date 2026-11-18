# Datasets

## ULB Credit-Card Fraud (primary, real)

| Field | Value |
|---|---|
| Original publisher | Machine Learning Group, Université Libre de Bruxelles (MLG-ULB) with Worldline |
| Paper | Dal Pozzolo, Caelen, Johnson, Bontempi (2015), *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE SSCI. DOI [10.1109/SSCI.2015.33](https://doi.org/10.1109/SSCI.2015.33) |
| Canonical record | [OpenML dataset 1597](https://www.openml.org/d/1597) |
| Nature | **Real** card transactions, anonymised by PCA |
| Volume | 284,807 transactions, 31 columns |
| Labels | 492 frauds (0.172 %), `Class ∈ {0, 1}` |
| Period / geography | Two days, September 2013, European cardholders |
| Missing values | None |
| Licence | Listed as ODbL on Kaggle; OpenML records it as public. Confirm before redistribution. |

### Why this dataset

It is the only widely used fraud dataset that is both **real** and **transaction-level with a
trustworthy label**, and it is the community's shared benchmark, so results are comparable.
It is *not* raw: `V1…V28` are PCA components, and there is no customer ID, merchant, or geography.
Those limits are why the anomaly explanation surface is `V*`/`Amount`/`Time` rather than
human-readable fields.

### Files in this repository

| Path | Rows | Purpose | Tracked in git |
|---|---|---|---|
| `data/raw/creditcard_ulb.pq` | 284,807 | Cached source, exactly as served by OpenML | No (`data/raw` is ignored) |
| `data/raw/creditcard_ulb.csv` | 284,807 | Full CSV for research-scale runs (151 MB) | No |
| `data/sample/creditcard_fraud_sample.csv` | 20,000 | App/benchmark sample: **all 492 frauds** + 19,508 normal rows, sorted by `Time` | No — regenerate with the script |

The sample is deliberately sized at 20,000 rows: it keeps every fraud, stays under the
25 MB upload limit, and sits exactly at the `Local Outlier Factor` row ceiling so all eight
registered detectors can be exercised on identical rows.

### Preparation and verification

```bash
python scripts/prepare_ulb_dataset.py
```

The script downloads the OpenML parquet when absent, then **fails loudly** unless the file is
exactly 284,807 × 31 with 492 frauds, no missing values, and the `V1…V28`/`Time`/`Amount`
schema. It prints SHA-256 digests of both the source and the sample.

Known digests for this copy:

```
SHA-256 (parquet): b7efcb35a428bbe22347a05d2437d9177bab07ce61e51214a17bec584ad9496d
SHA-256 (sample):  ede5aa5ca3656a79e724735dee2b02085fdcf6c60aa37d9d9792cbdfd12e7907
```

### Benchmark

```bash
python scripts/run_ulb_benchmark.py --json-out outputs/reports/ulb_benchmark_sample.json
```

Protocol: sort by `Time`, fit on the first 70 % and score the last 30 %, fit the scaler on the
fit window only, and derive the threshold from fit scores. `Class` is removed before training.
`contamination` is set to the fit-window fraud prevalence (2.74 %).

| Detector | PR-AUC | ROC-AUC | Precision | Recall | P@top-1 % | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|
| Histogram-Based Outlier Score | **0.3765** | 0.9343 | 0.3589 | 0.6944 | 0.5000 | 0.02 |
| Isolation Forest | 0.3108 | 0.9340 | 0.3867 | 0.5370 | 0.4167 | 0.28 |
| Elliptic Envelope | 0.1983 | 0.9028 | 0.1856 | 0.1667 | 0.1333 | 1.96 |
| k-Nearest Neighbors | 0.1754 | 0.9147 | 0.2081 | 0.2870 | 0.1333 | 0.20 |
| One-Class SVM | 0.1543 | 0.8792 | 0.1860 | 0.4167 | 0.2667 | 0.70 |
| PCA Reconstruction | 0.1528 | 0.8414 | 0.2035 | 0.3241 | 0.3000 | 0.01 |
| Gaussian Mixture Model | 0.1510 | 0.8281 | 0.2360 | 0.3889 | 0.1500 | 0.75 |
| Local Outlier Factor | 0.0510 | 0.6743 | 0.0844 | 0.1852 | 0.1667 | 0.21 |
| Consensus ensemble (rank Borda, 8 models) | 0.1634 | 0.9033 | — | — | 0.3000 | — |

Chance-level PR-AUC on the holdout is ≈ 0.018, so every detector except LOF clears it
substantially. **HBOS wins** on this data because fraud signal is concentrated in a few
marginal distributions (`V14`, `V17`, `V12`, `Amount`) and HBOS scores marginals independently.

### The split matters more than the model

The same `Isolation Forest` scores **PR-AUC 0.3108** under the time split above but
**PR-AUC 0.6471 / ROC-AUC 0.9426** under the API's default stratified random split
(`GET /api/evaluation/holdout`). Both are correctly computed; the random split lets the model
see frauds from the same two days during fit, which is not how a deployed detector operates.
Quote the time-split figure when the claim is about detection, and the random-split figure only
when comparing against papers that used that protocol.

### Caveats to state in any write-up

1. `V1…V28` are PCA components — the model cannot explain *why* in business terms, only which
   components moved. This is a dataset limit, not a framework limit.
2. Only 492 positives, over two days. Weekly seasonality and concept drift cannot be measured.
3. No account, merchant, device, or country fields, so account-level velocity features are
   impossible. For those, the Berka/PKDD'99 banking database is the complement.
4. Label noise: "normal" means *not charged back*, not *verified legitimate*.
5. Time-based evaluation is the honest default. Random splits inflate PR-AUC here by ~2×.

## Not included

No public transaction-level Indian banking dataset (RBI, NPCI, SBI, HDFC, ICICI, Axis) exists.
RBI and NPCI publish only aggregate statistics, and bank developer portals expose synthetic
sandbox payloads. Any "Indian bank transactions" file on Kaggle should be treated as synthetic
unless the publisher documents a named institution, period, and release agreement.

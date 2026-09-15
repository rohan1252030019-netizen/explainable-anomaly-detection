# Explainable Framework for Unsupervised Anomaly Detection

An end-to-end, production-ready machine learning framework and interactive dashboard designed to detect anomalies in tabular data without supervision and provide actionable model interpretability through SHAP and feature contribution analysis.

---

## Key Features

- **Unsupervised Detection Algorithms**:
  - **Isolation Forest**: Tree-based partition isolation for anomaly scoring.
  - **Local Outlier Factor (LOF)**: Density-based local neighborhood anomaly detection.
  - **One-Class SVM**: Support vector boundary estimation for high-dimensional data.
  - **Elliptic Envelope**: Robust covariance estimation for normally distributed data.
- **Model Explainability & Interpretability**:
  - **SHAP (SHapley Additive exPlanations)**: TreeExplainer and KernelExplainer integration.
  - **Feature Contributions**: Local feature-level deviations and contribution percentages.
  - **Natural Language Explanations**: Automated rule-based text justifications for detected anomalies.
- **Scoring & Ranking**:
  - Score normalization, quantile/statistical thresholding, and severity ranking.
- **Interactive Full-Stack Architecture**:
  - **Frontend**: Next.js (TypeScript + Tailwind CSS) interactive analytics dashboard.
  - **Backend**: FastAPI + Uvicorn REST API with modular routers.
  - **ML Core**: Modular, extensible Python library (`src/`).

---

## Project Architecture

```
explainable-anomaly-detection/
├── backend/                  # FastAPI backend server & routers
│   ├── routers/              # Dataset, models, detection, explainability, reports
│   ├── main.py               # API entry point
│   ├── schemas.py            # Pydantic request/response schemas
│   └── state.py              # In-memory pipeline session state
├── frontend/                 # Next.js frontend web application
│   ├── src/app/              # Next.js App Router pages
│   ├── src/components/       # UI & visualization components
│   └── src/lib/              # API clients and utilities
├── src/                      # ML Core modules
│   ├── data/                 # Loading, validation, and preprocessing pipelines
│   ├── detection/            # Scoring, thresholding, and anomaly ranking
│   ├── evaluation/           # Metrics (ROC-AUC, PR-AUC) and model comparison
│   ├── explainability/       # SHAP and feature contribution explainers
│   ├── models/               # Unsupervised detector implementations
│   └── visualization/        # Plotting and charting utilities
├── tests/                    # Unit and integration test suite
├── run.py                    # Unified CLI orchestrator
└── requirements.txt          # Python dependencies
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11 & 3.13)
- Node.js 18+ and npm

### 2. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/rohan1252030019-netizen/explainable-anomaly-detection.git
cd explainable-anomaly-detection

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

### Launch FastAPI Backend
```bash
python run.py --backend
```
The API server will run at: `http://localhost:8000`  
Swagger API Documentation: `http://localhost:8000/docs`

### Launch Next.js Dashboard
```bash
cd frontend
npm run dev
```
The web dashboard will run at: `http://localhost:3000`

### Run End-to-End Pipeline Demo
```bash
python run.py --demo
```
This runs a complete unsupervised anomaly detection pipeline on synthetic data, fits an Isolation Forest, computes scores, and exports results to `outputs/predictions/demo_results.csv`.

### Run Test Suite
```bash
python run.py --test
```

---

## License
MIT License

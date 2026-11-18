# CHANGELOG

All notable changes to the SENTINEL platform are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2026-11-18 — Rebrand to SENTINEL

### Major Rebrand: XAUD 2.0 ➔ SENTINEL 3.0
- **Platform Identity**: Rebranded as **SENTINEL — Autonomous Anomaly Intelligence Platform**.
- **Backronym**: *Scientific, Explainable, Next-gen Threat & Irregularity Neutralization and Evaluation Layer*.
- **Tagline**: *"Detect. Explain. Verify. Act."*
- **Package Name**: Registered as `sentinel-anomaly` in `pyproject.toml` with `sentinel` module compatibility shim.

### Added
- **SENTINEL_* Environment Variables**: Primary configuration namespace (`SENTINEL_HOST`, `SENTINEL_PORT`, `SENTINEL_API_KEY`, `SENTINEL_CORS_ORIGINS`, etc.). Legacy `XAUD_*` environment variables remain supported as deprecated fallbacks.
- **Wordmark & Logo**: High-resolution vector SVG wordmarks in `docs/sentinel-logo.svg` and `frontend/public/sentinel-logo.svg`.
- **Packaging Extras**: Declarations for optional heavy dependencies (`boost`, `deep`, `optuna`, `full`) enabling lightweight base installs.
- **GitHub Topics Documentation**: Cataloged in `docs/GITHUB_TOPICS.md`.

### Changed
- **API Metadata**: Root route, OpenAPI contract, and health probe re-centered around SENTINEL 3.0.
- **Frontend Console**: Brand marks, headers, report outputs, and metadata transitioned to SENTINEL 3.0.
- **Export Filenames**: Default CSV and JSON export attachments updated to `sentinel_predictions.csv`, `sentinel_anomalies_only.csv`, and `sentinel_explanations.json`.
- **Documentation**: Corrected repository clone URL to the official repository target.

---

## [2.1.0] - 2026-10-02
- Initial XAUD 2.0 research-grade unsupervised tabular platform release.
- 8 unsupervised anomaly detectors, Rank Borda count ensemble, synthetic anomaly benchmark lab, and constrained physical-unit counterfactuals.

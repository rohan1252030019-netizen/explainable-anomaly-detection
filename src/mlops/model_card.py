"""
src/mlops/model_card.py

Automated Model Card Generator for SENTINEL Anomaly Intelligence (Mitchell et al., 2019).

Generates standard, audit-ready Model Cards (JSON and Markdown) documenting:
1. Model Details & Architecture
2. Intended Domain & Operational Use
3. Data Provenance & Cryptographic Hashes
4. Quantitative Evaluation & Conformal Coverage
5. Subgroup Fairness & Robustness Audits
6. Technical Caveats & Non-Causal Disclaimers
"""

import datetime
import hashlib
import json
from typing import Any


def generate_model_card(
    model_name: str,
    version: str,
    dataset_name: str,
    dataset_hash: str,
    hyperparameters: dict[str, Any],
    feature_names: list[str],
    metrics: dict[str, Any],
    fairness_metrics: dict[str, Any] | None = None,
    robustness_metrics: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
    author: str = "SENTINEL Platform Automated Auditor",
) -> dict[str, Any]:
    """
    Generate structured Model Card dict and Markdown documentation.

    Returns
    -------
    dict containing:
        - model_card_data: dict
        - markdown: str
        - sha256_hash: str
    """
    default_limitations = [
        "Unsupervised anomaly scores represent relative distribution ranking indices, not posterior probabilities.",
        "Model attributions reflect associative model sensitivity, NOT ground-truth physical causation.",
        "Performance guarantees (conformal p-values) depend strictly on the exchangeability of calibration and test data.",
        "Extreme out-of-distribution domain shift requires re-calibration and human review.",
    ]
    lims = limitations or default_limitations

    card_data = {
        "model_details": {
            "name": model_name,
            "version": version,
            "architecture": model_name,
            "license": "Apache-2.0",
            "framework": "SENTINEL 3.0",
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "author": author,
        },
        "intended_use": {
            "primary_uses": [
                "Unsupervised tabular anomaly detection",
                "Explainable threat identification",
                "Human-in-the-loop triage prioritization",
            ],
            "out_of_scope_uses": [
                "Autonomous unmonitored decision making without human signoff",
                "Causal liability determination",
            ],
        },
        "data_provenance": {
            "dataset_name": dataset_name,
            "dataset_sha256": dataset_hash,
            "n_features": len(feature_names),
            "features": feature_names,
        },
        "hyperparameters": hyperparameters,
        "performance_and_validation": metrics,
        "fairness_audit": fairness_metrics or {},
        "robustness_audit": robustness_metrics or {},
        "limitations_and_caveats": lims,
    }

    # Deterministic fingerprint
    card_json = json.dumps(card_data, sort_keys=True)
    card_hash = hashlib.sha256(card_json.encode("utf-8")).hexdigest()
    card_data["card_hash"] = card_hash

    # Generate Markdown representation
    md_lines = [
        f"# Model Card: {model_name} (v{version})",
        "",
        f"**Card Hash**: `{card_hash}`  ",
        f"**Generated**: `{card_data['model_details']['created_at']}`  ",
        "**Framework**: `SENTINEL 3.0 Autonomous Anomaly Intelligence`  ",
        "",
        "## 1. Model Details",
        f"- **Model Identifier**: {model_name}",
        f"- **Version**: {version}",
        f"- **Author / Pipeline**: {author}",
        "- **License**: Apache-2.0",
        "",
        "## 2. Intended Use",
        "### In-Scope:",
        "\n".join([f"- {u}" for u in card_data["intended_use"]["primary_uses"]]),
        "",
        "### Out-of-Scope:",
        "\n".join([f"- {u}" for u in card_data["intended_use"]["out_of_scope_uses"]]),
        "",
        "## 3. Data Provenance",
        f"- **Training Dataset**: `{dataset_name}`",
        f"- **Dataset SHA-256**: `{dataset_hash}`",
        f"- **Feature Count**: {len(feature_names)}",
        f"- **Features**: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}",
        "",
        "## 4. Hyperparameters",
        "```json",
        json.dumps(hyperparameters, indent=2),
        "```",
        "",
        "## 5. Performance & Metrics",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]

    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)):
            md_lines.append(f"| {k} | {v} |")

    md_lines.extend([
        "",
        "## 6. Technical Limitations & Governance",
        "\n".join([f"> [!WARNING]\n> {lim}" for lim in lims]),
        "",
        "---",
        "*SENTINEL Model Governance Standard (Mitchell et al., 2019)*",
    ])

    markdown = "\n".join(md_lines)

    return {
        "model_card_data": card_data,
        "markdown": markdown,
        "sha256_hash": card_hash,
    }

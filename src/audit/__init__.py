"""
src/audit package

SENTINEL Root-Cause & Robustness Auditing Engine:
- Root-Cause Attribution Clustering (HDBSCAN on SHAP/LIME Vectors)
- Subgroup Fairness & Disparate Impact Equity Auditing
- Adversarial & Sensory Noise Perturbation Sensitivity
"""

from src.audit.fairness_robustness import (
    evaluate_noise_robustness,
    evaluate_subgroup_fairness,
)
from src.audit.root_cause import cluster_root_causes

__all__ = [
    "cluster_root_causes",
    "evaluate_subgroup_fairness",
    "evaluate_noise_robustness",
]

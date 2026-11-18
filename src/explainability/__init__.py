"""
src/explainability package

SENTINEL Advanced Explainability Engine:
- SHAP (TreeSHAP, KernelSHAP, PermutationSHAP)
- LIME Explainer & Cross-method Agreement (Spearman rho, Top-K Jaccard)
- Feature Interactions (TreeSHAP interactions, 2nd-order perturbation)
- Cohort Analysis & Subgroup Distribution Shifts
- Diverse Counterfactuals (DiCE-style with plausibility & sparsity constraints)
- Forensic Investigation Narratives
"""

from src.explainability.cohort import explain_cohort
from src.explainability.counterfactuals import nearest_normal_counterfactual
from src.explainability.diverse_counterfactuals import generate_diverse_counterfactuals
from src.explainability.explanation_generator import ExplanationGenerator
from src.explainability.interactions import compute_pairwise_interactions
from src.explainability.lime_explainer import LIMEExplainer, compute_explanation_agreement
from src.explainability.narrative import generate_investigation_narrative
from src.explainability.shap_explainer import SHAPExplainer

__all__ = [
    "SHAPExplainer",
    "ExplanationGenerator",
    "nearest_normal_counterfactual",
    "generate_diverse_counterfactuals",
    "LIMEExplainer",
    "compute_explanation_agreement",
    "compute_pairwise_interactions",
    "explain_cohort",
    "generate_investigation_narrative",
]

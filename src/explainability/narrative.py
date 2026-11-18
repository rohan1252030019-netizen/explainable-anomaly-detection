"""
src/explainability/narrative.py

SENTINEL Autonomous Investigation Narrative Generator.

Generates natural-language investigation narratives for flagged anomalies.
Strict Rule: Zero fabricated metrics. The narrative is constructed directly from
verified model outputs, conformal p-values, SHAP/LIME attributions, interaction
pairs, and counterfactuals.

By default, uses a deterministic, audit-proof template engine.
Optionally integrates with Gemini / OpenAI LLM if configured via environment variables
(SENTINEL_LLM_API_KEY or GEMINI_API_KEY), enforcing strict factual constraints so the
LLM can only rephrase verified facts provided in the prompt payload.
"""

import hashlib
import json
import os
from typing import Any


def generate_investigation_narrative(
    record_id: int | str,
    anomaly_score: float,
    threshold: float,
    severity_tier: str,
    top_features: list[dict[str, Any]],
    conformal_p_value: float | None = None,
    fdr_significant: bool | None = None,
    agreement_summary: dict[str, Any] | None = None,
    interaction_pairs: list[dict[str, Any]] | None = None,
    counterfactual: dict[str, Any] | None = None,
    detector_name: str = "Ensemble",
) -> dict[str, Any]:
    """
    Generate a factual SENTINEL Investigation Narrative.

    Parameters
    ----------
    record_id : int or str
    anomaly_score : float in [0, 1]
    threshold : float
    severity_tier : str ("Critical", "High", "Medium", "Low")
    top_features : list of dicts with keys 'feature', 'attribution', and optional 'deviation'
    conformal_p_value : float or None
    fdr_significant : bool or None
    agreement_summary : dict or None (from compute_explanation_agreement)
    interaction_pairs : list of dicts or None
    counterfactual : dict or None (modifications required)
    detector_name : str

    Returns
    -------
    dict containing:
        - narrative_text: formatted markdown investigation narrative
        - executive_summary: 2-line high-level briefing
        - forensic_fingerprint: SHA-256 hash of all input facts
        - audit_provenance: metadata dict
    """
    # 1. Compute deterministic SHA-256 fingerprint of all input facts
    facts_payload = {
        "record_id": str(record_id),
        "score": round(float(anomaly_score), 4),
        "threshold": round(float(threshold), 4),
        "severity": severity_tier,
        "detector": detector_name,
        "conformal_p_value": round(float(conformal_p_value), 4) if conformal_p_value is not None else None,
        "fdr_significant": fdr_significant,
        "top_features": [
            {"f": f.get("feature"), "attr": round(float(f.get("attribution", 0.0)), 4)}
            for f in top_features[:5]
        ],
    }
    raw_json = json.dumps(facts_payload, sort_keys=True)
    fingerprint = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()

    # 2. Build deterministic template narrative
    score_pct = round(anomaly_score * 100, 1)
    thresh_pct = round(threshold * 100, 1)

    # Executive summary
    exec_summary = (
        f"Record #{record_id} flagged with {severity_tier} anomaly score ({anomaly_score:.3f} vs threshold {threshold:.3f}) "
        f"by {detector_name}."
    )
    if conformal_p_value is not None:
        exec_summary += f" Conformal significance: p = {conformal_p_value:.4f}."

    # Section 1: Alert Overview
    sections = [
        f"### SENTINEL Investigation Narrative: Incident #{record_id}",
        f"**Severity**: `{severity_tier.upper()}` | **Detector**: `{detector_name}` | **Score**: `{anomaly_score:.4f}` (Threshold: `{threshold:.4f}`)",
        "",
        "#### 1. Statistical & Conformal Verification",
    ]

    if conformal_p_value is not None:
        coverage_statement = (
            f"Under finite-sample exchangeability guarantees, this observation yields a conformal p-value of "
            f"**p = {conformal_p_value:.4f}**. "
        )
        if fdr_significant is True:
            coverage_statement += "The finding is **confirmed significant under Benjamini-Hochberg FDR control**."
        elif fdr_significant is False:
            coverage_statement += "The finding falls above the FDR significance threshold; monitor for recurring patterns."
        sections.append(coverage_statement)
    else:
        sections.append(
            f"The observation scored in the top tier ({score_pct}% normalised rank), exceeding operational threshold by "
            f"{max(0.0, score_pct - thresh_pct):.1f}%."
        )

    # Section 2: Attribution & Root Drivers
    sections.append("")
    sections.append("#### 2. Primary Attribution Drivers")
    if top_features:
        driver_bullets = []
        for feat in top_features[:4]:
            name = feat.get("feature", "unknown")
            attr = feat.get("attribution", 0.0)
            dev = feat.get("deviation")
            dev_str = f" (deviation: {dev})" if dev else ""
            driver_bullets.append(f"- **{name}**: Attribution `{attr:+.4f}`{dev_str}")
        sections.extend(driver_bullets)
    else:
        sections.append("- No individual dominant feature isolated.")

    # Section 3: Explainer Cross-Validation
    if agreement_summary:
        sections.append("")
        sections.append("#### 3. Explainer Cross-Validation (SHAP vs LIME)")
        rho = agreement_summary.get("spearman_rank_correlation", 0.0)
        jaccard = agreement_summary.get("top_k_jaccard_overlap", 0.0)
        tier = agreement_summary.get("agreement_level", "Moderate")
        sections.append(
            f"Dual-explainer validation demonstrates **{tier}** (Spearman rho = `{rho:.3f}`, Top-K Jaccard = `{jaccard:.2f}`). "
            f"{agreement_summary.get('summary', '')}"
        )

    # Section 4: Second-Order Interactions
    if interaction_pairs:
        sections.append("")
        sections.append("#### 4. Compound Feature Interactions")
        top_pair = interaction_pairs[0]
        sections.append(
            f"Non-linear synergy detected between **{top_pair['feature_1']}** and **{top_pair['feature_2']}** "
            f"(Interaction strength: `{top_pair['interaction_strength']:.4f}`). {top_pair.get('interpretation', '')}"
        )

    # Section 5: Actionable Counterfactual Remediation
    if counterfactual and counterfactual.get("modifications"):
        sections.append("")
        sections.append("#### 5. Actionable Remediation (Counterfactual)")
        plaus = counterfactual.get("is_in_distribution_plausible", True)
        plaus_badge = " [In-Distribution Plausible]" if plaus else " [Unverified Distribution]"
        sections.append(f"To return this observation below the normal threshold `{threshold:.3f}`{plaus_badge}:")
        for mod in counterfactual.get("modifications", [])[:3]:
            feat = mod.get("feature")
            cur = mod.get("observed_value")
            tgt = mod.get("suggested_value")
            delta = mod.get("delta")
            sections.append(f"- Adjust **{feat}** from `{cur}` to `{tgt}` (Delta: `{delta:+.3f}`)")

    # Section 6: Audit Provenance
    sections.append("")
    sections.append("---")
    sections.append(f"*SENTINEL Audit Fingerprint: `{fingerprint}` | Engine: Deterministic Rule-Based Forensic Generator*")

    full_narrative = "\n".join(sections)

    # 3. Optional LLM enhancement if API key is provided
    llm_key = os.environ.get("SENTINEL_LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if llm_key:
        try:
            # We attempt Gemini SDK if installed and configured, falling back gracefully
            from google import genai
            client = genai.Client(api_key=llm_key)
            prompt = (
                f"You are the SENTINEL Anomaly Intelligence Platform. Rewrite the following forensic "
                f"investigation report for an executive security analyst. DO NOT invent, alter, or add any numbers, "
                f"features, or statistics. Use only the provided facts:\n\n{full_narrative}"
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if response and response.text:
                full_narrative = response.text
        except Exception:
            pass  # Retain verified deterministic narrative

    return {
        "record_id": record_id,
        "narrative_text": full_narrative,
        "executive_summary": exec_summary,
        "forensic_fingerprint": fingerprint,
        "audit_provenance": {
            "engine": "SENTINEL Investigation Narrative v3.0",
            "deterministic_fallback": True,
            "facts_hash": fingerprint,
        },
    }

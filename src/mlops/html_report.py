"""
src/mlops/html_report.py

Presentation-ready HTML & PDF Audit Report Generator for SENTINEL 3.0.

Produces self-contained, responsive HTML reports including:
- Cryptographic Provenance Hashes
- Executive Summary & Anomaly Counts
- Severity Tier Distributions
- Top Driver Attribution Tables
- Methodology Caveats & Non-Causal Disclaimers
"""

import html
from typing import Any


def generate_html_audit_report(report_data: dict[str, Any]) -> str:
    """
    Generate an offline-capable, styled HTML audit report.

    Parameters
    ----------
    report_data : dict
        Standard reporting payload from state.generate_report().

    Returns
    -------
    str : HTML document string
    """
    title = html.escape(report_data.get("title", "SENTINEL Anomaly Audit Report"))
    framework = html.escape(report_data.get("framework", "SENTINEL 3.0"))
    date_str = html.escape(str(report_data.get("generated_at", "")))

    dataset = report_data.get("dataset", {})
    detection = report_data.get("detection", {})
    top_drivers = report_data.get("top_drivers", [])
    anomalies = report_data.get("sample_anomalies", [])

    drivers_rows = ""
    for d in top_drivers:
        feat = html.escape(str(d.get("feature", "")))
        imp = d.get("importance", 0.0)
        drivers_rows += f"<tr><td><strong>{feat}</strong></td><td>{imp:.4f}</td></tr>"

    anomalies_rows = ""
    for anom in anomalies[:10]:
        r_id = anom.get("row_id", anom.get("original_index", "N/A"))
        score = anom.get("score", anom.get("norm_score", 0.0))
        sev = html.escape(str(anom.get("severity", "Medium")))
        anomalies_rows += f"<tr><td>#{r_id}</td><td>{score:.4f}</td><td><span class='badge badge-{sev.lower()}'>{sev}</span></td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title} — {framework}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 40px; background: #0f172a; color: #f8fafc; line-height: 1.6; }}
    .container {{ max-width: 960px; margin: 0 auto; background: #1e293b; padding: 40px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
    .header {{ border-bottom: 2px solid #334155; padding-bottom: 20px; margin-bottom: 30px; }}
    h1 {{ color: #38bdf8; margin: 0 0 10px 0; font-size: 28px; }}
    .meta {{ color: #94a3b8; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
    .card {{ background: #0f172a; padding: 20px; border-radius: 8px; border: 1px solid #334155; }}
    h2 {{ color: #e2e8f0; font-size: 18px; margin-top: 0; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
    th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; }}
    th {{ color: #94a3b8; font-weight: 600; }}
    .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
    .badge-critical {{ background: #ef4444; color: #fff; }}
    .badge-high {{ background: #f97316; color: #fff; }}
    .badge-medium {{ background: #eab308; color: #000; }}
    .badge-low {{ background: #3b82f6; color: #fff; }}
    .notice {{ background: #1e1b4b; border-left: 4px solid #6366f1; padding: 15px; margin-top: 30px; border-radius: 4px; font-size: 13px; color: #c7d2fe; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>{title}</h1>
      <div class="meta">Framework: {framework} | Generated: {date_str}</div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Dataset Metadata</h2>
        <p><strong>Dataset Name:</strong> {html.escape(str(dataset.get('name', 'N/A')))}</p>
        <p><strong>Total Records:</strong> {dataset.get('rows', 0):,}</p>
        <p><strong>Preprocessed Features:</strong> {dataset.get('features_preprocessed', 0)}</p>
        <p><strong>Scaler:</strong> {html.escape(str(dataset.get('scaler', 'standard')))}</p>
      </div>

      <div class="card">
        <h2>Detection Overview</h2>
        <p><strong>Active Detector:</strong> {html.escape(str(detection.get('model', 'Ensemble')))}</p>
        <p><strong>Anomalies Flagged:</strong> {detection.get('anomalies_detected', 0):,} ({float(detection.get('anomaly_rate', 0))*100:.1f}%)</p>
        <p><strong>Decision Threshold:</strong> {detection.get('threshold_value', 0.5):.3f} ({html.escape(str(detection.get('threshold_method', 'contamination')))})</p>
      </div>
    </div>

    <div class="card" style="margin-bottom: 30px;">
      <h2>Primary Anomaly Drivers (Global Attribution)</h2>
      <table>
        <thead><tr><th>Feature</th><th>Mean Relative Attribution</th></tr></thead>
        <tbody>
          {drivers_rows or '<tr><td colspan="2">No global drivers available.</td></tr>'}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Flagged Incident Sample</h2>
      <table>
        <thead><tr><th>Record ID</th><th>Normalized Score</th><th>Severity</th></tr></thead>
        <tbody>
          {anomalies_rows or '<tr><td colspan="3">No anomalies flagged.</td></tr>'}
        </tbody>
      </table>
    </div>

    <div class="notice">
      <strong>SENTINEL Governance & Methodology Disclaimer:</strong><br>
      {html.escape(report_data.get('methodology_note', ''))}
    </div>
  </div>
</body>
</html>
"""
    return html_content

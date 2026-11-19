import os
import sys
import copy
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def update_presentation(src_pptx: str, dest_pptx: str):
    # Load from pristine original template
    prs = Presentation(src_pptx)
    
    # =========================================================================
    # PALETTE DEFINITIONS (Strict contrast hierarchy)
    # =========================================================================
    # Slide 1 Dark Theme Palette (Background is #130E24)
    S1_TITLE = RGBColor(0xFF, 0xFF, 0xFF)       # Pure white
    S1_SUBTITLE = RGBColor(0xE4, 0xDC, 0xF3)    # Pale lavender
    S1_AMBER = RGBColor(0xFD, 0xE6, 0x8A)       # Light warm amber
    S1_MUTED = RGBColor(0xBD, 0xB2, 0xD6)       # Soft light purple
    
    # Slides 2-13 Light Theme Palette (Slide Canvas is pure white #FFFFFF)
    CLR_TITLE = RGBColor(0x1B, 0x12, 0x2E)      # Deep charcoal violet (Titles - Georgia 28pt bold)
    CLR_SUBTITLE = RGBColor(0x7D, 0x74, 0x93)   # Slate purple (Subtitles, Meta, Footers - Segoe UI 11.5pt)
    CLR_FOOTER = RGBColor(0x7D, 0x74, 0x93)     # Slate purple (Footers & page numbers - Segoe UI 8.5pt)
    CLR_PILL = RGBColor(0xB4, 0x53, 0x09)       # Warm amber / ochre (Section Pills & markers - Segoe UI 9.5pt)
    CLR_HEAVY = RGBColor(0x1B, 0x12, 0x2E)      # Deep charcoal violet (Bullet titles, metric numbers)
    CLR_BODY_DARK = RGBColor(0x3B, 0x34, 0x4F)  # Dark charcoal violet (Body paragraphs, explanations)
    CLR_STAT_NUM = RGBColor(0x1B, 0x12, 0x2E)   # Deep charcoal violet (Large stat numbers - Georgia 20-32pt)
    CLR_STAT_LABEL = RGBColor(0x7D, 0x74, 0x93) # Slate purple (Stat tile labels)
    CLR_ACCENT = RGBColor(0x6D, 0x28, 0xD9)     # Rich vivid violet (Feature tags, callout headings)
    CLR_GREEN = RGBColor(0x05, 0x96, 0x69)      # Emerald green (Resolved / Implemented badges)
    CLR_RED = RGBColor(0xB3, 0x1D, 0x2E)        # Crimson red (Limitations / Disclosed badges)
    
    # Tables
    CLR_TBL_HEADER = RGBColor(0xFF, 0xFF, 0xFF) # Table headers have dark fill, text is white
    CLR_TBL_CELL = RGBColor(0x3B, 0x34, 0x4F)   # Table data cells have white fill, text is dark charcoal
    
    # Slide 12 Bottom Dark Container Box
    CLR_S12_BOX_TITLE = RGBColor(0xFF, 0xFF, 0xFF)
    CLR_S12_BOX_AMBER = RGBColor(0xFD, 0xE6, 0x8A)
    CLR_S12_BOX_MUTED = RGBColor(0xBD, 0xB2, 0xD6)

    # =========================================================================
    # TEXT HELPER FUNCTIONS
    # =========================================================================
    def set_run_text(p, text, font_name="Segoe UI", font_size=Pt(11), color_rgb=CLR_BODY_DARK, bold=False):
        p.text = text
        if p.runs:
            r = p.runs[0]
            r.font.name = font_name
            r.font.size = font_size
            r.font.color.rgb = color_rgb
            r.font.bold = bold

    def set_shape_text(shape, text, font_name="Segoe UI", font_size=Pt(11), color_rgb=CLR_BODY_DARK, bold=False):
        if not shape.has_text_frame:
            return
        tf = shape.text_frame
        if not tf.paragraphs:
            p = tf.add_paragraph()
        else:
            p = tf.paragraphs[0]
        set_run_text(p, text, font_name, font_size, color_rgb, bold)

    def set_bullet_paragraph(p, marker, title, body, font_name="Segoe UI", font_size=Pt(11)):
        p.text = ""
        r0 = p.add_run()
        r0.text = marker + "  "
        r0.font.name = font_name
        r0.font.size = font_size
        r0.font.color.rgb = CLR_PILL
        r0.font.bold = False

        if title:
            r1 = p.add_run()
            r1.text = title + " "
            r1.font.name = font_name
            r1.font.size = font_size
            r1.font.color.rgb = CLR_HEAVY
            r1.font.bold = True

        if body:
            r2 = p.add_run()
            r2.text = body
            r2.font.name = font_name
            r2.font.size = font_size
            r2.font.color.rgb = CLR_BODY_DARK
            r2.font.bold = False

    # =========================================================================
    # GLOBAL FOOTERS & SLIDE NUMBERS
    # =========================================================================
    for idx, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if shape.has_text_frame:
                txt = shape.text_frame.text.strip()
                if "XAUD 2.0  ·  Explainable Anomaly Detection Framework" in txt:
                    col = S1_MUTED if idx == 1 else CLR_FOOTER
                    set_shape_text(shape, "SENTINEL 3.0  ·  Autonomous Anomaly Intelligence Platform", "Segoe UI", Pt(8.5), col)
                elif "Explainable Anomaly Detection Framework (XAUD 2.0)" in txt:
                    set_shape_text(shape, "SENTINEL 3.0  ·  Autonomous Anomaly Intelligence Platform  ·  Apache 2.0  ·  Python 3.11+ / FastAPI / PyTorch / Next.js 16", "Segoe UI", Pt(8.5), CLR_FOOTER)
                if txt.endswith("/ 12") and len(txt) <= 7:
                    new_page_str = f"{idx:02d} / 13"
                    col = S1_MUTED if idx == 1 else CLR_FOOTER
                    set_shape_text(shape, new_page_str, "Segoe UI", Pt(8.5), col)

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Background)
    # =========================================================================
    s1 = prs.slides[0]
    set_shape_text(s1.shapes[34], "SENTINEL — ANOMALY INTELLIGENCE PLATFORM", "Segoe UI", Pt(10), S1_AMBER, bold=True)
    set_shape_text(s1.shapes[35], "SENTINEL", "Georgia", Pt(84), S1_TITLE, bold=True)
    set_shape_text(s1.shapes[36], "Autonomous Anomaly Intelligence Platform", "Segoe UI", Pt(19), S1_SUBTITLE)
    set_shape_text(s1.shapes[37], "“Detect. Explain. Verify. Act.”", "Segoe UI", Pt(13.5), S1_AMBER)
    set_shape_text(s1.shapes[45], "19 November 2026", "Segoe UI", Pt(12), S1_TITLE)
    set_shape_text(s1.shapes[48], "Python 3.11+  ·  FastAPI  ·  scikit-learn  ·  PyTorch  ·  XGBoost  ·  SHAP  ·  Next.js 16  ·  Tailwind CSS", "Segoe UI", Pt(8.8), S1_MUTED)

    s1.notes_slide.notes_text_frame.text = (
        "Good morning. Today I am presenting SENTINEL 3.0, an Autonomous Anomaly Intelligence Platform for tabular data and streaming environments.\n\n"
        "Our core philosophy is captured in our tagline: 'Detect. Explain. Verify. Act.' SENTINEL does not merely output opaque outlier flags; "
        "it produces mathematically calibrated conformal error bounds, verifiable feature attributions, and actionable counterfactual recommendations.\n\n"
        "Every metric, algorithmic benchmark, and performance claim in this presentation is directly backed by reproducible, committed artifacts in our repository."
    )

    # =========================================================================
    # SLIDE 2: Problem Statement (Light Background)
    # =========================================================================
    s2 = prs.slides[1]
    set_shape_text(s2.shapes[3], "01 · PROBLEM STATEMENT", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s2.shapes[4], "Anomalies without a witness", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s2.shapes[5], "Unsupervised detection is easy to run, but dangerous to trust without proof", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    bullets2 = [
        ("▪", "Rare and unlabeled.", "Anomalies comprise <2% of rows, leaving practitioners with zero ground truth during fit."),
        ("▪", "Scores sold as probabilities.", "Unsupervised scores are relative ranks, not posterior probabilities P(anomaly|x)."),
        ("▪", "Silent calibration collapse.", "Arbitrary thresholds fail when production data distributions experience concept drift."),
        ("▪", "Unexplained explanations.", "Popular SHAP visualisations are displayed without testing mathematical faithfulness."),
        ("▪", "Absence of audit lineage.", "Traditional pipelines lack cryptographic fingerprinting and verifiable model cards.")
    ]
    tf2 = s2.shapes[11].text_frame
    for i, p in enumerate(tf2.paragraphs):
        if i < len(bullets2):
            marker, header, body = bullets2[i]
            set_bullet_paragraph(p, marker, header, body, "Segoe UI", Pt(11))

    set_shape_text(s2.shapes[12], "STANDARD TOOL  vs  SENTINEL 3.0", "Segoe UI", Pt(8.5), CLR_PILL)
    tbl2 = s2.shapes[13].table
    tbl2.cell(0, 2).text = "SENTINEL 3.0"
    tbl_data2 = [
        ("Score meaning", "Arbitrary probability claim", "Relative rank + Conformal p-value"),
        ("Data leakage", "Scaler fit across full table", "Strict train-only fit, holdout transform"),
        ("Ensembling", "Simple mean of uncalibrated scores", "Rank Borda count & Hybrid XGBoost stacking"),
        ("Attribution audit", "Visual plots, never measured", "TreeSHAP, 2nd-order interactions, Deletion-AUC"),
        ("Operational safety", "Static unmonitored scripts", "Streaming Half-Space Trees & PSI drift gates")
    ]
    for r_idx, row in enumerate(tbl_data2, start=1):
        c0, c1, c2 = tbl2.cell(r_idx, 0), tbl2.cell(r_idx, 1), tbl2.cell(r_idx, 2)
        c0.text, c1.text, c2.text = row[0], row[1], row[2]
        for c in (c0, c1, c2):
            for p in c.text_frame.paragraphs:
                p.font.name = "Segoe UI"
                p.font.size = Pt(9.5)
                p.font.color.rgb = CLR_TBL_CELL

    set_shape_text(s2.shapes[18], "It is not about raw accuracy — it is about evidence. SENTINEL is engineered so that every alert traces back to an honest split, a calibrated error rate, and a cryptographic fingerprint.", "Segoe UI", Pt(11), CLR_HEAVY)

    s2.notes_slide.notes_text_frame.text = (
        "In production anomaly detection, the biggest risk is not failing to find an anomaly; it is acting on an unsubstantiated alert.\n\n"
        "Most open-source tools commit three critical errors: they market raw outlier indices as 'probabilities', they leak future data by fitting scalers globally, "
        "and they display SHAP charts without checking whether the attribution is faithful to the model.\n\n"
        "As shown in our comparison table, SENTINEL replaces arbitrary heuristics with finite-sample conformal calibration, time-aware validation, and quantitative explanation audits."
    )

    # =========================================================================
    # SLIDE 3: Objectives & Scope (Light Background)
    # =========================================================================
    s3 = prs.slides[2]
    set_shape_text(s3.shapes[3], "02 · OBJECTIVES & SCOPE", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s3.shapes[4], "What SENTINEL sets out to do", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s3.shapes[5], "Detect → calibrate → explain → monitor → triage → reproduce", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    bullets3 = [
        ("▪", "Unsupervised & hybrid zoo.", "20 detectors across 5 statistical families — from tree ensembles to deep VAEs."),
        ("▪", "Conformal false discovery control.", "Finite-sample tail p-values with Benjamini-Hochberg error budgeting."),
        ("▪", "Multi-angle explainability.", "TreeSHAP, 2nd-order interactions, LOF counterfactuals, and narrative reports."),
        ("▪", "Streaming & drift gates.", "Half-Space Trees with continuous sliding-window PSI monitoring and recalibration."),
        ("▪", "Full MLOps lifecycle.", "Automated Mitchell et al. model cards, active learning queues, and SHA-256 lineage.")
    ]
    tf3 = s3.shapes[11].text_frame
    for i, p in enumerate(tf3.paragraphs):
        if i < len(bullets3):
            marker, header, body = bullets3[i]
            set_bullet_paragraph(p, marker, header, body, "Segoe UI", Pt(11))

    # Metric numbers and labels
    set_shape_text(s3.shapes[39], "20", "Georgia", Pt(32), CLR_STAT_NUM, bold=True)
    set_shape_text(s3.shapes[40], "DETECTOR ZOO (5 FAMILIES)", "Segoe UI", Pt(7.8), CLR_STAT_LABEL)
    set_shape_text(s3.shapes[43], "14,800+", "Georgia", Pt(32), CLR_STAT_NUM, bold=True)
    set_shape_text(s3.shapes[44], "LINES OF ML & MLOPS CODE", "Segoe UI", Pt(7.8), CLR_STAT_LABEL)
    set_shape_text(s3.shapes[47], "359", "Georgia", Pt(32), CLR_STAT_NUM, bold=True)
    set_shape_text(s3.shapes[48], "PYTEST TESTS (87% COVERAGE)", "Segoe UI", Pt(7.8), CLR_STAT_LABEL)
    set_shape_text(s3.shapes[51], "21", "Georgia", Pt(32), CLR_STAT_NUM, bold=True)
    set_shape_text(s3.shapes[52], "CONSOLE VIEWS & WORKFLOWS", "Segoe UI", Pt(7.8), CLR_STAT_LABEL)
    set_shape_text(s3.shapes[53], "Built with  Python 3.11+  ·  FastAPI  ·  scikit-learn  ·  PyTorch  ·  XGBoost  ·  SHAP  ·  Next.js 16  ·  Tailwind CSS", "Segoe UI", Pt(8.5), CLR_SUBTITLE)

    s3.notes_slide.notes_text_frame.text = (
        "Our objective was to build a complete, enterprise-ready anomaly intelligence platform rather than an isolated algorithm script.\n\n"
        "The system scope spans the complete pipeline: from automatic schema discovery and 20 diverse detectors to conformal false discovery rate control, "
        "streaming Half-Space Trees, and human-in-the-loop active learning.\n\n"
        "These four metrics represent our current repository state: 20 registered models, over 14,800 lines of verified code, 359 tests passing with 87% test coverage, and 21 interactive Next.js console views."
    )

    # =========================================================================
    # SLIDE 4: Data & Protocols (Light Background)
    # =========================================================================
    s4 = prs.slides[3]
    set_shape_text(s4.shapes[3], "03 · DATA & PROTOCOLS", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s4.shapes[4], "Benchmark Datasets & Split Protocols", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s4.shapes[5], "Transparent benchmarks: Synthetic controlled lab and real ULB credit transactions", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    # Stat tiles
    set_shape_text(s4.shapes[13], "880", "Georgia", Pt(20), CLR_STAT_NUM, bold=True)
    set_shape_text(s4.shapes[14], "ROWS", "Segoe UI", Pt(8), CLR_STAT_LABEL)
    set_shape_text(s4.shapes[17], "10", "Georgia", Pt(20), CLR_STAT_NUM, bold=True)
    set_shape_text(s4.shapes[18], "COLUMNS", "Segoe UI", Pt(8), CLR_STAT_LABEL)
    set_shape_text(s4.shapes[21], "9.09%", "Georgia", Pt(20), CLR_STAT_NUM, bold=True)
    set_shape_text(s4.shapes[22], "ANOMALY RATIO", "Segoe UI", Pt(8), CLR_STAT_LABEL)
    set_shape_text(s4.shapes[25], "6 + 2", "Georgia", Pt(20), CLR_STAT_NUM, bold=True)
    set_shape_text(s4.shapes[26], "NUMERIC + CATEGORICAL", "Segoe UI", Pt(8), CLR_STAT_LABEL)

    bullets4 = [
        ("▪", "Synthetic benchmark sample.", "880 rows, 10 columns, 80 injected anomalies (9.09% ratio) across known patterns."),
        ("▪", "ULB Credit Card fraud sample.", "20,000 rows preserving all 492 fraud cases (SHA-256: ede5aa5c...) for reproducible tests."),
        ("▪", "Honest chronological time-split.", "70% reference fit window (14,000 rows), 30% holdout evaluation window (6,000 rows)."),
        ("▪", "Zero future leakage.", "Scalers, encoders, and quantile thresholds are fit strictly on the reference window."),
        ("▪", "No random split inflation.", "Random splitting leaks same-day transaction patterns, artificially inflating PR-AUC from 0.31 to 0.65.")
    ]
    tf4 = s4.shapes[27].text_frame
    for i, p in enumerate(tf4.paragraphs):
        if i < len(bullets4):
            marker, header, body = bullets4[i]
            set_bullet_paragraph(p, marker, header, body, "Segoe UI", Pt(11))

    set_shape_text(s4.shapes[30], "Pearson correlation — outputs/figures/correlation_heatmap.png", "Segoe UI", Pt(8.5), CLR_SUBTITLE)
    set_shape_text(s4.shapes[32], "PREPARATION — NINE ORDERED, LOGGED STEPS   (src/data/preprocessing.py)", "Segoe UI", Pt(8.5), CLR_PILL)

    s4.notes_slide.notes_text_frame.text = (
        "We evaluate SENTINEL under two complementary data benchmarks: a controlled 880-row synthetic dataset and a real-world 20,000-transaction credit card fraud sample containing all 492 frauds.\n\n"
        "Crucially, we enforce a strict chronological time-split protocol. We fit our models on the first 70% of time-ordered transactions and evaluate on the remaining 30% holdout.\n\n"
        "We demonstrate in our documentation that conventional random splitting leaks customer behavior across days, artificially doubling reported PR-AUC from 0.31 to 0.65. SENTINEL refuses such deceptive practices."
    )

    # =========================================================================
    # SLIDE 5: Methodology & Architecture (Light Background)
    # =========================================================================
    s5 = prs.slides[4]
    set_shape_text(s5.shapes[3], "04 · METHODOLOGY & ARCHITECTURE", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s5.shapes[4], "Five stages, one audited path", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s5.shapes[5], "From raw stream to audited intervention — zero label leakage and full conformal bounds", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    stages = [
        (15, 16, "INGEST & PROFILE", [("·", "Schema discovery & health score"), ("·", "Multicollinearity condition number"), ("·", "Cryptographic SHA-256 fingerprint")]),
        (21, 22, "DETECTOR ZOO", [("·", "20 classical & deep models"), ("·", "Empirical rank-Borda consensus"), ("·", "Out-of-fold XGBoost surrogate")]),
        (27, 28, "CONFORMAL & FDR", [("·", "Finite-sample p-values"), ("·", "Mondrian subgroup fallback"), ("·", "Benjamini-Hochberg error budget")]),
        (33, 34, "EXPLAIN & AUDIT", [("·", "Sub-millisecond TreeSHAP"), ("·", "Deletion-AUC faithfulness audit"), ("·", "LOF-bounded counterfactuals")]),
        (39, 40, "STREAM & TRIAGE", [("·", "Streaming Half-Space Trees"), ("·", "Population Stability Index drift"), ("·", "Active learning feedback queue")])
    ]

    for title_idx, body_idx, st_title, items in stages:
        set_shape_text(s5.shapes[title_idx], st_title, "Segoe UI", Pt(10), CLR_HEAVY, bold=True)
        tf = s5.shapes[body_idx].text_frame
        while len(tf.paragraphs) < len(items):
            tf.add_paragraph()
        for p_idx, (bullet_sym, text_val) in enumerate(items):
            p = tf.paragraphs[p_idx]
            p.text = ""
            r0 = p.add_run()
            r0.text = bullet_sym + "  "
            r0.font.name = "Segoe UI"
            r0.font.size = Pt(9.5)
            r0.font.color.rgb = CLR_PILL
            r1 = p.add_run()
            r1.text = text_val
            r1.font.name = "Segoe UI"
            r1.font.size = Pt(9.5)
            r1.font.color.rgb = CLR_BODY_DARK

    s5.notes_slide.notes_text_frame.text = (
        "Here is the five-stage architecture of SENTINEL. In Stage 1, raw data is ingested with automated health profiling, collinearity condition numbers and SHA-256 hashes.\n\n"
        "Stage 2 executes our 20-model detector zoo and out-of-fold surrogate distillation.\n\n"
        "Stage 3 applies conformal prediction, converting ranking indices into mathematically guaranteed p-values under false discovery control.\n\n"
        "Stage 4 validates feature attributions through Deletion-AUC and LOF counterfactuals, while Stage 5 operates real-time streaming gates and active learning feedback."
    )

    # =========================================================================
    # SLIDE 6: Models / Detector Zoo (Light Background)
    # =========================================================================
    s6 = prs.slides[5]
    set_shape_text(s6.shapes[3], "05 · DETECTOR ZOO", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s6.shapes[4], "Twenty detectors across five families", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s6.shapes[5], "src/models/ — a modular capability registry, not a hardcoded list", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    tbl6 = s6.shapes[11].table
    tbl6.cell(0, 0).text = "Model Family"
    tbl6.cell(0, 1).text = "Core Algorithms"
    tbl6.cell(0, 2).text = "Complexity"
    tbl6.cell(0, 3).text = "Distinct Statistical Advantage"

    zoo_data = [
        ("Tree & Subspace", "Isolation Forest, Ext. iForest, PCA", "O(T·n log s)", "Fast, robust to noise; oblique hyperplanes catch linear correlations"),
        ("Statistical & Marginal", "HBOS, ECOD, COPOD", "O(n·d)", "Ultra-fast linear time; exact empirical CDFs without tuning parameters"),
        ("Density & Distance", "LOF, k-NN, DBSCAN/HDBSCAN", "O(n²·d)", "Detects local density dips and variable-density cluster isolates"),
        ("Covariance & Prob.", "Elliptic Envelope, Mahalanobis, GMM", "O(n·d² + d³)", "Optimal Gaussian/elliptical boundaries with Ledoit-Wolf shrinkage"),
        ("Deep Neural", "Autoencoder, VAE, Deep SVDD", "O(E·n·d)", "PyTorch CPU-friendly; exact native feature reconstruction attribution"),
        ("Supervised & Surrogate", "XGBoost Surrogate, Classifier", "O(T·n·d)", "Distills ensemble into 1ms inference with exact TreeSHAP explanations"),
        ("Hybrid Stacking", "Out-of-Fold Meta-Learner", "O(K·n·(M+T))", "Stacks 8 out-of-fold unsupervised scores into a boosted meta-boundary"),
        ("Fast Scalable Ensemble", "SUOD Ensemble, Rank-Borda", "O(T·n·d_sub)", "Accelerates expensive base estimators via random subspace projections")
    ]
    for r_idx, (fam, algs, comp, adv) in enumerate(zoo_data, start=1):
        c0, c1, c2, c3 = tbl6.cell(r_idx, 0), tbl6.cell(r_idx, 1), tbl6.cell(r_idx, 2), tbl6.cell(r_idx, 3)
        c0.text, c1.text, c2.text, c3.text = fam, algs, comp, adv
        for c in (c0, c1, c2, c3):
            for p in c.text_frame.paragraphs:
                p.font.name = "Segoe UI"
                p.font.size = Pt(8.5)
                p.font.color.rgb = CLR_TBL_CELL

    set_shape_text(s6.shapes[14], "CONSENSUS ENGINE", "Segoe UI", Pt(8.5), CLR_ACCENT, bold=True)
    set_shape_text(s6.shapes[15], "Empirical rank normalisation → Rank Borda count & XGBoost stacking. Raw scores from disparate statistical geometries are projected onto uniform quantiles before consensus aggregation.", "Segoe UI", Pt(9.5), CLR_BODY_DARK)

    s6.notes_slide.notes_text_frame.text = (
        "No single algorithm fits every anomaly geometry. Anomaly detection spans points, micro-clusters, and manifold fractures.\n\n"
        "SENTINEL incorporates 20 distinct detectors across 5 statistical families: from fast linear CDF models like ECOD and HBOS to deep variational autoencoders and out-of-fold XGBoost surrogates.\n\n"
        "To combine them without unit distortion, our ensemble engine normalizes raw scores into empirical quantile ranks before executing Borda counts or hybrid stacking."
    )

    # =========================================================================
    # SLIDE 7: Explainability (Light Background)
    # =========================================================================
    s7 = prs.slides[6]
    set_shape_text(s7.shapes[3], "06 · EXPLAINABILITY", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s7.shapes[4], "Explainability that survives a scientific audit", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s7.shapes[5], "Attributions are generated, verified for faithfulness, and checked for cross-method agreement", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    bullets7 = [
        ("▪", "TreeSHAP on Surrogate.", "Sub-millisecond exact Shapley attribution without expensive perturbation sampling."),
        ("▪", "2nd-Order Interactions.", "Pairwise non-linear interaction matrices uncover multi-feature fraud synergies."),
        ("▪", "Faithfulness Audit.", "Deletion-AUC quantitatively proves that zeroing top features collapses anomaly scores."),
        ("▪", "Cross-Method Agreement.", "Quantifies Spearman correlation and top-K feature overlap between SHAP and LIME."),
        ("▪", "Plausible Counterfactuals.", "Actionable recourse paths constrained by Local Outlier Factor density and L0 sparsity.")
    ]
    tf7 = s7.shapes[11].text_frame
    for i, p in enumerate(tf7.paragraphs):
        if i < len(bullets7):
            marker, header, body = bullets7[i]
            set_bullet_paragraph(p, marker, header, body, "Segoe UI", Pt(11))

    s7.notes_slide.notes_text_frame.text = (
        "In many tools, explainability stops at a pretty waterfall chart. In SENTINEL, attribution is treated as an empirical claim that must be tested.\n\n"
        "We verify three mathematical properties: Additivity, ensuring SHAP values sum to the score; Faithfulness, testing via Deletion-AUC that masking top features collapses the anomaly score; "
        "and Stability, verifying that explanations do not fluctuate under random re-seeding.\n\n"
        "Furthermore, our counterfactual engine ensures recourse suggestions remain inside the dense, nominal data manifold using an LOF density filter."
    )

    # =========================================================================
    # SLIDE 8: Results & Benchmarks (Light Background)
    # =========================================================================
    s8 = prs.slides[7]
    set_shape_text(s8.shapes[3], "07 · EMPIRICAL RESULTS", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s8.shapes[4], "Real-world & benchmark detection performance", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s8.shapes[5], "ULB Credit Card benchmark (20k sample, 492 frauds) & 880-row controlled sample", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    # Stat numbers & labels
    set_shape_text(s8.shapes[13], "0.3765", "Georgia", Pt(30), CLR_STAT_NUM, bold=True)
    set_shape_text(s8.shapes[14], "ULB HBOS PR-AUC", "Segoe UI", Pt(7.8), CLR_STAT_LABEL)
    set_shape_text(s8.shapes[17], "0.9343", "Georgia", Pt(30), CLR_STAT_NUM, bold=True)
    set_shape_text(s8.shapes[18], "ULB HBOS ROC-AUC", "Segoe UI", Pt(7.8), CLR_STAT_LABEL)
    set_shape_text(s8.shapes[21], "0.3108", "Georgia", Pt(30), CLR_STAT_NUM, bold=True)
    set_shape_text(s8.shapes[22], "ULB iFOREST PR-AUC", "Segoe UI", Pt(7.8), CLR_STAT_LABEL)
    set_shape_text(s8.shapes[25], "0.02s", "Georgia", Pt(30), CLR_STAT_NUM, bold=True)
    set_shape_text(s8.shapes[26], "HBOS FIT LATENCY", "Segoe UI", Pt(7.8), CLR_STAT_LABEL)
    set_shape_text(s8.shapes[29], "1.000", "Georgia", Pt(30), CLR_STAT_NUM, bold=True)
    set_shape_text(s8.shapes[30], "SAMPLE PRECISION", "Segoe UI", Pt(7.8), CLR_STAT_LABEL)

    set_shape_text(s8.shapes[47], "ULB Real Benchmark: HBOS achieves PR-AUC 0.3765 (Precision@1% 0.5000), outperforming iForest (0.3108) at 14x faster speed (0.02s vs 0.28s). Random-split optimism gap is honestly disclosed.", "Segoe UI", Pt(9.5), CLR_BODY_DARK)

    s8.notes_slide.notes_text_frame.text = (
        "Here are our verified empirical results on the ULB Credit Card fraud benchmark under honest chronological time-split evaluation.\n\n"
        "HBOS emerged as the top-performing detector, achieving PR-AUC of 0.3765 and Precision@Top-1% of 0.5000, while fitting in just 20 milliseconds. "
        "Isolation Forest followed closely with PR-AUC of 0.3108 and ROC-AUC of 0.9340.\n\n"
        "On our synthetic benchmark sample, Isolation Forest catches 44 anomalies with zero false alarms, achieving an honest 10-fold precision lift over the 9% base rate."
    )

    # =========================================================================
    # SLIDE 9: Evidence & New Capabilities (Light Background)
    # =========================================================================
    s9 = prs.slides[8]
    set_shape_text(s9.shapes[3], "08 · ADVANCED CAPABILITIES", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s9.shapes[4], "Conformal bounds, streaming, and active triage", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s9.shapes[5], "Transitioning from static batch scoring to continuous operational intelligence", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    bullets9_1 = [
        ("▪", "Conformal p-values.", "Non-conformity calibration guarantees valid tail bounds under exchangeability."),
        ("▪", "Mondrian fallback.", "Automatically reverts to global calibrators when subgroup sample size n < ceil(1/α)-1."),
        ("▪", "Benjamini-Hochberg FDR.", "Explicit false discovery rate error budgeting for compliance-driven triage.")
    ]
    tf9_1 = s9.shapes[17].text_frame
    for i, p in enumerate(tf9_1.paragraphs):
        if i < len(bullets9_1):
            marker, header, body = bullets9_1[i]
            set_bullet_paragraph(p, marker, header, body, "Segoe UI", Pt(11))

    bullets9_2 = [
        ("▪", "Streaming Half-Space Trees.", "Online anomaly scoring over infinite streams without full retraining."),
        ("▪", "Population Stability Index.", "Continuous distribution drift monitoring triggering automated alert flags."),
        ("▪", "Active Feedback Queue.", "Human-in-the-loop analyst feedback updating surrogate boundaries dynamically.")
    ]
    tf9_2 = s9.shapes[18].text_frame
    for i, p in enumerate(tf9_2.paragraphs):
        if i < len(bullets9_2):
            marker, header, body = bullets9_2[i]
            set_bullet_paragraph(p, marker, header, body, "Segoe UI", Pt(11))

    set_shape_text(s9.shapes[20], "SENTINEL 3.0 bridges statistical theory and operational deployment: finite-sample error guarantees provide compliance, while streaming trees and active learning ensure continuous adaptation.", "Segoe UI", Pt(10.5), CLR_HEAVY)

    s9.notes_slide.notes_text_frame.text = (
        "In Version 3.0, we elevated SENTINEL into an operational platform by adding three advanced capabilities.\n\n"
        "First, Conformal Prediction provides finite-sample p-values with Mondrian fallback and Benjamini-Hochberg false discovery rate control.\n\n"
        "Second, Streaming Half-Space Trees process real-time events while Population Stability Index monitors distributional shift.\n\n"
        "Third, an Active Feedback queue enables human analysts to confirm or dismiss alerts, feeding verified labels back into surrogate retraining."
    )

    # =========================================================================
    # SLIDE 10: Key Findings & MLOps Lineage (Light Background)
    # =========================================================================
    s10 = prs.slides[9]
    set_shape_text(s10.shapes[3], "09 · MLOPS LINEAGE & GOVERNANCE", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s10.shapes[4], "Auditable, reproducible machine learning", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s10.shapes[5], "Standardized Model Cards, cryptographic hashes, and interactive forensic dossiers", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    bullets10 = [
        ("▪", "Standardized Model Cards.", "Auto-generated Mitchell et al. cards detailing intended use, limitations, and demographic parity."),
        ("▪", "Serialized model bundles.", "Stores model weights, schemas, and dataset hashes in outputs/models/ for complete auditability."),
        ("▪", "Immutable experiment lineage.", "SHA-256 fingerprinting guarantees every benchmark is mathematically reproducible."),
        ("▪", "Surrogate fidelity certification.", "Tracks agreement Spearman rank correlation and top-K overlap (is_ground_truth=False)."),
        ("▪", "Audit-ready HTML reporting.", "One-click export of standalone, client-shareable forensic investigation dossiers.")
    ]
    tf10 = s10.shapes[11].text_frame
    for i, p in enumerate(tf10.paragraphs):
        if i < len(bullets10):
            marker, header, body = bullets10[i]
            set_bullet_paragraph(p, marker, header, body, "Segoe UI", Pt(11))

    set_shape_text(s10.shapes[14], "MLOPS INTEGRITY", "Segoe UI", Pt(8.5), CLR_ACCENT, bold=True)
    set_shape_text(s10.shapes[15], "SENTINEL rejects black-box deployment. Every fitted model artifact is serialized alongside its training dataset SHA-256 hash, hyperparameter envelope, and demographic robustness audit.", "Segoe UI", Pt(10), CLR_BODY_DARK)

    s10.notes_slide.notes_text_frame.text = (
        "Enterprise machine learning requires accountability. In SENTINEL, governance is not an afterthought; it is built into the core API.\n\n"
        "We dynamically generate standardized Model Cards based on the Mitchell et al. framework, specifying demographic parity metrics and training boundaries.\n\n"
        "Every experiment and fitted weight file is tracked against its cryptographic SHA-256 input hash, and analysts can export complete, standalone HTML dossiers for regulatory audits."
    )

    # =========================================================================
    # SLIDE 11: Honesty Check & Limitations (Light Background)
    # =========================================================================
    s11 = prs.slides[10]
    set_shape_text(s11.shapes[3], "10 · HONESTY CHECK & LIMITATIONS", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s11.shapes[4], "What was solved, and what boundaries remain", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s11.shapes[5], "Transparent disclosure of completed capabilities and fundamental statistical limitations", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    set_shape_text(s11.shapes[11], "RESOLVED IN SENTINEL 3.0", "Segoe UI", Pt(9.5), CLR_GREEN, bold=True)
    set_shape_text(s11.shapes[12], "STATISTICAL LIMITATIONS (DISCLOSED)", "Segoe UI", Pt(9.5), CLR_RED, bold=True)

    # Left side (Resolved)
    set_shape_text(s11.shapes[15], "Conformal p-values", "Segoe UI", Pt(10), CLR_HEAVY, bold=True)
    set_shape_text(s11.shapes[16], "Finite-sample non-conformity p-values with Mondrian fallback and Benjamini-Hochberg FDR.", "Segoe UI", Pt(9), CLR_BODY_DARK)
    set_shape_text(s11.shapes[17], "IMPLEMENTED · ConformalEngine", "Segoe UI", Pt(8), CLR_GREEN, bold=True)

    set_shape_text(s11.shapes[20], "Streaming detection", "Segoe UI", Pt(10), CLR_HEAVY, bold=True)
    set_shape_text(s11.shapes[21], "Half-Space Trees with sliding reference mass and real-time PSI drift monitoring.", "Segoe UI", Pt(9), CLR_BODY_DARK)
    set_shape_text(s11.shapes[22], "IMPLEMENTED · StreamingDetector", "Segoe UI", Pt(8), CLR_GREEN, bold=True)

    set_shape_text(s11.shapes[25], "Deep learning detectors", "Segoe UI", Pt(10), CLR_HEAVY, bold=True)
    set_shape_text(s11.shapes[26], "Autoencoder, VAE, and Deep SVDD implemented with PyTorch native reconstruction attribution.", "Segoe UI", Pt(9), CLR_BODY_DARK)
    set_shape_text(s11.shapes[27], "IMPLEMENTED · DeepAutoencoder", "Segoe UI", Pt(8), CLR_GREEN, bold=True)

    set_shape_text(s11.shapes[30], "Hybrid stacking & tuning", "Segoe UI", Pt(10), CLR_HEAVY, bold=True)
    set_shape_text(s11.shapes[31], "Out-of-fold multi-model stacking with Optuna nested cross-validation hyperparameter search.", "Segoe UI", Pt(9), CLR_BODY_DARK)
    set_shape_text(s11.shapes[32], "IMPLEMENTED · HybridStacking", "Segoe UI", Pt(8), CLR_GREEN, bold=True)

    # Right side (Limitations from docs/STATISTICAL_ASSUMPTIONS.md)
    set_shape_text(s11.shapes[35], "Exchangeability assumption", "Segoe UI", Pt(10), CLR_HEAVY, bold=True)
    set_shape_text(s11.shapes[36], "Conformal bounds assume test points are exchangeable with calibration data. Non-stationary drift (PSI ≥ 0.25) invalidates bounds.", "Segoe UI", Pt(9), CLR_BODY_DARK)
    set_shape_text(s11.shapes[37], "DISCLOSED · Exchangeability gate", "Segoe UI", Pt(8), CLR_RED, bold=True)

    set_shape_text(s11.shapes[40], "Non-causal attribution", "Segoe UI", Pt(10), CLR_HEAVY, bold=True)
    set_shape_text(s11.shapes[41], "SHAP and counterfactuals measure observational model surface correlation, not physical causation. Changing a feature may not alter reality.", "Segoe UI", Pt(9), CLR_BODY_DARK)
    set_shape_text(s11.shapes[42], "DISCLOSED · Non-causal notice", "Segoe UI", Pt(8), CLR_RED, bold=True)

    set_shape_text(s11.shapes[45], "Calibration set contamination", "Segoe UI", Pt(10), CLR_HEAVY, bold=True)
    set_shape_text(s11.shapes[46], "Unsupervised calibration sets may contain undetected anomalies, making p-values conservative. Documented in /api/v1/conformal/status.", "Segoe UI", Pt(9), CLR_BODY_DARK)
    set_shape_text(s11.shapes[47], "DISCLOSED · Documented behavior", "Segoe UI", Pt(8), CLR_RED, bold=True)

    set_shape_text(s11.shapes[50], "Surrogate pseudo-label gap", "Segoe UI", Pt(10), CLR_HEAVY, bold=True)
    set_shape_text(s11.shapes[51], "Surrogate TreeSHAP explains the surrogate model, not directly the base ensemble. Measured via agreement_* metrics (is_ground_truth=False).", "Segoe UI", Pt(9), CLR_BODY_DARK)
    set_shape_text(s11.shapes[52], "DISCLOSED · Surrogate boundary", "Segoe UI", Pt(8), CLR_RED, bold=True)

    s11.notes_slide.notes_text_frame.text = (
        "Scientific credibility requires stating what your system cannot do alongside what it can do.\n\n"
        "On the left, we highlight major engineering milestones achieved in Version 3.0: conformal calibration, streaming HSTrees, deep autoencoders, and hybrid stacking.\n\n"
        "On the right, we disclose fundamental statistical limitations from our project documentation: the exchangeability requirement for conformal guarantees, "
        "the non-causal nature of observational SHAP attributions, potential calibration contamination in unsupervised regimes, and the surrogate explanation boundary."
    )

    # =========================================================================
    # SLIDE 12: Conclusion & Repository Map (Hybrid Light Top / Dark Bottom)
    # =========================================================================
    s12 = prs.slides[11]
    set_shape_text(s12.shapes[3], "11 · CONCLUSION", "Segoe UI", Pt(9.5), CLR_PILL)
    set_shape_text(s12.shapes[4], "Beyond detection. Into intelligence.", "Georgia", Pt(28), CLR_TITLE, bold=True)
    set_shape_text(s12.shapes[5], "SENTINEL 3.0 — evidence over scores, conservative bounds, reproducible science", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    # Top light cards
    set_shape_text(s12.shapes[13], "Evidence over scores", "Georgia", Pt(14), CLR_HEAVY, bold=True)
    set_shape_text(s12.shapes[14], "Every anomaly alert carries finite-sample conformal bounds and a verified attribution audit.", "Segoe UI", Pt(9.5), CLR_BODY_DARK)

    set_shape_text(s12.shapes[17], "Conservative by design", "Georgia", Pt(14), CLR_HEAVY, bold=True)
    set_shape_text(s12.shapes[18], "Mondrian fallback and Benjamini-Hochberg false discovery control protect against over-alerting.", "Segoe UI", Pt(9.5), CLR_BODY_DARK)

    set_shape_text(s12.shapes[21], "Reproducible end to end", "Georgia", Pt(14), CLR_HEAVY, bold=True)
    set_shape_text(s12.shapes[22], "SHA-256 dataset fingerprints, 359 tests (87% coverage), and Mitchell et al. model cards.", "Segoe UI", Pt(9.5), CLR_BODY_DARK)

    # Bottom dark box
    set_shape_text(s12.shapes[25], "Thank you", "Georgia", Pt(24), CLR_S12_BOX_TITLE, bold=True)
    set_shape_text(s12.shapes[26], "Questions & discussion welcome", "Segoe UI", Pt(11), CLR_S12_BOX_AMBER)
    set_shape_text(s12.shapes[27], "The deck, the code and every figure in it are reproducible from the repository.", "Segoe UI", Pt(9), CLR_S12_BOX_MUTED)

    set_shape_text(s12.shapes[28], "REPOSITORY MAP", "Segoe UI", Pt(8.5), CLR_S12_BOX_AMBER, bold=True)
    set_shape_text(s12.shapes[29], "Core ML", "Segoe UI", Pt(8.5), CLR_S12_BOX_MUTED)
    set_shape_text(s12.shapes[30], "src/", "Segoe UI", Pt(8.5), CLR_S12_BOX_TITLE)
    set_shape_text(s12.shapes[31], "Backend API", "Segoe UI", Pt(8.5), CLR_S12_BOX_MUTED)
    set_shape_text(s12.shapes[32], "backend/", "Segoe UI", Pt(8.5), CLR_S12_BOX_TITLE)
    set_shape_text(s12.shapes[33], "Console", "Segoe UI", Pt(8.5), CLR_S12_BOX_MUTED)
    set_shape_text(s12.shapes[34], "frontend/", "Segoe UI", Pt(8.5), CLR_S12_BOX_TITLE)
    set_shape_text(s12.shapes[35], "Artefacts", "Segoe UI", Pt(8.5), CLR_S12_BOX_MUTED)
    set_shape_text(s12.shapes[36], "outputs/", "Segoe UI", Pt(8.5), CLR_S12_BOX_TITLE)
    set_shape_text(s12.shapes[37], "Tests", "Segoe UI", Pt(8.5), CLR_S12_BOX_MUTED)
    set_shape_text(s12.shapes[38], "tests/ (359 passing)", "Segoe UI", Pt(8.5), CLR_S12_BOX_TITLE)

    s12.notes_slide.notes_text_frame.text = (
        "In conclusion, SENTINEL 3.0 moves tabular anomaly detection beyond isolated algorithms and into a coherent, audited intelligence platform.\n\n"
        "We prioritize evidence over raw scores, design conservative finite-sample bounds, and provide end-to-end mathematical reproducibility.\n\n"
        "Thank you for your time. The complete codebase, test suite, and experiment history are open and available in the repository. I look forward to your questions."
    )

    # =========================================================================
    # SLIDE 13: Viva Defense & Technical Q&A (Light Theme)
    # =========================================================================
    blank_layout = prs.slide_layouts[6]
    s13 = prs.slides.add_slide(blank_layout)

    # Left edge vertical purple stripe (matching slides 2-12)
    v_stripe = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.25), Inches(7.5))
    v_stripe.fill.solid()
    v_stripe.fill.fore_color.rgb = CLR_ACCENT
    v_stripe.line.fill.background()

    # Section Pill
    pill_box = s13.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(8.0), Inches(0.35))
    pp = pill_box.text_frame.paragraphs[0]
    set_run_text(pp, "12 · VIVA DEFENSE & PANEL Q&A", "Segoe UI", Pt(9.5), CLR_PILL, bold=True)

    # Slide Title
    title_box = s13.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.5), Inches(0.6))
    tp = title_box.text_frame.paragraphs[0]
    set_run_text(tp, "Viva Defense & Technical Discussion", "Georgia", Pt(28), CLR_TITLE, bold=True)

    # Slide Subtitle
    sub_box = s13.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.5), Inches(0.4))
    sp = sub_box.text_frame.paragraphs[0]
    set_run_text(sp, "Five architectural justifications and mathematical defenses prepared for panel examination", "Segoe UI", Pt(11.5), CLR_SUBTITLE)

    # Five Questions formatted into elegant light cards
    questions = [
        ("Q1: Why refuse to call unsupervised anomaly scores 'probabilities'?",
         "Unsupervised scores measure geometric distance, density, or tree path lengths without a calibrated reference null. Calling them probabilities implies an uncalibrated posterior P(y=1|x) which leads to arbitrary thresholding. We use conformal p-values instead."),
        ("Q2: How does Mondrian conformal prediction guarantee false discovery control?",
         "It partitions calibration points into demographic or categorical subgroups, computing non-conformity quantiles per group. If a subgroup has n < ceil(1/α)-1 samples, it safely falls back to the global reference set with Benjamini-Hochberg FDR budgeting."),
        ("Q3: Why does HBOS outperform Isolation Forest on the ULB credit fraud dataset?",
         "ULB features are orthogonal PCA components (V1-V28). HBOS exploits independent marginal histograms directly, scoring extreme variance components in linear O(n·d) time (0.02s) without tree-axis splitting artifacts."),
        ("Q4: How does SENTINEL guarantee zero data leakage during surrogate training?",
         "Surrogate XGBoost models are trained strictly via K-fold out-of-fold (OOF) pseudo-labels. Scalers, quantile thresholds, and surrogates are fit exclusively on training folds and evaluated on held-out folds."),
        ("Q5: What happens to conformal guarantees under concept drift?",
         "Finite-sample validity assumes exchangeability. Non-stationary drift breaks this assumption. SENTINEL monitors Population Stability Index (PSI); if PSI ≥ 0.25, drift gates alarm and trigger reference mass recalibration.")
    ]

    card_left = Inches(0.8)
    card_width = Inches(11.733)
    card_height = Inches(0.88)
    y_start = Inches(1.85)
    gap = Inches(0.96)

    for q_idx, (q_text, a_text) in enumerate(questions):
        y_pos = y_start + (q_idx * gap)

        # Light card container
        q_card = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, card_left, y_pos, card_width, card_height)
        q_card.fill.solid()
        q_card.fill.fore_color.rgb = RGBColor(0xF6, 0xF4, 0xFA)
        q_card.line.color.rgb = RGBColor(0xD8, 0xD0, 0xE8)
        q_card.line.width = Pt(1)

        # Question title
        tb = s13.shapes.add_textbox(card_left + Inches(0.2), y_pos + Inches(0.06), card_width - Inches(0.4), Inches(0.35))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p0 = tf.paragraphs[0]
        set_run_text(p0, q_text, "Segoe UI", Pt(10.5), CLR_ACCENT, bold=True)

        # Answer body
        tb2 = s13.shapes.add_textbox(card_left + Inches(0.2), y_pos + Inches(0.38), card_width - Inches(0.4), Inches(0.45))
        tf2 = tb2.text_frame
        tf2.word_wrap = True
        tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0
        p1 = tf2.paragraphs[0]
        set_run_text(p1, a_text, "Segoe UI", Pt(9.5), CLR_BODY_DARK)

    # Footer on Slide 13
    ft_box = s13.shapes.add_textbox(Inches(0.8), Inches(6.9), Inches(8.0), Inches(0.3))
    ftp = ft_box.text_frame.paragraphs[0]
    set_run_text(ftp, "SENTINEL 3.0  ·  Autonomous Anomaly Intelligence Platform", "Segoe UI", Pt(8.5), CLR_FOOTER)

    pg_box = s13.shapes.add_textbox(Inches(11.5), Inches(6.9), Inches(1.0), Inches(0.3))
    pgp = pg_box.text_frame.paragraphs[0]
    set_run_text(pgp, "13 / 13", "Segoe UI", Pt(8.5), CLR_FOOTER)

    # Notes slide for Slide 13
    ns0 = prs.slides[0].notes_slide
    for sp_elem in ns0.shapes:
        if sp_elem.is_placeholder and sp_elem.placeholder_format.type == pptx.enum.shapes.PP_PLACEHOLDER.BODY:
            new_sp = copy.deepcopy(sp_elem._element)
            s13.notes_slide.shapes._spTree.append(new_sp)

    if s13.notes_slide.notes_text_frame:
        s13.notes_slide.notes_text_frame.text = (
            "Here are five questions a viva examiner or technical panel is most likely to ask, along with our crisp answers:\n\n"
            "1. Why not treat anomaly scores as probabilities? Because unsupervised algorithms output uncalibrated density or distance metrics. Without a reference null or conformal bounds, claiming P(anomaly|x) is mathematical fiction.\n\n"
            "2. How does Mondrian conformal prediction guarantee false discovery control? By calibrating non-conformity scores within subgroups, falling back to global calibrators when sample size n is below the ceil(1/alpha)-1 threshold, and applying the Benjamini-Hochberg procedure.\n\n"
            "3. Why did HBOS beat Isolation Forest on ULB credit fraud? Because PCA-preprocessed features are orthogonal. HBOS estimates marginal histograms in linear O(n*d) time without tree partitioning variance, running in 20 milliseconds.\n\n"
            "4. How do you prevent surrogate leakage? We strictly use K-fold out-of-fold training. Scalers and surrogate trees are fit only on training folds and scored on unseen holdout folds.\n\n"
            "5. What happens under drift? Conformal validity requires exchangeability. We monitor Population Stability Index in real-time; if PSI exceeds 0.25, SENTINEL flags an exchangeability violation and triggers reference mass recalibration."
        )

    # Save presentation
    prs.save(dest_pptx)
    print(f"Successfully generated presentation: {dest_pptx} (Total slides: {len(prs.slides)})")

if __name__ == "__main__":
    src_file = r"C:\Users\ADMIN\Documents\ML PROJECT\explainable-anomaly-detection\presentation\original_presentation.pptx"
    dest_file = r"C:\Users\ADMIN\Documents\ML PROJECT\explainable-anomaly-detection\presentation\Explainable_Anomaly_Detection_12slides.pptx"
    update_presentation(src_file, dest_file)

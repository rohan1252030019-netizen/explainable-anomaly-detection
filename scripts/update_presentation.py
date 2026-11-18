import os
import sys
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def update_presentation(pptx_path: str):
    prs = Presentation(pptx_path)
    
    # Palette constants
    CLR_AMBER = RGBColor(0xFD, 0xE6, 0x8A)
    CLR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    CLR_MUTED = RGBColor(0xBD, 0xB2, 0xD6)
    CLR_LAVENDER = RGBColor(0xE4, 0xDC, 0xF3)
    CLR_BG_DARK = RGBColor(0x13, 0x0E, 0x24)
    CLR_CARD_BG = RGBColor(0x1F, 0x18, 0x38)
    CLR_CARD_BORDER = RGBColor(0x3B, 0x2D, 0x64)
    
    def set_run_text(p, text, font_name=None, font_size=None, color_rgb=None, bold=None):
        if not p.runs:
            r = p.add_run()
        else:
            r = p.runs[0]
            for extra in p.runs[1:]:
                extra.text = ""
        r.text = text
        if font_name:
            r.font.name = font_name
        if font_size:
            r.font.size = font_size
        if color_rgb:
            r.font.color.rgb = color_rgb
        if bold is not None:
            r.font.bold = bold

    def set_shape_text(shape, text, font_name=None, font_size=None, color_rgb=None, bold=None):
        if not shape.has_text_frame:
            return
        tf = shape.text_frame
        p = tf.paragraphs[0]
        set_run_text(p, text, font_name, font_size, color_rgb, bold)

    # Global footer and page number updater
    for idx, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if shape.has_text_frame:
                txt = shape.text_frame.text.strip()
                if "XAUD 2.0  ·  Explainable Anomaly Detection Framework" in txt:
                    set_shape_text(shape, "SENTINEL 3.0  ·  Autonomous Anomaly Intelligence Platform", "Segoe UI", Pt(8.5), CLR_MUTED)
                elif "Explainable Anomaly Detection Framework (XAUD 2.0)" in txt:
                    set_shape_text(shape, "SENTINEL 3.0  ·  Autonomous Anomaly Intelligence Platform  ·  Apache 2.0  ·  Python 3.11+ / FastAPI / PyTorch / Next.js 16", "Segoe UI", Pt(8.5), CLR_MUTED)
                # Update page number e.g. 02 / 12 -> 02 / 13
                if txt.endswith("/ 12") and len(txt) <= 7:
                    new_page_str = f"{idx:02d} / 13"
                    set_shape_text(shape, new_page_str, "Segoe UI", Pt(8.5), CLR_MUTED)

    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    s1 = prs.slides[0]
    set_shape_text(s1.shapes[34], "SENTINEL — ANOMALY INTELLIGENCE PLATFORM", "Segoe UI", Pt(10), CLR_AMBER, bold=True)
    set_shape_text(s1.shapes[35], "SENTINEL", "Georgia", Pt(84), CLR_WHITE, bold=True)
    set_shape_text(s1.shapes[36], "Autonomous Anomaly Intelligence Platform", "Segoe UI", Pt(19), CLR_LAVENDER)
    set_shape_text(s1.shapes[37], "“Detect. Explain. Verify. Act.”", "Segoe UI", Pt(13.5), CLR_AMBER)
    set_shape_text(s1.shapes[45], "19 November 2026", "Segoe UI", Pt(12), CLR_WHITE)
    set_shape_text(s1.shapes[48], "Python 3.11+  ·  FastAPI  ·  scikit-learn  ·  PyTorch  ·  XGBoost  ·  SHAP  ·  Next.js 16  ·  Tailwind CSS", "Segoe UI", Pt(8.8), CLR_MUTED)
    
    s1.notes_slide.notes_text_frame.text = (
        "Good morning. Today I am presenting SENTINEL 3.0, an Autonomous Anomaly Intelligence Platform for tabular data and streaming environments.\n\n"
        "Our core philosophy is captured in our tagline: 'Detect. Explain. Verify. Act.' SENTINEL does not merely output opaque outlier flags; "
        "it produces mathematically calibrated conformal error bounds, verifiable feature attributions, and actionable counterfactual recommendations.\n\n"
        "Every metric, algorithmic benchmark, and performance claim in this presentation is directly backed by reproducible, committed artifacts in our repository."
    )

    # ==========================================
    # SLIDE 2: Problem Statement
    # ==========================================
    s2 = prs.slides[1]
    set_shape_text(s2.shapes[3], "01 · PROBLEM STATEMENT", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s2.shapes[4], "Anomalies without a witness", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s2.shapes[5], "Unsupervised detection is easy to run, but dangerous to trust without proof", "Segoe UI", Pt(11.5), CLR_LAVENDER)
    
    # Bullet points shape 11
    tf2 = s2.shapes[11].text_frame
    bullets2 = [
        "▪  Rare and unlabeled. Anomalies comprise <2% of rows, leaving practitioners with zero ground truth during fit.",
        "▪  Scores sold as probabilities. Unsupervised scores are relative ranks, not posterior probabilities P(anomaly|x).",
        "▪  Silent calibration collapse. Arbitrary thresholds fail when production data distributions experience concept drift.",
        "▪  Unexplained explanations. Popular SHAP visualisations are displayed without testing mathematical faithfulness.",
        "▪  Absence of audit lineage. Traditional pipelines lack cryptographic fingerprinting and verifiable model cards."
    ]
    for i, p in enumerate(tf2.paragraphs):
        if i < len(bullets2):
            set_run_text(p, bullets2[i], "Segoe UI", Pt(11), CLR_WHITE)
            
    # Table shape 13
    set_shape_text(s2.shapes[12], "STANDARD TOOL  vs  SENTINEL 3.0", "Segoe UI", Pt(8.5), CLR_MUTED)
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
                p.font.color.rgb = CLR_WHITE

    set_shape_text(s2.shapes[18], "It is not about raw accuracy — it is about evidence. SENTINEL is engineered so that every alert traces back to an honest split, a calibrated error rate, and a cryptographic fingerprint.", "Segoe UI", Pt(11), CLR_WHITE)

    s2.notes_slide.notes_text_frame.text = (
        "In production anomaly detection, the biggest risk is not failing to find an anomaly; it is acting on an unsubstantiated alert.\n\n"
        "Most open-source tools commit three critical errors: they market raw outlier indices as 'probabilities', they leak future data by fitting scalers globally, "
        "and they display SHAP charts without checking whether the attribution is faithful to the model.\n\n"
        "As shown in our comparison table, SENTINEL replaces arbitrary heuristics with finite-sample conformal calibration, time-aware validation, and quantitative explanation audits."
    )

    # ==========================================
    # SLIDE 3: Objectives & Scope
    # ==========================================
    s3 = prs.slides[2]
    set_shape_text(s3.shapes[3], "02 · OBJECTIVES & SCOPE", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s3.shapes[4], "What SENTINEL sets out to do", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s3.shapes[5], "Detect → calibrate → explain → monitor → triage → reproduce", "Segoe UI", Pt(11.5), CLR_LAVENDER)
    
    tf3 = s3.shapes[11].text_frame
    bullets3 = [
        "▪  Unsupervised & hybrid zoo. 18 detectors across 5 statistical families — from tree ensembles to deep VAEs.",
        "▪  Conformal false discovery control. Finite-sample tail p-values with Benjamini-Hochberg error budgeting.",
        "▪  Multi-angle explainability. TreeSHAP, 2nd-order interactions, LOF counterfactuals, and narrative reports.",
        "▪  Streaming & drift gates. Half-Space Trees with continuous sliding-window PSI monitoring and recalibration.",
        "▪  Full MLOps lifecycle. Automated Mitchell et al. model cards, active learning queues, and SHA-256 lineage."
    ]
    for i, p in enumerate(tf3.paragraphs):
        if i < len(bullets3):
            set_run_text(p, bullets3[i], "Segoe UI", Pt(11), CLR_WHITE)

    # Metrics tiles
    set_shape_text(s3.shapes[39], "18", "Georgia", Pt(32), CLR_AMBER, bold=True)
    set_shape_text(s3.shapes[40], "DETECTOR ZOO (5 FAMILIES)", "Segoe UI", Pt(7.8), CLR_MUTED)
    
    set_shape_text(s3.shapes[43], "14,800+", "Georgia", Pt(32), CLR_AMBER, bold=True)
    set_shape_text(s3.shapes[44], "LINES OF ML & MLOPS CODE", "Segoe UI", Pt(7.8), CLR_MUTED)
    
    set_shape_text(s3.shapes[47], "359", "Georgia", Pt(32), CLR_AMBER, bold=True)
    set_shape_text(s3.shapes[48], "PYTEST TESTS (87% COVERAGE)", "Segoe UI", Pt(7.8), CLR_MUTED)
    
    set_shape_text(s3.shapes[51], "21", "Georgia", Pt(32), CLR_AMBER, bold=True)
    set_shape_text(s3.shapes[52], "CONSOLE VIEWS & WORKFLOWS", "Segoe UI", Pt(7.8), CLR_MUTED)

    set_shape_text(s3.shapes[53], "Built with  Python 3.11+  ·  FastAPI  ·  scikit-learn  ·  PyTorch  ·  XGBoost  ·  SHAP  ·  Next.js 16  ·  Tailwind CSS", "Segoe UI", Pt(8.5), CLR_MUTED)

    s3.notes_slide.notes_text_frame.text = (
        "Our objective was to build a complete, enterprise-ready anomaly intelligence platform rather than an isolated algorithm script.\n\n"
        "The system scope spans the complete pipeline: from automatic schema discovery and 18 diverse detectors to conformal false discovery rate control, "
        "streaming Half-Space Trees, and human-in-the-loop active learning.\n\n"
        "These four metrics represent our current repository state: 18 registered models, over 14,800 lines of verified code, 359 tests passing with 87% test coverage, and 21 interactive Next.js console views."
    )

    # ==========================================
    # SLIDE 4: Data & Preparation
    # ==========================================
    s4 = prs.slides[3]
    set_shape_text(s4.shapes[3], "03 · DATA & PROTOCOLS", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s4.shapes[4], "Benchmark Datasets & Split Protocols", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s4.shapes[5], "Transparent benchmarks: Synthetic controlled lab and real ULB credit transactions", "Segoe UI", Pt(11.5), CLR_LAVENDER)
    
    tf4 = s4.shapes[27].text_frame
    bullets4 = [
        "▪  Synthetic benchmark sample (880 rows, 10 columns): 80 injected anomalies (9.09% ratio) across known multivariate patterns.",
        "▪  ULB Credit Card fraud sample (20,000 rows): Preserves all 492 fraud cases (SHA-256: ede5aa5c...) for reproducible evaluation.",
        "▪  Honest chronological time-split: 70% reference fit window (14,000 rows), 30% holdout evaluation window (6,000 rows).",
        "▪  Zero future leakage: Scalers, encoders, and quantile thresholds are fit strictly on the reference window.",
        "▪  No random split inflation: Random splitting leaks same-day transaction patterns, artificially inflating PR-AUC from 0.31 to 0.65."
    ]
    for i, p in enumerate(tf4.paragraphs):
        if i < len(bullets4):
            set_run_text(p, bullets4[i], "Segoe UI", Pt(11), CLR_WHITE)

    s4.notes_slide.notes_text_frame.text = (
        "We evaluate SENTINEL under two complementary data benchmarks: a controlled 880-row synthetic dataset and a real-world 20,000-transaction credit card fraud sample containing all 492 frauds.\n\n"
        "Crucially, we enforce a strict chronological time-split protocol. We fit our models on the first 70% of time-ordered transactions and evaluate on the remaining 30% holdout.\n\n"
        "We demonstrate in our documentation that conventional random splitting leaks customer behavior across days, artificially doubling reported PR-AUC from 0.31 to 0.65. SENTINEL refuses such deceptive practices."
    )

    # ==========================================
    # SLIDE 5: Methodology
    # ==========================================
    s5 = prs.slides[4]
    set_shape_text(s5.shapes[3], "04 · METHODOLOGY & ARCHITECTURE", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s5.shapes[4], "Five stages, one audited path", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s5.shapes[5], "From raw stream to audited intervention — zero label leakage and full conformal bounds", "Segoe UI", Pt(11.5), CLR_LAVENDER)
    
    set_shape_text(s5.shapes[15], "INGEST & PROFILE", "Segoe UI", Pt(10), CLR_AMBER, bold=True)
    set_shape_text(s5.shapes[16], "· Schema discovery & health score\n· Multicollinearity condition number\n· Cryptographic SHA-256 fingerprint", "Segoe UI", Pt(9.5), CLR_WHITE)
    
    set_shape_text(s5.shapes[21], "DETECTOR ZOO", "Segoe UI", Pt(10), CLR_AMBER, bold=True)
    set_shape_text(s5.shapes[22], "· 18 classical & deep models\n· Empirical rank-Borda consensus\n· Out-of-fold XGBoost surrogate", "Segoe UI", Pt(9.5), CLR_WHITE)

    set_shape_text(s5.shapes[27], "CONFORMAL CALIBRATE", "Segoe UI", Pt(10), CLR_AMBER, bold=True)
    set_shape_text(s5.shapes[28], "· Finite-sample p-value calibration\n· Mondrian subgroup fallback\n· Benjamini-Hochberg FDR control", "Segoe UI", Pt(9.5), CLR_WHITE)

    set_shape_text(s5.shapes[33], "AUDITED EXPLAIN", "Segoe UI", Pt(10), CLR_AMBER, bold=True)
    set_shape_text(s5.shapes[34], "· TreeSHAP & 2nd-order interactions\n· SHAP vs LIME agreement index\n· LOF-constrained counterfactuals", "Segoe UI", Pt(9.5), CLR_WHITE)

    set_shape_text(s5.shapes[39], "MONITOR & ACT", "Segoe UI", Pt(10), CLR_AMBER, bold=True)
    set_shape_text(s5.shapes[40], "· Streaming Half-Space Trees\n· Sliding PSI & KS drift alarms\n· Active learning triage & model card", "Segoe UI", Pt(9.5), CLR_WHITE)

    s5.notes_slide.notes_text_frame.text = (
        "This slide depicts the five architectural stages of SENTINEL.\n\n"
        "Stage 1 profiles the raw data, computing condition numbers and SHA-256 hashes. Stage 2 executes our 18-model detector zoo and out-of-fold surrogate distillation.\n\n"
        "Stage 3 applies conformal prediction, converting ranking indices into mathematically guaranteed p-values under false discovery control.\n\n"
        "Stage 4 validates feature attributions through Deletion-AUC and LOF counterfactuals, while Stage 5 operates real-time streaming gates and active learning feedback."
    )

    # ==========================================
    # SLIDE 6: Models / Detector Zoo
    # ==========================================
    s6 = prs.slides[5]
    set_shape_text(s6.shapes[3], "05 · DETECTOR ZOO", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s6.shapes[4], "Eighteen detectors across five families", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s6.shapes[5], "src/models/ — a modular capability registry, not a hardcoded list", "Segoe UI", Pt(11.5), CLR_LAVENDER)
    
    tbl6 = s6.shapes[11].table
    # Update table headers and rows
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
                p.font.color.rgb = CLR_WHITE

    set_shape_text(s6.shapes[14], "CONSENSUS ENGINE", "Segoe UI", Pt(8.5), CLR_AMBER)
    set_shape_text(s6.shapes[15], "Empirical rank normalisation → Rank Borda count & XGBoost stacking. Raw scores from disparate statistical geometries are projected onto uniform quantiles before consensus aggregation.", "Segoe UI", Pt(9.5), CLR_WHITE)

    s6.notes_slide.notes_text_frame.text = (
        "No single algorithm fits every anomaly geometry. Anomaly detection spans points, micro-clusters, and manifold fractures.\n\n"
        "SENTINEL incorporates 18 distinct detectors across 5 statistical families: from fast linear CDF models like ECOD and HBOS to deep variational autoencoders and out-of-fold XGBoost surrogates.\n\n"
        "To combine them without unit distortion, our ensemble engine normalizes raw scores into empirical quantile ranks before executing Borda counts or hybrid stacking."
    )

    # ==========================================
    # SLIDE 7: Explainability
    # ==========================================
    s7 = prs.slides[6]
    set_shape_text(s7.shapes[3], "06 · EXPLAINABILITY", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s7.shapes[4], "Explainability that survives a scientific audit", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s7.shapes[5], "Attributions are generated, verified for faithfulness, and checked for cross-method agreement", "Segoe UI", Pt(11.5), CLR_LAVENDER)
    
    tf7 = s7.shapes[11].text_frame
    bullets7 = [
        "▪  TreeSHAP on Surrogate. Sub-millisecond exact Shapley attribution without expensive perturbation sampling.",
        "▪  2nd-Order Interactions. Pairwise non-linear interaction matrices uncover multi-feature fraud synergies.",
        "▪  Cross-Method Agreement. Quantifies Spearman correlation and top-K feature overlap between SHAP and LIME.",
        "▪  Plausible Counterfactuals. Actionable recourse paths constrained by Local Outlier Factor density and L0 sparsity.",
        "▪  Forensic Narrative Reports. Natural-language investigation briefings with cryptographic SHA-256 fingerprinting."
    ]
    for i, p in enumerate(tf7.paragraphs):
        if i < len(bullets7):
            set_run_text(p, bullets7[i], "Segoe UI", Pt(11), CLR_WHITE)

    s7.notes_slide.notes_text_frame.text = (
        "In many tools, explainability stops at a pretty waterfall chart. In SENTINEL, attribution is treated as an empirical claim that must be tested.\n\n"
        "We verify three mathematical properties: Additivity, ensuring SHAP values sum to the score; Faithfulness, testing via Deletion-AUC that masking top features collapses the anomaly score; "
        "and Stability, verifying that explanations do not fluctuate under random re-seeding.\n\n"
        "Furthermore, our counterfactual engine ensures recourse suggestions remain inside the dense, nominal data manifold using an LOF density filter."
    )

    # ==========================================
    # SLIDE 8: Results & Benchmarks
    # ==========================================
    s8 = prs.slides[7]
    set_shape_text(s8.shapes[3], "07 · EMPIRICAL RESULTS", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s8.shapes[4], "Real-world & benchmark detection performance", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s8.shapes[5], "ULB Credit Card benchmark (20k sample, 492 frauds) & 880-row controlled sample", "Segoe UI", Pt(11.5), CLR_LAVENDER)

    set_shape_text(s8.shapes[13], "0.3765", "Georgia", Pt(30), CLR_AMBER, bold=True)
    set_shape_text(s8.shapes[14], "ULB HBOS PR-AUC", "Segoe UI", Pt(7.8), CLR_MUTED)

    set_shape_text(s8.shapes[17], "0.9343", "Georgia", Pt(30), CLR_AMBER, bold=True)
    set_shape_text(s8.shapes[18], "ULB HBOS ROC-AUC", "Segoe UI", Pt(7.8), CLR_MUTED)

    set_shape_text(s8.shapes[21], "0.3108", "Georgia", Pt(30), CLR_AMBER, bold=True)
    set_shape_text(s8.shapes[22], "ULB iFOREST PR-AUC", "Segoe UI", Pt(7.8), CLR_MUTED)

    set_shape_text(s8.shapes[25], "0.02s", "Georgia", Pt(30), CLR_AMBER, bold=True)
    set_shape_text(s8.shapes[26], "HBOS FIT LATENCY", "Segoe UI", Pt(7.8), CLR_MUTED)

    set_shape_text(s8.shapes[29], "1.000", "Georgia", Pt(30), CLR_AMBER, bold=True)
    set_shape_text(s8.shapes[30], "SAMPLE PRECISION", "Segoe UI", Pt(7.8), CLR_MUTED)

    set_shape_text(s8.shapes[47], "ULB Real Benchmark: HBOS achieves PR-AUC 0.3765 (Precision@1% 0.5000), outperforming iForest (0.3108) at 14x faster speed (0.02s vs 0.28s). Random-split optimism gap is honestly disclosed.", "Segoe UI", Pt(9.5), CLR_WHITE)

    s8.notes_slide.notes_text_frame.text = (
        "Here are our verified empirical results on the ULB Credit Card fraud benchmark under honest chronological time-split evaluation.\n\n"
        "HBOS emerged as the top-performing detector, achieving PR-AUC of 0.3765 and Precision@Top-1% of 0.5000, while fitting in just 20 milliseconds. "
        "Isolation Forest followed closely with PR-AUC of 0.3108 and ROC-AUC of 0.9340.\n\n"
        "On our synthetic benchmark sample, Isolation Forest catches 44 anomalies with zero false alarms, achieving an honest 10-fold precision lift over the 9% base rate."
    )

    # ==========================================
    # SLIDE 9: Evidence & New Capabilities
    # ==========================================
    s9 = prs.slides[8]
    set_shape_text(s9.shapes[3], "08 · ADVANCED CAPABILITIES", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s9.shapes[4], "Conformal bounds, streaming, and active triage", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s9.shapes[5], "Transitioning from static batch scoring to continuous operational intelligence", "Segoe UI", Pt(11.5), CLR_LAVENDER)
    
    tf9_1 = s9.shapes[17].text_frame
    bullets9_1 = [
        "▪  Conformal p-values: Non-conformity calibration guarantees valid tail bounds under exchangeability.",
        "▪  Mondrian fallback: Automatically reverts to global calibrators when subgroup sample size n < ceil(1/α)-1.",
        "▪  Benjamini-Hochberg FDR: Explicit false discovery rate error budgeting for compliance-driven triage."
    ]
    for i, p in enumerate(tf9_1.paragraphs):
        if i < len(bullets9_1):
            set_run_text(p, bullets9_1[i], "Segoe UI", Pt(10.5), CLR_WHITE)

    tf9_2 = s9.shapes[18].text_frame
    bullets9_2 = [
        "▪  Streaming Half-Space Trees: Sub-millisecond continuous row scoring with mass-based sliding reference updates.",
        "▪  Continuous drift monitor: PSI and Kolmogorov-Smirnov statistical tests fire human-in-the-loop recalibration alarms.",
        "▪  Active learning triage: Uncertainty and disagreement sampling loop flags borderline rows for expert human review."
    ]
    for i, p in enumerate(tf9_2.paragraphs):
        if i < len(bullets9_2):
            set_run_text(p, bullets9_2[i], "Segoe UI", Pt(10.5), CLR_WHITE)

    set_shape_text(s9.shapes[20], "Root-Cause Archetyping: HDBSCAN clusters SHAP vectors to group identical failure modes, transforming raw alerts into actionable incident playbooks.", "Segoe UI", Pt(10), CLR_LAVENDER)

    s9.notes_slide.notes_text_frame.text = (
        "Slide 9 highlights the four most significant architectural additions in SENTINEL 3.0.\n\n"
        "First, conformal prediction guarantees a finite-sample error rate without assuming a parametric score distribution. "
        "If a specific demographic subgroup has too few samples, our Mondrian calibrator safely falls back to the global reference set.\n\n"
        "Second, our streaming Half-Space Trees engine ingests rows sequentially while continuous PSI monitors detect data drift.\n\n"
        "Finally, root-cause clustering automatically aggregates individual SHAP vectors into failure archetypes so operations teams resolve underlying causes rather than isolated tickets."
    )

    # ==========================================
    # SLIDE 10: Governance & MLOps
    # ==========================================
    s10 = prs.slides[9]
    set_shape_text(s10.shapes[3], "09 · GOVERNANCE & MLOPS", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s10.shapes[4], "Model cards, audit lineage, and reproducible state", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s10.shapes[5], "Every prediction, explanation, and model artifact is cryptographically traceable", "Segoe UI", Pt(11.5), CLR_LAVENDER)
    
    tf10 = s10.shapes[11].text_frame
    bullets10 = [
        "▪  Automated Mitchell et al. Model Cards: Generated dynamically, documenting intended use, bounds, and ethical factors.",
        "▪  Versioned model registry: Persists model weights, schemas, and dataset hashes in outputs/models/ for complete auditability.",
        "▪  Immutable experiment lineage: SHA-256 fingerprinting guarantees every benchmark is mathematically reproducible.",
        "▪  Surrogate fidelity certification: Tracks agreement Spearman rank correlation and top-K overlap (is_ground_truth=False).",
        "▪  Audit-ready HTML reporting: One-click export of standalone, client-shareable forensic investigation dossiers."
    ]
    for i, p in enumerate(tf10.paragraphs):
        if i < len(bullets10):
            set_run_text(p, bullets10[i], "Segoe UI", Pt(11), CLR_WHITE)

    set_shape_text(s10.shapes[14], "MLOPS INTEGRITY", "Segoe UI", Pt(8.5), CLR_AMBER)
    set_shape_text(s10.shapes[15], "SENTINEL rejects black-box deployment. Every fitted model artifact is serialized alongside its training dataset SHA-256 hash, hyperparameter envelope, and demographic robustness audit.", "Segoe UI", Pt(10), CLR_WHITE)

    s10.notes_slide.notes_text_frame.text = (
        "Enterprise machine learning requires accountability. In SENTINEL, governance is not an afterthought; it is built into the core API.\n\n"
        "We dynamically generate standardized Model Cards based on the Mitchell et al. framework, specifying demographic parity metrics and training boundaries.\n\n"
        "Every experiment and fitted weight file is tracked against its cryptographic SHA-256 input hash, and analysts can export complete, standalone HTML dossiers for regulatory audits."
    )

    # ==========================================
    # SLIDE 11: Honesty Check & Limitations
    # ==========================================
    s11 = prs.slides[10]
    set_shape_text(s11.shapes[3], "10 · HONESTY CHECK & LIMITATIONS", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s11.shapes[4], "What was solved, and what boundaries remain", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s11.shapes[5], "Transparent disclosure of completed capabilities and fundamental statistical limitations", "Segoe UI", Pt(11.5), CLR_LAVENDER)

    set_shape_text(s11.shapes[11], "RESOLVED IN SENTINEL 3.0", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s11.shapes[12], "STATISTICAL LIMITATIONS (DISCLOSED)", "Segoe UI", Pt(9.5), CLR_AMBER)

    # Left side (Resolved)
    set_shape_text(s11.shapes[15], "Conformal p-values", "Segoe UI", Pt(10), CLR_WHITE, bold=True)
    set_shape_text(s11.shapes[16], "Finite-sample non-conformity p-values with Mondrian fallback and Benjamini-Hochberg FDR.", "Segoe UI", Pt(9), CLR_MUTED)
    set_shape_text(s11.shapes[17], "IMPLEMENTED · ConformalEngine", "Segoe UI", Pt(8), CLR_AMBER)

    set_shape_text(s11.shapes[20], "Streaming detection", "Segoe UI", Pt(10), CLR_WHITE, bold=True)
    set_shape_text(s11.shapes[21], "Half-Space Trees with sliding reference mass and real-time PSI drift monitoring.", "Segoe UI", Pt(9), CLR_MUTED)
    set_shape_text(s11.shapes[22], "IMPLEMENTED · StreamingDetector", "Segoe UI", Pt(8), CLR_AMBER)

    set_shape_text(s11.shapes[25], "Deep learning detectors", "Segoe UI", Pt(10), CLR_WHITE, bold=True)
    set_shape_text(s11.shapes[26], "Autoencoder, VAE, and Deep SVDD implemented with PyTorch native reconstruction attribution.", "Segoe UI", Pt(9), CLR_MUTED)
    set_shape_text(s11.shapes[27], "IMPLEMENTED · DeepAutoencoder", "Segoe UI", Pt(8), CLR_AMBER)

    set_shape_text(s11.shapes[30], "Hybrid stacking & tuning", "Segoe UI", Pt(10), CLR_WHITE, bold=True)
    set_shape_text(s11.shapes[31], "Out-of-fold multi-model stacking with Optuna nested cross-validation hyperparameter search.", "Segoe UI", Pt(9), CLR_MUTED)
    set_shape_text(s11.shapes[32], "IMPLEMENTED · HybridStacking", "Segoe UI", Pt(8), CLR_AMBER)

    # Right side (Limitations from docs/STATISTICAL_ASSUMPTIONS.md)
    set_shape_text(s11.shapes[35], "Exchangeability assumption", "Segoe UI", Pt(10), CLR_WHITE, bold=True)
    set_shape_text(s11.shapes[36], "Conformal bounds assume test points are exchangeable with calibration data. Non-stationary drift (PSI ≥ 0.25) invalidates bounds.", "Segoe UI", Pt(9), CLR_MUTED)
    set_shape_text(s11.shapes[37], "DISCLOSED · Exchangeability gate", "Segoe UI", Pt(8), CLR_AMBER)

    set_shape_text(s11.shapes[40], "Non-causal attribution", "Segoe UI", Pt(10), CLR_WHITE, bold=True)
    set_shape_text(s11.shapes[41], "SHAP and counterfactuals measure observational model surface correlation, not physical causation. Changing a feature may not alter reality.", "Segoe UI", Pt(9), CLR_MUTED)
    set_shape_text(s11.shapes[42], "DISCLOSED · Non-causal notice", "Segoe UI", Pt(8), CLR_AMBER)

    set_shape_text(s11.shapes[45], "Curse of dimensionality", "Segoe UI", Pt(10), CLR_WHITE, bold=True)
    set_shape_text(s11.shapes[46], "Distance and density metrics (k-NN, LOF) concentrate in high dimensions (P > 50). Subspace methods (PCA, iForest, ECOD) recommended.", "Segoe UI", Pt(9), CLR_MUTED)
    set_shape_text(s11.shapes[47], "DISCLOSED · Subspace fallback", "Segoe UI", Pt(8), CLR_AMBER)

    set_shape_text(s11.shapes[50], "Benchmark label noise", "Segoe UI", Pt(10), CLR_WHITE, bold=True)
    set_shape_text(s11.shapes[51], "In real fraud data, Class=0 denotes absence of reported dispute within 48h, not mathematical proof of genuine legitimacy.", "Segoe UI", Pt(9), CLR_MUTED)
    set_shape_text(s11.shapes[52], "DISCLOSED · Ground-truth disclaimer", "Segoe UI", Pt(8), CLR_AMBER)

    s11.notes_slide.notes_text_frame.text = (
        "I believe an engineering defense is strongest when it is completely honest about its mathematical limits.\n\n"
        "On the left, notice that the four capabilities marked 'PLANNED' in our previous presentation—conformal p-values, streaming detection, deep architectures, and hybrid stacking—are now fully implemented and tested.\n\n"
        "On the right, we openly document our theoretical boundaries: conformal guarantees depend on data exchangeability, SHAP explains decision surfaces rather than physical causation, "
        "and Euclidean distance metrics degrade in high dimensions. We build software that respects these mathematical realities."
    )

    # ==========================================
    # SLIDE 12: Conclusion & Future Scope
    # ==========================================
    s12 = prs.slides[11]
    set_shape_text(s12.shapes[3], "11 · CONCLUSION & FUTURE SCOPE", "Segoe UI", Pt(9.5), CLR_AMBER)
    set_shape_text(s12.shapes[4], "Beyond detection. Into intelligence.", "Georgia", Pt(28), CLR_WHITE)
    set_shape_text(s12.shapes[5], "SENTINEL 3.0 — Evidence over scores, conservative by design, reproducible end to end", "Segoe UI", Pt(11.5), CLR_LAVENDER)
    
    set_shape_text(s12.shapes[13], "Evidence Over Scores", "Segoe UI", Pt(11), CLR_AMBER, bold=True)
    set_shape_text(s12.shapes[14], "Every alert carries verified TreeSHAP attributions, conformal p-values, and actionable counterfactual recommendations.", "Segoe UI", Pt(9.5), CLR_WHITE)

    set_shape_text(s12.shapes[17], "Autonomous Operations", "Segoe UI", Pt(11), CLR_AMBER, bold=True)
    set_shape_text(s12.shapes[18], "Streaming Half-Space Trees, sliding PSI drift monitoring, and active learning queues close the loop between detection and triage.", "Segoe UI", Pt(9.5), CLR_WHITE)

    set_shape_text(s12.shapes[21], "Scientific Reproducibility", "Segoe UI", Pt(11), CLR_AMBER, bold=True)
    set_shape_text(s12.shapes[22], "359 passing tests (87% coverage), 18 model capabilities, and cryptographic SHA-256 experiment tracking.", "Segoe UI", Pt(9.5), CLR_WHITE)

    set_shape_text(s12.shapes[25], "Thank you · Questions Welcome", "Segoe UI", Pt(12), CLR_AMBER, bold=True)
    set_shape_text(s12.shapes[26], "Future Scope: Graph anomaly detection (GNNs), dynamic business cost learning, and online conformal martingales.", "Segoe UI", Pt(9.5), CLR_WHITE)
    set_shape_text(s12.shapes[27], "The repository, test suite, and live console are fully verified and reproducible.", "Segoe UI", Pt(8.5), CLR_MUTED)

    s12.notes_slide.notes_text_frame.text = (
        "To conclude, SENTINEL 3.0 achieves three foundational milestones.\n\n"
        "First, evidence over scores: every alert is supported by verified attributions and rigorous conformal error guarantees.\n\n"
        "Second, autonomous operations: our streaming and active learning loops enable human analysts to triage and retrain models continuously.\n\n"
        "Third, total reproducibility: 359 tests at 87% coverage ensure production reliability. In future work, we plan to extend this platform to graph neural networks for transaction topologies. Thank you."
    )

    # ==========================================
    # SLIDE 13: Add New Slide — Q&A / Viva Defense
    # ==========================================
    # We add a new slide using the blank layout (layout 6)
    blank_layout = prs.slide_layouts[6]
    s13 = prs.slides.add_slide(blank_layout)
    
    # 1. Background dark card
    bg = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = CLR_BG_DARK
    bg.line.color.rgb = CLR_BG_DARK

    # 2. Header pill shape
    header_pill = s13.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(4.0), Inches(0.4))
    hp_p = header_pill.text_frame.paragraphs[0]
    set_run_text(hp_p, "12 · QUESTIONS & ANSWERS", "Segoe UI", Pt(9.5), CLR_AMBER, bold=True)

    # 3. Title & Subtitle
    title_box = s13.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.7))
    tp = title_box.text_frame.paragraphs[0]
    set_run_text(tp, "Viva Defense & Technical Discussion", "Georgia", Pt(28), CLR_WHITE, bold=True)

    sub_box = s13.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.5), Inches(0.4))
    sp = sub_box.text_frame.paragraphs[0]
    set_run_text(sp, "Five architectural justifications and mathematical defenses prepared for panel examination", "Segoe UI", Pt(11.5), CLR_LAVENDER)

    # 4. Five Questions formatted into neat cards
    questions = [
        ("Q1: Why refuse to call unsupervised anomaly scores 'probabilities'?",
         "Unsupervised scores measure geometric, density, or tree path lengths without a calibrated reference null. Calling them probabilities implies an uncalibrated posterior P(y=1|x) which leads to arbitrary thresholding. We use conformal p-values instead."),
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
    card_height = Inches(0.9)
    y_start = Inches(1.85)
    gap = Inches(0.98)

    for q_idx, (q_text, a_text) in enumerate(questions):
        y_pos = y_start + (q_idx * gap)
        
        # Background rect for question card
        q_card = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, card_left, y_pos, card_width, card_height)
        q_card.fill.solid()
        q_card.fill.fore_color.rgb = CLR_CARD_BG
        q_card.line.color.rgb = CLR_CARD_BORDER
        q_card.line.width = Pt(1)

        # Question title
        tb = s13.shapes.add_textbox(card_left + Inches(0.2), y_pos + Inches(0.06), card_width - Inches(0.4), Inches(0.35))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p0 = tf.paragraphs[0]
        set_run_text(p0, q_text, "Segoe UI", Pt(11), CLR_AMBER, bold=True)

        # Answer text
        tb2 = s13.shapes.add_textbox(card_left + Inches(0.2), y_pos + Inches(0.38), card_width - Inches(0.4), Inches(0.45))
        tf2 = tb2.text_frame
        tf2.word_wrap = True
        tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0
        p1 = tf2.paragraphs[0]
        set_run_text(p1, a_text, "Segoe UI", Pt(9.5), CLR_WHITE)

    # 5. Footer on Slide 13
    ft_box = s13.shapes.add_textbox(Inches(0.8), Inches(6.9), Inches(8.0), Inches(0.3))
    ftp = ft_box.text_frame.paragraphs[0]
    set_run_text(ftp, "SENTINEL 3.0  ·  Autonomous Anomaly Intelligence Platform", "Segoe UI", Pt(8.5), CLR_MUTED)

    pg_box = s13.shapes.add_textbox(Inches(11.5), Inches(6.9), Inches(1.0), Inches(0.3))
    pgp = pg_box.text_frame.paragraphs[0]
    set_run_text(pgp, "13 / 13", "Segoe UI", Pt(8.5), CLR_MUTED)

    # Speaker notes for Slide 13
    import copy
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
    prs.save(pptx_path)
    print(f"Successfully updated presentation: {pptx_path} (Total slides: {len(prs.slides)})")

if __name__ == "__main__":
    pptx_file = r"C:\Users\ADMIN\Documents\ML PROJECT\explainable-anomaly-detection\presentation\Explainable_Anomaly_Detection_12slides.pptx"
    update_presentation(pptx_file)

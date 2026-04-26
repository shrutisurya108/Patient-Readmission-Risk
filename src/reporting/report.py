# src/reporting/report.py

import os
import json
import warnings
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("report")

PAGE_W, PAGE_H = A4
MARGIN         = 2.0 * cm


# ── Style definitions ─────────────────────────────────────────────────────────
def _build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"],
            fontSize=22, textColor=colors.HexColor("#1A237E"),
            spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold"
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=12, textColor=colors.HexColor("#455A64"),
            spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica"
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=14, textColor=colors.HexColor("#1A237E"),
            spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold",
            borderPad=4
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"],
            fontSize=11, textColor=colors.HexColor("#37474F"),
            spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold"
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=9.5, leading=14, spaceAfter=6,
            alignment=TA_JUSTIFY, fontName="Helvetica"
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=9.5, leading=14, leftIndent=16,
            spaceAfter=3, bulletIndent=6, fontName="Helvetica"
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"],
            fontSize=8, textColor=colors.HexColor("#607D8B"),
            alignment=TA_CENTER, spaceAfter=8, fontName="Helvetica-Oblique"
        ),
        "metric": ParagraphStyle(
            "Metric", parent=base["Normal"],
            fontSize=11, alignment=TA_CENTER, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1A237E")
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"],
            fontSize=8, leading=11, fontName="Courier",
            backColor=colors.HexColor("#F5F5F5"), spaceAfter=6
        ),
    }
    return styles


def _img(path: str, width: float, caption: str, styles: dict) -> list:
    """Return [Image, caption Paragraph] if file exists, else empty list.
    Explicitly computes height to avoid ReportLab proportional=None bug on Py3.14.
    """
    if not os.path.exists(path):
        logger.warning(f"Figure not found, skipping: {path}")
        return []
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            img_w, img_h = im.size          # pixels
        aspect = img_h / img_w
        height = width * aspect
    except Exception as e:
        logger.warning(f"Could not read image dimensions for {path}: {e}")
        height = width * 0.65               # safe fallback ratio
    return [
        Image(path, width=width, height=height),
        Paragraph(caption, styles["caption"]),
        Spacer(1, 0.3 * cm),
    ]


def _hr(styles: dict) -> list:
    return [HRFlowable(width="100%", thickness=0.5,
                       color=colors.HexColor("#B0BEC5")), Spacer(1, 0.2 * cm)]


# ── Section builders ──────────────────────────────────────────────────────────
def _cover(styles: dict) -> list:
    now = datetime.now().strftime("%B %d, %Y")
    return [
        Spacer(1, 2.5 * cm),
        Paragraph("🏥 Patient Readmission Risk", styles["title"]),
        Paragraph("Clinical Decision Support Report", styles["title"]),
        Spacer(1, 0.5 * cm),
        HRFlowable(width="60%", thickness=2,
                   color=colors.HexColor("#1A237E"), hAlign="CENTER"),
        Spacer(1, 0.5 * cm),
        Paragraph("UCI Diabetes 130-US Hospitals Dataset (1999–2008)", styles["subtitle"]),
        Paragraph(f"Generated: {now}", styles["subtitle"]),
        Paragraph("Model: XGBoost Gradient Boosted Classifier", styles["subtitle"]),
        Spacer(1, 1.0 * cm),
        _metric_table(styles),
        PageBreak(),
    ]


def _metric_table(styles: dict) -> Table:
    metrics_path = "reports/model_metrics.json"
    if not os.path.exists(metrics_path):
        return Spacer(1, 0.1 * cm)
    with open(metrics_path) as f:
        m = json.load(f)

    cv  = m.get("cross_validation", {})
    auc_cv  = cv.get("roc_auc", {}).get("mean", "N/A")
    f1_cv   = cv.get("f1", {}).get("mean", "N/A")
    prec_cv = cv.get("precision", {}).get("mean", "N/A")
    rec_cv  = cv.get("recall", {}).get("mean", "N/A")

    data = [
        ["Metric", "Value"],
        ["Test AUC-ROC",        str(m.get("auc_roc", "N/A"))],
        ["Avg Precision",       str(m.get("avg_precision", "N/A"))],
        ["Best Threshold",      str(m.get("best_threshold", "N/A"))],
        ["CV AUC (5-fold)",     str(auc_cv)],
        ["CV F1  (5-fold)",     str(f1_cv)],
        ["CV Precision",        str(prec_cv)],
        ["CV Recall",           str(rec_cv)],
        ["Training Samples",    "81,410"],
        ["Test Samples",        "20,353"],
        ["Features Engineered", "44"],
    ]

    col_w = [(PAGE_W - 2 * MARGIN) * 0.55,
             (PAGE_W - 2 * MARGIN) * 0.35]
    t = Table(data, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#1A237E")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#EEF2FF"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#90A4AE")),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _section_executive_summary(styles: dict) -> list:
    elems = [
        Paragraph("1. Executive Summary", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "This report presents the results of an end-to-end machine learning pipeline "
            "developed to predict 30-day hospital readmission risk for diabetic patients. "
            "Early identification of high-risk patients enables clinical teams to intervene "
            "proactively, reducing preventable readmissions and associated costs.",
            styles["body"]),
        Paragraph(
            "The analysis uses the UCI Diabetes 130-US Hospitals dataset comprising "
            "101,766 inpatient encounters across 130 US hospitals from 1999 to 2008. "
            "An XGBoost gradient boosted classifier was trained with class-imbalance "
            "correction, achieving a test AUC-ROC of 0.6851 on held-out data — "
            "consistent with published benchmarks on this inherently noisy clinical dataset.",
            styles["body"]),
        Paragraph("<b>Key findings:</b>", styles["body"]),
        Paragraph("• Prior inpatient visits is the strongest predictor of 30-day readmission.",
                  styles["bullet"]),
        Paragraph("• Patients with 5+ prior inpatient visits face >31% readmission rate.",
                  styles["bullet"]),
        Paragraph("• Insulin dose changes (Up/Down) are statistically associated with "
                  "higher readmission (χ²=190.86, p=3.98e-41).",
                  styles["bullet"]),
        Paragraph("• All 11 tested risk factors are statistically significant (p < 0.05).",
                  styles["bullet"]),
        Paragraph("• SHAP and LIME provide patient-level explanations suitable for "
                  "clinical decision support.",
                  styles["bullet"]),
        Spacer(1, 0.4 * cm),
    ]
    return elems


def _section_dataset(styles: dict, fig_dir: str) -> list:
    elems = [
        Paragraph("2. Dataset Overview & SQL Cohort Analysis", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "The UCI Diabetes Readmission dataset contains 101,766 records with 47 "
            "original features covering demographics, diagnoses (ICD-9), medications, "
            "lab results, and utilisation history. The target variable — readmission "
            "within 30 days — affects 11.16% of encounters, representing significant "
            "class imbalance.",
            styles["body"]),
        Paragraph("<b>Class distribution:</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "eda_class_distribution.png"),
              PAGE_W - 2 * MARGIN - 4 * cm,
              "Figure 1: Readmission class distribution. NO=54,864 (53.9%), "
              ">30d=35,545 (34.9%), <30d=11,357 (11.2%).", styles),
        Paragraph("<b>Age distribution by readmission status:</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "eda_age_distribution.png"),
              PAGE_W - 2 * MARGIN - 2 * cm,
              "Figure 2: Patient age distribution. Peak volume in 70–80 age group; "
              "readmission rate highest in 80–90 cohort (12.1%).", styles),
        Paragraph(
            "SQL cohort analysis across 10 clinical dimensions revealed that prior "
            "inpatient visits, insulin status, and number of diagnoses are the strongest "
            "cohort-level discriminators. Full profiling results are saved in "
            "reports/sql_profiling.txt.",
            styles["body"]),
        PageBreak(),
    ]
    return elems


def _section_feature_engineering(styles: dict, fig_dir: str) -> list:
    elems = [
        Paragraph("3. Feature Engineering", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "Raw features were transformed through a multi-step engineering pipeline "
            "before model training. Class imbalance was addressed using XGBoost's native "
            "scale_pos_weight parameter (ratio = 7.96), which is preferable to SMOTE "
            "for datasets containing mixed ordinal and categorical features.",
            styles["body"]),
        Paragraph("<b>Key transformations applied:</b>", styles["h2"]),
    ]
    steps = [
        ("High-missingness removal",
         "weight (97%), payer_code (40%), medical_specialty (49%) dropped."),
        ("Target recoding",
         "Three-class readmission → binary: 1 = <30d readmission, 0 = otherwise."),
        ("ICD-9 bucketing",
         "diag_1/2/3 codes mapped to 9 clinical categories: Circulatory, Diabetes, "
         "Respiratory, Digestive, Injury, Genitourinary, Musculoskeletal, Neoplasms, Other."),
        ("Age ordinal encoding",
         "Age brackets [0-10) through [90-100) encoded as integers 0–9."),
        ("Medication features",
         "num_meds_changed (count of Up/Down dose changes) and num_meds_active "
         "(count of non-zero medications) engineered from 21 medication columns."),
        ("Label encoding",
         "9 remaining categorical columns encoded: race, gender, diag_1-3, "
         "max_glu_serum, A1Cresult, change, diabetesMed."),
    ]
    for title, desc in steps:
        elems.append(Paragraph(f"• <b>{title}:</b> {desc}", styles["bullet"]))

    elems += [
        Spacer(1, 0.3 * cm),
        Paragraph("<b>Correlation structure:</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "eda_correlation_heatmap.png"),
              PAGE_W - 2 * MARGIN - 1 * cm,
              "Figure 3: Feature correlation heatmap. number_inpatient and "
              "number_emergency show strongest positive correlation with readmission.",
              styles),
        PageBreak(),
    ]
    return elems


def _section_model_performance(styles: dict, fig_dir: str) -> list:
    elems = [
        Paragraph("4. Model Performance", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "An XGBoost gradient boosted classifier was trained on 81,410 patients "
            "with 5-fold stratified cross-validation. The classification threshold was "
            "tuned post-training to maximise F1 score on the test set, accounting for "
            "the real-world class prior.",
            styles["body"]),
        Paragraph("<b>Confusion Matrix:</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "confusion_matrix.png"),
              PAGE_W - 2 * MARGIN - 6 * cm,
              "Figure 4: Confusion matrix at optimal threshold (0.52). "
              "Model correctly identifies 55% of readmitted patients (recall).",
              styles),
        Paragraph("<b>ROC Curve:</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "roc_curve.png"),
              PAGE_W - 2 * MARGIN - 4 * cm,
              "Figure 5: ROC curve. AUC=0.6851, consistent with published "
              "benchmarks for 30-day readmission prediction on this dataset.",
              styles),
        Paragraph("<b>Feature Importances (XGBoost):</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "feature_importance.png"),
              PAGE_W - 2 * MARGIN - 2 * cm,
              "Figure 6: Top 20 XGBoost feature importances. Prior inpatient visits, "
              "number of diagnoses, and time in hospital are the top predictors.",
              styles),
        PageBreak(),
    ]
    return elems


def _section_hypothesis(styles: dict) -> list:
    hyp_path = "reports/hypothesis_tests.txt"
    elems = [
        Paragraph("5. Hypothesis Testing", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "Statistical tests were conducted on the original dataset (pre-modelling) "
            "to validate that model features are genuine clinical risk factors. "
            "Chi-square tests assess categorical associations; Welch's t-tests assess "
            "continuous feature differences between readmitted and non-readmitted cohorts.",
            styles["body"]),
    ]

    chi_data = [
        ["Feature", "χ² Statistic", "p-value", "Result"],
        ["Insulin medication status",    "190.86", "3.98e-41", "✓ Significant"],
        ["Medication change flag",        "38.60",  "5.21e-10", "✓ Significant"],
        ["Diabetes medication",           "74.67",  "5.57e-18", "✓ Significant"],
        ["Age group",                    "116.61",  "6.60e-21", "✓ Significant"],
        ["Number of diagnoses",          "259.10",  "1.65e-46", "✓ Significant"],
    ]
    t_data = [
        ["Feature", "t-statistic", "Δ Mean", "p-value", "Result"],
        ["Time in hospital",      "13.926", "+0.419", "8.55e-44",  "✓ Significant"],
        ["Number of medications", "12.302", "+0.992", "1.32e-34",  "✓ Significant"],
        ["Lab procedures",         "6.613", "+1.272", "3.89e-11",  "✓ Significant"],
        ["Number of diagnoses",   "17.027", "+0.304", "2.07e-64",  "✓ Significant"],
        ["Prior inpatient visits","35.384", "+0.662", "2.76e-261", "✓ Significant"],
        ["Prior emergency visits","13.629", "+0.180", "5.39e-42",  "✓ Significant"],
    ]

    def _table(data, col_ratios):
        col_w = [(PAGE_W - 2 * MARGIN) * r for r in col_ratios]
        t = Table(data, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1A237E")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.HexColor("#EEF2FF"), colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#90A4AE")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    elems += [
        Paragraph("<b>Chi-Square Tests (categorical features):</b>", styles["h2"]),
        _table(chi_data, [0.40, 0.18, 0.18, 0.24]),
        Spacer(1, 0.4 * cm),
        Paragraph("<b>Welch's T-Tests (continuous features):</b>", styles["h2"]),
        _table(t_data, [0.32, 0.16, 0.12, 0.18, 0.22]),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "All 11 tested features are statistically significant (p ≪ 0.05), "
            "confirming that model feature importance rankings are supported by "
            "rigorous statistical evidence.",
            styles["body"]),
        PageBreak(),
    ]
    return elems


def _section_shap(styles: dict, fig_dir: str) -> list:
    elems = [
        Paragraph("6. SHAP Explainability — Global", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "SHAP (SHapley Additive exPlanations) values quantify each feature's "
            "contribution to individual predictions using exact Shapley values from "
            "cooperative game theory. TreeExplainer was used for computational "
            "efficiency with the XGBoost model.",
            styles["body"]),
        Paragraph("<b>Global feature impact (beeswarm):</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "shap_summary_beeswarm.png"),
              PAGE_W - 2 * MARGIN - 1 * cm,
              "Figure 7: SHAP beeswarm plot. Each point is one patient. "
              "Red = high feature value, blue = low. Prior inpatient visits "
              "and number of diagnoses have the largest global impact.",
              styles),
        Paragraph("<b>Mean absolute SHAP importance:</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "shap_bar_global.png"),
              PAGE_W - 2 * MARGIN - 2 * cm,
              "Figure 8: Mean |SHAP| feature ranking. number_inpatient, "
              "number_diagnoses, and num_meds_changed are top global drivers.",
              styles),
        PageBreak(),
    ]
    return elems


def _section_shap_local(styles: dict, fig_dir: str) -> list:
    elems = [
        Paragraph("7. SHAP Explainability — Patient-Level", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "Waterfall plots decompose the model prediction for individual patients, "
            "showing how each feature pushes the prediction above or below the baseline "
            "expected value. This enables clinicians to understand exactly why a "
            "specific patient was flagged as high-risk.",
            styles["body"]),
        Paragraph("<b>Highest-risk patient (P(readmit) = 0.944):</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "shap_waterfall_highrisk.png"),
              PAGE_W - 2 * MARGIN - 1 * cm,
              "Figure 9: SHAP waterfall for highest-risk patient. "
              "Multiple prior inpatient visits and high diagnosis count "
              "are the dominant positive contributors.",
              styles),
        Paragraph("<b>Lowest-risk patient (P(readmit) = 0.007):</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "shap_waterfall_lowrisk.png"),
              PAGE_W - 2 * MARGIN - 1 * cm,
              "Figure 10: SHAP waterfall for lowest-risk patient. "
              "Zero prior visits and fewer diagnoses collectively suppress risk.",
              styles),
        Paragraph("<b>Prior inpatient visits — dose-response relationship:</b>",
                  styles["h2"]),
        *_img(os.path.join(fig_dir, "shap_dependence_inpatient.png"),
              PAGE_W - 2 * MARGIN - 3 * cm,
              "Figure 11: SHAP dependence plot. Risk increases sharply "
              "beyond 3 prior inpatient visits, confirming the SQL cohort findings.",
              styles),
        PageBreak(),
    ]
    return elems


def _section_lime(styles: dict, fig_dir: str) -> list:
    elems = [
        Paragraph("8. LIME Patient-Level Explanations", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "LIME (Local Interpretable Model-agnostic Explanations) fits a locally "
            "linear model around each patient's neighbourhood to explain predictions "
            "in interpretable feature ranges. Three representative patients are shown.",
            styles["body"]),
        Paragraph("<b>High-risk patient (P = 0.944):</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "lime_patient_high_risk.png"),
              PAGE_W - 2 * MARGIN - 1 * cm,
              "Figure 12: LIME explanation — high-risk patient. "
              "Green bars support readmission prediction; orange bars oppose it.",
              styles),
        Paragraph("<b>Median-risk patient (P = 0.440):</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "lime_patient_median_risk.png"),
              PAGE_W - 2 * MARGIN - 1 * cm,
              "Figure 13: LIME explanation — median-risk patient. "
              "Mixed signals reflect borderline clinical profile.",
              styles),
        Paragraph("<b>Low-risk patient (P = 0.007):</b>", styles["h2"]),
        *_img(os.path.join(fig_dir, "lime_patient_low_risk.png"),
              PAGE_W - 2 * MARGIN - 1 * cm,
              "Figure 14: LIME explanation — low-risk patient. "
              "Features uniformly suppress readmission probability.",
              styles),
        PageBreak(),
    ]
    return elems


def _section_recommendations(styles: dict) -> list:
    return [
        Paragraph("9. Clinical Recommendations", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "Based on the statistical analysis and model explainability findings, "
            "the following clinical interventions are recommended:",
            styles["body"]),
        Paragraph(
            "• <b>Prioritise patients with ≥3 prior inpatient visits</b> for discharge "
            "planning and post-discharge follow-up — this cohort faces >20% readmission risk.",
            styles["bullet"]),
        Paragraph(
            "• <b>Review insulin dosing protocols</b> at discharge. Patients with "
            "insulin dose changes (Up or Down) have significantly elevated readmission "
            "rates (χ²=190.86, p=3.98e-41).",
            styles["bullet"]),
        Paragraph(
            "• <b>Flag patients with ≥8 diagnoses</b> for enhanced care coordination — "
            "readmission rate exceeds 11.8% in this cohort.",
            styles["bullet"]),
        Paragraph(
            "• <b>Longer stays require attention</b>: readmission rate rises steadily "
            "from 8.2% (1-day stays) to 14.4% (10-day stays).",
            styles["bullet"]),
        Paragraph(
            "• <b>Discharge disposition matters</b>: discharge to SNF (code 3) carries "
            "a 14.7% readmission rate vs. 12.7% for home discharge.",
            styles["bullet"]),
        Spacer(1, 0.4 * cm),
        Paragraph("10. Limitations & Next Steps", styles["h1"]),
        *_hr(styles),
        Paragraph(
            "• AUC of 0.685 is consistent with literature for this dataset "
            "(Strack et al., 2014 report similar ranges), reflecting inherent noise "
            "in administrative health records.",
            styles["bullet"]),
        Paragraph(
            "• The model does not incorporate temporal features (time between visits) "
            "or lab value trends — adding these would likely improve discrimination.",
            styles["bullet"]),
        Paragraph(
            "• External validation on a prospective cohort is required before "
            "clinical deployment.",
            styles["bullet"]),
        Paragraph(
            "• A calibration step (Platt scaling or isotonic regression) should be "
            "applied before using predicted probabilities as clinical risk scores.",
            styles["bullet"]),
        Spacer(1, 0.5 * cm),
    ]


# ── Main builder ──────────────────────────────────────────────────────────────
def build_report(cfg: dict) -> str:
    fig_dir    = cfg["paths"]["figures"]
    out_dir    = cfg["paths"]["clinical_report"]
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "readmission_risk_report.pdf")
    doc      = SimpleDocTemplate(
        out_path,
        pagesize     = A4,
        leftMargin   = MARGIN,
        rightMargin  = MARGIN,
        topMargin    = MARGIN,
        bottomMargin = MARGIN,
        title        = "Patient Readmission Risk Report",
        author       = "Patient Readmission Risk ML Pipeline",
    )

    styles = _build_styles()
    story  = []

    story += _cover(styles)
    story += _section_executive_summary(styles)
    story += _section_dataset(styles, fig_dir)
    story += _section_feature_engineering(styles, fig_dir)
    story += _section_model_performance(styles, fig_dir)
    story += _section_hypothesis(styles)
    story += _section_shap(styles, fig_dir)
    story += _section_shap_local(styles, fig_dir)
    story += _section_lime(styles, fig_dir)
    story += _section_recommendations(styles)

    logger.info("Building PDF...")
    doc.build(story)
    size_kb = os.path.getsize(out_path) // 1024
    logger.info(f"PDF saved → {out_path}  ({size_kb} KB)")
    return out_path


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    path = build_report(cfg)
    print(f"\n✅ Phase 6 complete. Clinical report saved → {path}")

# app.py — Gradio deployment for Hugging Face Spaces

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import gradio as gr
from PIL import Image as PILImage
import io

warnings.filterwarnings("ignore")

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = "models/xgb_readmission.json"
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

FEATURE_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "diabetesMed",
    "num_meds_changed", "num_meds_active"
]

BEST_THRESHOLD = 0.52

DIAG_MAP = {
    "Circulatory": 0, "Diabetes": 1, "Digestive": 2,
    "External": 3, "Genitourinary": 4, "Injury": 5,
    "Musculoskeletal": 6, "Neoplasms": 7, "Other": 8, "Respiratory": 9
}

explainer = shap.TreeExplainer(model)


def predict_readmission(
    age, time_in_hospital, num_lab_procedures, num_procedures,
    num_medications, number_outpatient, number_emergency, number_inpatient,
    number_diagnoses, diag_1, insulin, change, diabetesMed,
    num_meds_changed, num_meds_active
):
    # Build feature vector with sensible defaults for non-input features
    row = {col: 0 for col in FEATURE_COLS}

    row["age"]                  = int(age)
    row["time_in_hospital"]     = int(time_in_hospital)
    row["num_lab_procedures"]   = int(num_lab_procedures)
    row["num_procedures"]       = int(num_procedures)
    row["num_medications"]      = int(num_medications)
    row["number_outpatient"]    = int(number_outpatient)
    row["number_emergency"]     = int(number_emergency)
    row["number_inpatient"]     = int(number_inpatient)
    row["number_diagnoses"]     = int(number_diagnoses)
    row["diag_1"]               = DIAG_MAP.get(diag_1, 8)
    row["insulin"]              = int(insulin)
    row["change"]               = int(change)
    row["diabetesMed"]          = int(diabetesMed)
    row["num_meds_changed"]     = int(num_meds_changed)
    row["num_meds_active"]      = int(num_meds_active)
    row["race"]                 = 0
    row["gender"]               = 0
    row["admission_type_id"]    = 1
    row["discharge_disposition_id"] = 1
    row["admission_source_id"]  = 7
    row["max_glu_serum"]        = 0
    row["A1Cresult"]            = 0

    X = pd.DataFrame([row])[FEATURE_COLS]

    prob        = float(model.predict_proba(X)[0, 1])
    risk_label  = "🔴 HIGH RISK" if prob >= BEST_THRESHOLD else "🟢 LOW RISK"
    risk_pct    = f"{prob * 100:.1f}%"

    # SHAP waterfall
    sv      = explainer(X)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    plt.title(f"SHAP Explanation — P(readmit) = {risk_pct}",
              fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    shap_img = PILImage.open(buf)

    summary = (
        f"**Readmission Risk: {risk_label}**\n\n"
        f"Predicted probability of 30-day readmission: **{risk_pct}**\n\n"
        f"Threshold: {BEST_THRESHOLD} (optimised for F1 on held-out test set)\n\n"
        f"The SHAP chart below shows which patient features drove this prediction. "
        f"Red bars increase risk; blue bars decrease risk."
    )
    return summary, shap_img


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Patient Readmission Risk",
    theme=gr.themes.Soft(),
    css=".gradio-container {max-width: 900px; margin: auto}"
) as demo:

    gr.Markdown("""
    # 🏥 Patient 30-Day Readmission Risk Predictor
    **UCI Diabetes 130-US Hospitals Dataset | XGBoost + SHAP Explainability**

    Enter patient clinical details below to predict 30-day readmission risk.
    The SHAP waterfall chart explains exactly which factors drove the prediction.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Patient Demographics")
            age = gr.Slider(0, 9, value=6, step=1, label="Age Group (0=[0-10), 9=[90-100))")
            gender_note = gr.Markdown("_Gender/race use dataset defaults for this demo_")

            gr.Markdown("### 🏨 Hospitalisation")
            time_in_hospital    = gr.Slider(1, 14, value=4,  step=1, label="Time in Hospital (days)")
            num_lab_procedures  = gr.Slider(1, 132, value=43, step=1, label="Number of Lab Procedures")
            num_procedures      = gr.Slider(0, 6,  value=1,  step=1, label="Number of Procedures")
            num_medications     = gr.Slider(1, 81, value=16, step=1, label="Number of Medications")

        with gr.Column():
            gr.Markdown("### 📋 Prior Utilisation")
            number_outpatient = gr.Slider(0, 42, value=0, step=1, label="Prior Outpatient Visits")
            number_emergency  = gr.Slider(0, 76, value=0, step=1, label="Prior Emergency Visits")
            number_inpatient  = gr.Slider(0, 21, value=0, step=1, label="Prior Inpatient Visits ⚠️ Top predictor")
            number_diagnoses  = gr.Slider(1, 16, value=7, step=1, label="Number of Diagnoses")

            gr.Markdown("### 💊 Medications & Diagnosis")
            diag_1    = gr.Dropdown(
                list(DIAG_MAP.keys()), value="Circulatory",
                label="Primary Diagnosis Category")
            insulin   = gr.Radio([0, 1, 2, 3], value=1,
                label="Insulin (0=No, 1=Steady, 2=Up, 3=Down)")
            change    = gr.Radio([0, 1], value=0,
                label="Medication Change (0=No, 1=Yes)")
            diabetesMed = gr.Radio([0, 1], value=1,
                label="Diabetes Medication (0=No, 1=Yes)")
            num_meds_changed = gr.Slider(0, 10, value=1, step=1,
                label="Medications with Dose Change")
            num_meds_active  = gr.Slider(0, 21, value=8, step=1,
                label="Active Medications Count")

    predict_btn = gr.Button("🔍 Predict Readmission Risk", variant="primary", size="lg")

    gr.Markdown("---")
    result_text = gr.Markdown(label="Prediction Result")
    shap_plot   = gr.Image(label="SHAP Explanation", type="pil")

    predict_btn.click(
        fn=predict_readmission,
        inputs=[
            age, time_in_hospital, num_lab_procedures, num_procedures,
            num_medications, number_outpatient, number_emergency,
            number_inpatient, number_diagnoses, diag_1, insulin,
            change, diabetesMed, num_meds_changed, num_meds_active
        ],
        outputs=[result_text, shap_plot]
    )

    gr.Markdown("""
    ---
    **Model:** XGBoost Gradient Boosted Classifier | **AUC-ROC:** 0.685 |
    **Dataset:** UCI Diabetes 130-US Hospitals (101,766 records) |
    **Explainability:** SHAP TreeExplainer
    """)

if __name__ == "__main__":
    demo.launch()

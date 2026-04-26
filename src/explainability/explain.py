# src/explainability/explain.py

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import shap
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer

from src.utils.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger("explain")

FIG_DPI  = 150
TITLE_FS = 13


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_artifacts(processed_dir: str, model_path: str):
    logger.info("Loading model and test data...")
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"X_test: {X_test.shape} | X_train: {X_train.shape}")
    return model, X_train, X_test, y_test


# ── SHAP ──────────────────────────────────────────────────────────────────────
def run_shap(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
             fig_dir: str, processed_dir: str) -> np.ndarray:
    os.makedirs(fig_dir, exist_ok=True)
    logger.info("Computing SHAP values with TreeExplainer...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)          # Explanation object
    sv_array    = shap_values.values         # (n_samples, n_features)

    # Save raw SHAP values for use in PDF report
    np.save(os.path.join(processed_dir, "shap_values.npy"), sv_array)
    logger.info(f"SHAP values saved → {processed_dir}/shap_values.npy  "
                f"shape={sv_array.shape}")

    # 1 — Beeswarm summary plot (global)
    logger.info("Generating SHAP beeswarm summary plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Global Feature Impact on Readmission Risk",
              fontsize=TITLE_FS, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(fig_dir, "shap_summary_beeswarm.png")
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved → {path}")

    # 2 — Bar plot (mean absolute SHAP)
    logger.info("Generating SHAP global bar plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("SHAP Global Feature Importance (Mean |SHAP|)",
              fontsize=TITLE_FS, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(fig_dir, "shap_bar_global.png")
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved → {path}")

    # 3 — Waterfall: highest-risk patient
    y_prob        = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(y_prob))
    low_risk_idx  = int(np.argmin(y_prob))

    logger.info(f"High-risk patient idx={high_risk_idx}  "
                f"prob={y_prob[high_risk_idx]:.4f}")
    logger.info(f"Low-risk  patient idx={low_risk_idx}   "
                f"prob={y_prob[low_risk_idx]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[high_risk_idx], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Highest-Risk Patient "
              f"(P(readmit)={y_prob[high_risk_idx]:.3f})",
              fontsize=TITLE_FS, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(fig_dir, "shap_waterfall_highrisk.png")
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved → {path}")

    # 4 — Waterfall: lowest-risk patient
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[low_risk_idx], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Lowest-Risk Patient "
              f"(P(readmit)={y_prob[low_risk_idx]:.3f})",
              fontsize=TITLE_FS, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(fig_dir, "shap_waterfall_lowrisk.png")
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved → {path}")

    # 5 — Dependence plot: number_inpatient (top clinical predictor)
    dep_feature = "number_inpatient"
    if dep_feature in X_test.columns:
        logger.info(f"Generating SHAP dependence plot for '{dep_feature}'...")
        feat_idx = list(X_test.columns).index(dep_feature)
        fig, ax  = plt.subplots(figsize=(9, 6))
        shap.dependence_plot(
            feat_idx, sv_array, X_test,
            feature_names=list(X_test.columns),
            ax=ax, show=False
        )
        ax.set_title(
            f"SHAP Dependence — {dep_feature.replace('_', ' ').title()}",
            fontsize=TITLE_FS, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(fig_dir, "shap_dependence_inpatient.png")
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close("all")
        logger.info(f"Saved → {path}")

    return sv_array


# ── LIME ──────────────────────────────────────────────────────────────────────
def run_lime(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
             y_test: pd.Series, fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    logger.info("Initialising LIME TabularExplainer...")

    # Use a background sample for efficiency
    bg_size  = min(500, len(X_train))
    bg_data  = X_train.sample(n=bg_size, random_state=42).values

    explainer = LimeTabularExplainer(
        training_data   = bg_data,
        feature_names   = list(X_test.columns),
        class_names     = ["Not Readmitted", "Readmitted <30d"],
        mode            = "classification",
        discretize_continuous = True,
        random_state    = 42
    )

    y_prob = model.predict_proba(X_test)[:, 1]

    # Select 3 representative patients:
    # highest risk, median risk, lowest risk
    sorted_idx   = np.argsort(y_prob)
    median_idx   = int(sorted_idx[len(sorted_idx) // 2])
    high_idx     = int(np.argmax(y_prob))
    low_idx      = int(np.argmin(y_prob))
    patients     = [
        (high_idx,   "High-Risk",    y_prob[high_idx]),
        (median_idx, "Median-Risk",  y_prob[median_idx]),
        (low_idx,    "Low-Risk",     y_prob[low_idx]),
    ]

    for patient_idx, label, prob in patients:
        logger.info(f"Running LIME for {label} patient "
                    f"(idx={patient_idx}, prob={prob:.4f})...")
        instance = X_test.iloc[patient_idx].values

        exp = explainer.explain_instance(
            data_row        = instance,
            predict_fn      = model.predict_proba,
            num_features    = 15,
            num_samples     = 1000
        )

        fig = exp.as_pyplot_figure(label=1)
        fig.set_size_inches(10, 6)
        fig.suptitle(
            f"LIME Explanation — {label} Patient  "
            f"(P(readmit)={prob:.3f})",
            fontsize=TITLE_FS, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        safe_label = label.lower().replace("-", "_")
        path = os.path.join(fig_dir, f"lime_patient_{safe_label}.png")
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close("all")
        logger.info(f"Saved → {path}")


# ── Orchestrator ──────────────────────────────────────────────────────────────
def run_explainability(cfg: dict) -> None:
    model, X_train, X_test, y_test = load_artifacts(
        cfg["paths"]["processed_data"],
        "models/xgb_readmission.json"
    )

    sv_array = run_shap(
        model, X_train, X_test,
        cfg["paths"]["figures"],
        cfg["paths"]["processed_data"]
    )

    run_lime(
        model, X_train, X_test, y_test,
        cfg["paths"]["figures"]
    )

    logger.info("Phase 5 complete — SHAP + LIME explainability done.")


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_explainability(cfg)
    print("\n✅ Phase 5 complete. Check reports/figures/ for all explainability plots.")

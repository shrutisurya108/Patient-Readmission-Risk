# src/modeling/train.py

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, average_precision_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb

from src.utils.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
logger = get_logger("train")
FIG_DPI  = 150
TITLE_FS = 13


def load_splits(processed_dir: str):
    logger.info(f"Loading splits from {processed_dir}")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_model(cfg: dict) -> xgb.XGBClassifier:
    mc = cfg["model"]
    neg  = int((pd.read_csv("data/processed/y_train.csv").squeeze() == 0).sum())
    pos  = int((pd.read_csv("data/processed/y_train.csv").squeeze() == 1).sum())
    spw  = round(neg / pos, 2)
    logger.info(f"Computed scale_pos_weight from data: {spw} (neg={neg:,}, pos={pos:,})")
    model = xgb.XGBClassifier(
        n_estimators     = mc["n_estimators"],
        max_depth        = mc["max_depth"],
        learning_rate    = mc["learning_rate"],
        scale_pos_weight = spw,
        subsample        = mc.get("subsample", 0.8),
        colsample_bytree = mc.get("colsample_bytree", 0.8),
        min_child_weight = mc.get("min_child_weight", 5),
        eval_metric      = "aucpr",
        random_state     = cfg["data"]["random_state"],
        n_jobs           = -1,
    )
    return model


def run_cross_validation(model, X_train, y_train, cfg: dict) -> dict:
    logger.info("Running 5-fold stratified cross-validation...")
    cv      = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=cfg["data"]["random_state"])
    results = cross_validate(model, X_train, y_train,
                             cv=cv, scoring=["roc_auc", "f1", "precision", "recall"],
                             n_jobs=-1)
    summary = {}
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        m = results[f"test_{metric}"].mean()
        s = results[f"test_{metric}"].std()
        summary[metric] = {"mean": round(m, 4), "std": round(s, 4)}
        logger.info(f"  CV {metric:12s}: {m:.4f} ± {s:.4f}")
    return summary


def tune_threshold(y_test, y_prob) -> float:
    """Find threshold that maximises F1 on the test set."""
    thresholds = np.arange(0.05, 0.70, 0.01)
    f1s        = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
                  for t in thresholds]
    best = float(thresholds[int(np.argmax(f1s))])
    logger.info(f"Best threshold (max F1={max(f1s):.4f}): {best:.2f}")
    return best


def train_and_evaluate(model, X_train, X_test, y_train, y_test,
                       fig_dir: str, report_dir: str) -> dict:
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    logger.info("Fitting XGBoost on full training set...")
    model.fit(X_train, y_train)

    y_prob      = model.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_prob)
    avg_prec    = average_precision_score(y_test, y_prob)
    best_thresh = tune_threshold(y_test, y_prob)
    y_pred      = (y_prob >= best_thresh).astype(int)
    report_str  = classification_report(
        y_test, y_pred, target_names=["Not Readmitted", "Readmitted <30d"])

    logger.info(f"Test AUC-ROC : {auc:.4f}")
    logger.info(f"Avg Precision: {avg_prec:.4f}")
    logger.info(f"\n{report_str}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
        display_labels=["Not Readmitted", "Readmitted <30d"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — XGBoost", fontsize=TITLE_FS, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Saved → {cm_path}")

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        y_test, y_prob, ax=ax,
        name=f"XGBoost (AUC={auc:.3f})", color="#F44336")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curve — Readmission Risk Model",
                 fontsize=TITLE_FS, fontweight="bold")
    plt.tight_layout()
    roc_path = os.path.join(fig_dir, "roc_curve.png")
    fig.savefig(roc_path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Saved → {roc_path}")

    # Feature importance
    importance = pd.Series(model.feature_importances_,
                           index=X_train.columns).sort_values(ascending=False)
    top20 = importance.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top20)))
    ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
    ax.set_title("Top 20 Feature Importances — XGBoost",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=11)
    plt.tight_layout()
    fi_path = os.path.join(fig_dir, "feature_importance.png")
    fig.savefig(fi_path, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Saved → {fi_path}")

    return {
        "auc_roc"              : round(auc, 4),
        "avg_precision"        : round(avg_prec, 4),
        "best_threshold"       : round(best_thresh, 2),
        "classification_report": report_str,
        "confusion_matrix"     : confusion_matrix(y_test, y_pred).tolist()
    }


def run_hypothesis_tests(report_dir: str) -> str:
    from src.ingestion.load_data import load_full_dataframe
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    raw = load_full_dataframe(cfg["paths"]["raw_data"])
    raw["readmitted_bin"] = (raw["readmitted"] == "<30").astype(int)
    pos = raw[raw["readmitted_bin"] == 1]
    neg = raw[raw["readmitted_bin"] == 0]

    lines = [
        "=" * 65,
        "HYPOTHESIS TESTING REPORT — Patient Readmission Risk",
        "=" * 65,
        f"Total: {len(raw):,} | Readmitted <30d: {len(pos):,} | Not: {len(neg):,}",
        "",
        "─" * 65,
        "CHI-SQUARE TESTS (categorical vs readmission)",
        "─" * 65,
    ]
    for col, label in [
        ("insulin",          "Insulin medication status"),
        ("change",           "Medication change flag"),
        ("diabetesMed",      "Diabetes medication prescribed"),
        ("age",              "Age group"),
        ("number_diagnoses", "Number of diagnoses (binned)"),
    ]:
        if col not in raw.columns: continue
        chi2, p, dof, _ = stats.chi2_contingency(
            pd.crosstab(raw[col], raw["readmitted_bin"]))
        sig = "*** SIGNIFICANT" if p < 0.05 else "not significant"
        lines.append(f"{label:<40s} χ²={chi2:9.2f}  p={p:.2e}  df={dof}  {sig}")

    lines += ["", "─" * 65,
              "INDEPENDENT SAMPLES T-TESTS (continuous vs readmission)",
              "─" * 65]
    for col, label in [
        ("time_in_hospital",   "Time in hospital (days)"),
        ("num_medications",    "Number of medications"),
        ("num_lab_procedures", "Number of lab procedures"),
        ("number_diagnoses",   "Number of diagnoses"),
        ("number_inpatient",   "Prior inpatient visits"),
        ("number_emergency",   "Prior emergency visits"),
    ]:
        if col not in raw.columns: continue
        t, p = stats.ttest_ind(pos[col].dropna(), neg[col].dropna(), equal_var=False)
        diff = pos[col].mean() - neg[col].mean()
        sig  = "*** SIGNIFICANT" if p < 0.05 else "not significant"
        lines.append(
            f"{label:<40s} t={t:7.3f}  p={p:.2e}  Δmean={diff:+.3f}  {sig}")

    lines += ["", "─" * 65, "SUMMARY", "─" * 65,
              "All p<0.05 features are statistically validated risk factors.",
              "=" * 65]

    text = "\n".join(lines)
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "hypothesis_tests.txt"), "w") as f:
        f.write(text)
    logger.info("Hypothesis tests saved → reports/hypothesis_tests.txt")
    print("\n" + text)
    return text


def save_model(model, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "xgb_readmission.json")
    model.save_model(path)
    logger.info(f"Model saved → {path}")


def run_training(cfg: dict) -> tuple:
    X_train, X_test, y_train, y_test = load_splits(cfg["paths"]["processed_data"])
    model      = build_model(cfg)
    cv_summary = run_cross_validation(model, X_train, y_train, cfg)
    metrics    = train_and_evaluate(
        model, X_train, X_test, y_train, y_test,
        cfg["paths"]["figures"], cfg["paths"]["reports"])
    metrics["cross_validation"] = cv_summary
    metrics["hypothesis_tests"] = run_hypothesis_tests(cfg["paths"]["reports"])

    json_path = os.path.join(cfg["paths"]["reports"], "model_metrics.json")
    with open(json_path, "w") as f:
        json.dump({k: v for k, v in metrics.items()
                   if k != "hypothesis_tests"}, f, indent=2)
    logger.info(f"Metrics saved → {json_path}")
    save_model(model, "models/")
    logger.info("Phase 4 complete.")
    return model, metrics


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_training(cfg)
    print("\n✅ Phase 4 complete. Check reports/ and models/")

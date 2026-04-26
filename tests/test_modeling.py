# tests/test_modeling.py

import os
import json
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_PATH    = "models/xgb_readmission.json"
METRICS_PATH  = "reports/model_metrics.json"
FIGURES_DIR   = "reports/figures"
REQUIRED_FIGS = [
    "confusion_matrix.png",
    "roc_curve.png",
    "feature_importance.png",
]


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model file missing: {MODEL_PATH}"


def test_model_loads_and_predicts():
    model  = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    X_test = pd.read_csv("data/processed/X_test.csv")
    preds  = model.predict(X_test)
    assert len(preds) == len(X_test), "Prediction count mismatch"
    assert set(preds).issubset({0, 1}), "Predictions must be binary"


def test_model_predict_proba():
    model  = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    X_test = pd.read_csv("data/processed/X_test.csv")
    proba  = model.predict_proba(X_test)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0, \
        "Probabilities out of [0,1] range"


def test_auc_above_threshold():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    auc = metrics["auc_roc"]
    assert auc >= 0.65, f"AUC too low for clinical use: {auc}"


def test_cv_auc_above_threshold():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    cv_auc = metrics["cross_validation"]["roc_auc"]["mean"]
    assert cv_auc >= 0.65, f"CV AUC too low: {cv_auc}"


def test_best_threshold_saved():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    assert "best_threshold" in metrics, "best_threshold not saved to metrics"
    t = metrics["best_threshold"]
    assert 0.05 <= t <= 0.70, f"Threshold out of expected range: {t}"


def test_metrics_file_exists():
    assert os.path.exists(METRICS_PATH), f"Metrics file missing: {METRICS_PATH}"


def test_required_figures_exist():
    for fig in REQUIRED_FIGS:
        path = os.path.join(FIGURES_DIR, fig)
        assert os.path.exists(path), f"Missing figure: {fig}"


def test_hypothesis_report_exists():
    path = "reports/hypothesis_tests.txt"
    assert os.path.exists(path), "Hypothesis test report missing"
    with open(path) as f:
        content = f.read()
    assert "SIGNIFICANT" in content, "No significance results in report"
    assert "p=" in content, "p-values not found in report"

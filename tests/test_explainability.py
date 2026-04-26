# tests/test_explainability.py

import os
import numpy as np
import pytest

FIG_DIR        = "reports/figures"
PROCESSED_DIR  = "data/processed"

SHAP_FIGURES = [
    "shap_summary_beeswarm.png",
    "shap_bar_global.png",
    "shap_waterfall_highrisk.png",
    "shap_waterfall_lowrisk.png",
    "shap_dependence_inpatient.png",
]

LIME_FIGURES = [
    "lime_patient_high_risk.png",
    "lime_patient_median_risk.png",
    "lime_patient_low_risk.png",
]


def test_shap_figures_exist():
    for fig in SHAP_FIGURES:
        path = os.path.join(FIG_DIR, fig)
        assert os.path.exists(path), f"Missing SHAP figure: {fig}"


def test_lime_figures_exist():
    for fig in LIME_FIGURES:
        path = os.path.join(FIG_DIR, fig)
        assert os.path.exists(path), f"Missing LIME figure: {fig}"


def test_shap_values_saved():
    path = os.path.join(PROCESSED_DIR, "shap_values.npy")
    assert os.path.exists(path), "shap_values.npy not found"


def test_shap_values_shape():
    import pandas as pd
    sv   = np.load(os.path.join(PROCESSED_DIR, "shap_values.npy"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    assert sv.shape[0] == len(X_test), \
        f"SHAP rows {sv.shape[0]} != X_test rows {len(X_test)}"
    assert sv.shape[1] == X_test.shape[1], \
        f"SHAP cols {sv.shape[1]} != feature cols {X_test.shape[1]}"


def test_shap_figures_nonzero_size():
    for fig in SHAP_FIGURES:
        path = os.path.join(FIG_DIR, fig)
        if os.path.exists(path):
            size = os.path.getsize(path)
            assert size > 10_000, f"{fig} suspiciously small: {size} bytes"


def test_lime_figures_nonzero_size():
    for fig in LIME_FIGURES:
        path = os.path.join(FIG_DIR, fig)
        if os.path.exists(path):
            size = os.path.getsize(path)
            assert size > 10_000, f"{fig} suspiciously small: {size} bytes"

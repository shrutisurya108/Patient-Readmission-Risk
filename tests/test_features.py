# tests/test_features.py

import os
import pytest
import pandas as pd
import numpy as np
import yaml

PROCESSED_DIR = "data/processed"
REQUIRED_FILES = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]


@pytest.fixture(scope="module")
def splits():
    """Load all four processed splits once for the module."""
    data = {}
    for fname in REQUIRED_FILES:
        path = os.path.join(PROCESSED_DIR, fname)
        assert os.path.exists(path), f"Missing file: {path}"
        data[fname.replace(".csv", "")] = pd.read_csv(path)
    return data


def test_processed_files_exist():
    for fname in REQUIRED_FILES:
        assert os.path.exists(os.path.join(PROCESSED_DIR, fname)), \
            f"Missing: {fname}"


def test_no_nulls_in_splits(splits):
    for name, df in splits.items():
        nulls = df.isnull().sum().sum()
        assert nulls == 0, f"{name} has {nulls} null values"


def test_no_object_columns(splits):
    """All columns must be numeric after encoding."""
    for name, df in splits.items():
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        assert obj_cols == [], f"{name} still has object columns: {obj_cols}"


def test_feature_column_alignment(splits):
    """X_train and X_test must have identical columns."""
    assert list(splits["X_train"].columns) == list(splits["X_test"].columns), \
        "Column mismatch between X_train and X_test"


def test_train_class_ratio(splits):
    """Training set must preserve original imbalance (not SMOTE-balanced)."""
    y    = splits["y_train"]["readmitted"]
    pct  = y.sum() / len(y)
    assert 0.08 <= pct <= 0.15, \
        f"Unexpected train class ratio: {pct:.3f} — expected ~0.11"


def test_test_set_untouched(splits):
    """Test set must preserve original imbalance — minority class < 20%."""
    y = splits["y_test"]["readmitted"]
    minority_pct = y.value_counts()[1] / len(y)
    assert minority_pct < 0.20, \
        f"Test set looks resampled — minority class is {minority_pct:.1%}"


def test_row_counts(splits):
    """X and y must have matching row counts for both train and test."""
    assert len(splits["X_train"]) == len(splits["y_train"]), \
        "X_train / y_train row mismatch"
    assert len(splits["X_test"]) == len(splits["y_test"]), \
        "X_test / y_test row mismatch"


def test_engineered_features_present(splits):
    """Custom-engineered features must exist in X_train."""
    required = ["num_meds_changed", "num_meds_active"]
    for feat in required:
        assert feat in splits["X_train"].columns, \
            f"Engineered feature missing: {feat}"


def test_target_binary(splits):
    """Target column must contain only 0 and 1."""
    for name in ["y_train", "y_test"]:
        vals = set(splits[name]["readmitted"].unique())
        assert vals.issubset({0, 1}), \
            f"{name} target has unexpected values: {vals}"

# tests/test_ingestion.py

import os
import pandas as pd
import pytest
from src.ingestion.load_data import download_dataset, build_sqlite, load_full_dataframe

RAW_DIR = "data/raw"

def test_raw_files_exist():
    """After download, both CSVs must exist."""
    assert os.path.exists(os.path.join(RAW_DIR, "features.csv")), "features.csv missing"
    assert os.path.exists(os.path.join(RAW_DIR, "targets.csv")),  "targets.csv missing"

def test_features_shape():
    """Dataset must have 100k+ rows and expected key columns."""
    X = pd.read_csv(os.path.join(RAW_DIR, "features.csv"))
    assert X.shape[0] > 100_000, f"Expected 100k+ rows, got {X.shape[0]}"
    required = ["age", "time_in_hospital", "num_medications",
                "number_diagnoses", "insulin", "number_inpatient"]
    for col in required:
        assert col in X.columns, f"Missing column: {col}"

def test_targets_shape():
    """Targets must have same row count as features and a readmitted column."""
    X = pd.read_csv(os.path.join(RAW_DIR, "features.csv"))
    y = pd.read_csv(os.path.join(RAW_DIR, "targets.csv"))
    assert y.shape[0] == X.shape[0], "Row count mismatch between features and targets"
    assert "readmitted" in y.columns, "Missing 'readmitted' column in targets"

def test_target_classes():
    """readmitted column must contain exactly the 3 expected class values."""
    y = pd.read_csv(os.path.join(RAW_DIR, "targets.csv"))
    classes = set(y["readmitted"].unique())
    assert classes == {"<30", ">30", "NO"}, f"Unexpected classes: {classes}"

def test_sqlite_load():
    """SQLite must load without error and return correct row count."""
    X = pd.read_csv(os.path.join(RAW_DIR, "features.csv"))
    y = pd.read_csv(os.path.join(RAW_DIR, "targets.csv"))
    conn = build_sqlite(X, y)
    result = pd.read_sql("SELECT COUNT(*) as n FROM patients", conn)
    assert result.iloc[0, 0] == X.shape[0]

def test_load_full_dataframe():
    """Merged DataFrame must have both feature cols and readmitted col."""
    df = load_full_dataframe(RAW_DIR)
    assert "readmitted" in df.columns
    assert df.shape[0] > 100_000

# src/features/engineer.py

import os
import warnings
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*DtypeWarning.*")

logger = get_logger("engineer")

# ── ICD-9 bucketing ───────────────────────────────────────────────────────────
def _icd9_to_category(code) -> str:
    if pd.isnull(code) or str(code).strip() in ("?", ""):
        return "Other"
    code = str(code).strip()
    if code.startswith("E") or code.startswith("V"):
        return "External"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 390 <= num <= 459 or num == 785: return "Circulatory"
    if 460 <= num <= 519 or num == 786: return "Respiratory"
    if 520 <= num <= 579 or num == 787: return "Digestive"
    if 250 <= num <= 250.99:            return "Diabetes"
    if 800 <= num <= 999:               return "Injury"
    if 710 <= num <= 739:               return "Musculoskeletal"
    if 580 <= num <= 629 or num == 788: return "Genitourinary"
    if 140 <= num <= 239:               return "Neoplasms"
    return "Other"

AGE_MAP = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
}

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone"
]

DROP_COLS        = ["encounter_id", "patient_nbr", "examide", "citoglipton"]
HIGH_MISS_COLS   = ["weight", "payer_code", "medical_specialty"]


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering pipeline...")
    df = df.copy()

    # 1 — Replace "?" with NaN, drop useless columns
    df.replace("?", np.nan, inplace=True)
    drop = [c for c in HIGH_MISS_COLS + DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True)
    logger.info(f"Dropped {len(drop)} columns: {drop}")

    # 2 — Binary target: 1 = readmitted <30d
    df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    pos = df["readmitted"].sum()
    logger.info(f"Target: positive={pos:,} | negative={len(df)-pos:,} | ratio=1:{(len(df)-pos)//pos}")

    # 3 — Remove Invalid gender (3 rows)
    df = df[df["gender"] != "Unknown/Invalid"].copy()

    # 4 — ICD-9 bucketing
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].apply(_icd9_to_category)
    logger.info("ICD-9 codes bucketed into clinical categories")

    # 5 — Age ordinal
    if "age" in df.columns:
        df["age"] = df["age"].map(AGE_MAP).fillna(4).astype(int)

    # 6 — Medication engineered features (BEFORE encoding med columns)
    med_present = [c for c in MED_COLS if c in df.columns]
    df["num_meds_changed"] = df[med_present].apply(
        lambda r: r.isin(["Up", "Down"]).sum(), axis=1)
    df["num_meds_active"]  = df[med_present].apply(
        lambda r: (r != "No").sum(), axis=1)
    logger.info("Engineered: num_meds_changed, num_meds_active")

    # 7 — Encode medication columns
    med_order = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    for col in med_present:
        if col in df.columns:
            df[col] = df[col].map(med_order).fillna(0).astype(int)

    # 8 — Label encode remaining categoricals
    cat_cols = [c for c in df.select_dtypes("object").columns if c != "readmitted"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    logger.info(f"Label encoded: {cat_cols}")

    # 9 — Median imputation for any remaining nulls
    for col in df.columns[df.isnull().any()]:
        df[col].fillna(df[col].median(), inplace=True)

    logger.info(f"Final shape: {df.shape} | Features: {df.shape[1]-1}")
    return df


def split_and_smote(df: pd.DataFrame, cfg: dict) -> tuple:
    """
    Split only — no SMOTE.
    Class imbalance is handled inside XGBoost via scale_pos_weight.
    This avoids synthetic interpolation artifacts on categorical/ordinal features.
    """
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = cfg["data"]["test_size"],
        random_state = cfg["data"]["random_state"],
        stratify     = y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train pos: {y_train.sum():,} | neg: {(y_train==0).sum():,}")
    logger.info(f"Test  pos: {y_test.sum():,}  | neg: {(y_test==0).sum():,}")
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv( os.path.join(out_dir, "X_test.csv"),  index=False)
    pd.Series(y_train, name="readmitted").to_csv(
        os.path.join(out_dir, "y_train.csv"), index=False)
    pd.Series(y_test,  name="readmitted").to_csv(
        os.path.join(out_dir, "y_test.csv"),  index=False)
    logger.info(f"Saved splits → {out_dir}")


def run_feature_engineering(cfg: dict) -> tuple:
    from src.ingestion.load_data import load_full_dataframe
    df = load_full_dataframe(cfg["paths"]["raw_data"])
    df = clean_and_encode(df)
    X_train, X_test, y_train, y_test = split_and_smote(df, cfg)
    save_splits(X_train, X_test, y_train, y_test, cfg["paths"]["processed_data"])
    logger.info("Phase 3 complete.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_feature_engineering(cfg)
    print("\n✅ Phase 3 complete. Check data/processed/ and logs/engineer.log")

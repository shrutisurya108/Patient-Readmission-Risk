# src/ingestion/load_data.py

import os
import sqlite3
import pandas as pd
from ucimlrepo import fetch_ucirepo
from src.utils.logger import get_logger

logger = get_logger("load_data")


def download_dataset(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download UCI Diabetes Readmission dataset and save to raw/."""
    os.makedirs(raw_dir, exist_ok=True)
    features_path = os.path.join(raw_dir, "features.csv")
    targets_path  = os.path.join(raw_dir, "targets.csv")

    if os.path.exists(features_path) and os.path.exists(targets_path):
        logger.info("Raw data already exists — loading from disk.")
        X = pd.read_csv(features_path)
        y = pd.read_csv(targets_path)
        return X, y

    logger.info("Fetching dataset from UCI ML Repository (ID=296)...")
    dataset = fetch_ucirepo(id=296)
    X = dataset.data.features
    y = dataset.data.targets

    X.to_csv(features_path, index=False)
    y.to_csv(targets_path,  index=False)
    logger.info(f"Saved features → {features_path}  shape={X.shape}")
    logger.info(f"Saved targets  → {targets_path}   shape={y.shape}")
    return X, y


def build_sqlite(X: pd.DataFrame, y: pd.DataFrame) -> sqlite3.Connection:
    """Load features + targets into an in-memory SQLite database."""
    logger.info("Building in-memory SQLite database...")
    conn = sqlite3.connect(":memory:")
    df = X.copy()
    df["readmitted"] = y["readmitted"].values
    df.to_sql("patients", conn, if_exists="replace", index=False)
    count = pd.read_sql("SELECT COUNT(*) as n FROM patients", conn).iloc[0, 0]
    logger.info(f"SQLite loaded — {count:,} rows in 'patients' table.")
    return conn


def run_profiling_queries(conn: sqlite3.Connection, report_dir: str) -> None:
    """Run SQL cohort profiling queries and save results."""
    os.makedirs(report_dir, exist_ok=True)
    out_path = os.path.join(report_dir, "sql_profiling.txt")
    lines = []

    def section(title: str, sql: str) -> None:
        logger.info(f"Running query: {title}")
        df = pd.read_sql(sql, conn)
        block = f"\n{'='*60}\n{title}\n{'='*60}\n{df.to_string(index=False)}\n"
        lines.append(block)
        print(block)

    # 1 — Overall readmission distribution
    section("1. Overall Readmission Distribution", """
        SELECT readmitted,
               COUNT(*) as patient_count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
        FROM patients
        GROUP BY readmitted
        ORDER BY patient_count DESC
    """)

    # 2 — Readmission rate by age group
    section("2. Readmission Rate by Age Group", """
        SELECT age,
               COUNT(*) as total,
               SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as readmitted_30,
               ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as readmit_rate_pct
        FROM patients
        GROUP BY age
        ORDER BY age
    """)

    # 3 — Readmission rate by number of inpatient visits
    section("3. Readmission Rate by Prior Inpatient Visits", """
        SELECT number_inpatient,
               COUNT(*) as total,
               SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as readmitted_30,
               ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as readmit_rate_pct
        FROM patients
        GROUP BY number_inpatient
        ORDER BY number_inpatient
    """)

    # 4 — Readmission rate by insulin status
    section("4. Readmission Rate by Insulin Status", """
        SELECT insulin,
               COUNT(*) as total,
               SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as readmitted_30,
               ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as readmit_rate_pct
        FROM patients
        GROUP BY insulin
        ORDER BY readmit_rate_pct DESC
    """)

    # 5 — Readmission rate by number of diagnoses
    section("5. Readmission Rate by Number of Diagnoses", """
        SELECT number_diagnoses,
               COUNT(*) as total,
               SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as readmitted_30,
               ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as readmit_rate_pct
        FROM patients
        GROUP BY number_diagnoses
        ORDER BY number_diagnoses
    """)

    # 6 — Readmission rate by time in hospital
    section("6. Readmission Rate by Time in Hospital (days)", """
        SELECT time_in_hospital,
               COUNT(*) as total,
               SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as readmitted_30,
               ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as readmit_rate_pct
        FROM patients
        GROUP BY time_in_hospital
        ORDER BY time_in_hospital
    """)

    # 7 — Readmission rate by discharge disposition (top 10)
    section("7. Top 10 Discharge Dispositions by Readmission Rate", """
        SELECT discharge_disposition_id,
               COUNT(*) as total,
               SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as readmitted_30,
               ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as readmit_rate_pct
        FROM patients
        GROUP BY discharge_disposition_id
        HAVING total > 100
        ORDER BY readmit_rate_pct DESC
        LIMIT 10
    """)

    # 8 — Readmission rate by race
    section("8. Readmission Rate by Race", """
        SELECT race,
               COUNT(*) as total,
               SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as readmitted_30,
               ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as readmit_rate_pct
        FROM patients
        GROUP BY race
        ORDER BY readmit_rate_pct DESC
    """)

    # 9 — Avg medications and lab procedures for readmitted vs not
    section("9. Avg Clinical Metrics — Readmitted <30 vs Not Readmitted", """
        SELECT
            CASE WHEN readmitted = '<30' THEN 'Readmitted <30' ELSE 'Not Readmitted' END as cohort,
            ROUND(AVG(num_medications), 2)     as avg_medications,
            ROUND(AVG(num_lab_procedures), 2)  as avg_lab_procedures,
            ROUND(AVG(time_in_hospital), 2)    as avg_time_in_hospital,
            ROUND(AVG(number_diagnoses), 2)    as avg_diagnoses,
            COUNT(*) as n
        FROM patients
        GROUP BY cohort
    """)

    # 10 — Gender breakdown
    section("10. Readmission Rate by Gender", """
        SELECT gender,
               COUNT(*) as total,
               SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) as readmitted_30,
               ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as readmit_rate_pct
        FROM patients
        GROUP BY gender
        ORDER BY readmit_rate_pct DESC
    """)

    with open(out_path, "w") as f:
        f.writelines(lines)
    logger.info(f"SQL profiling report saved → {out_path}")


def load_full_dataframe(raw_dir: str) -> pd.DataFrame:
    """Return merged features+targets as a single DataFrame (used by downstream modules)."""
    X = pd.read_csv(os.path.join(raw_dir, "features.csv"))
    y = pd.read_csv(os.path.join(raw_dir, "targets.csv"))
    df = X.copy()
    df["readmitted"] = y["readmitted"].values
    return df


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    X, y = download_dataset(cfg["paths"]["raw_data"])
    conn  = build_sqlite(X, y)
    run_profiling_queries(conn, cfg["paths"]["reports"])

    logger.info("Phase 1 complete — data acquired and profiled.")
    print("\n✅ Phase 1 complete. Check logs/load_data.log and reports/sql_profiling.txt")

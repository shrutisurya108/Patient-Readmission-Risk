# main.py

import sys
import time
import yaml

from src.utils.logger import get_logger

logger = get_logger("main")


def run_pipeline(cfg: dict) -> None:
    start = time.time()
    logger.info("=" * 60)
    logger.info("PATIENT READMISSION RISK — FULL PIPELINE")
    logger.info("=" * 60)

    # ── Phase 1: Data ingestion & SQL profiling ───────────────────
    logger.info("[1/5] Data acquisition & SQL profiling...")
    from src.ingestion.load_data import (
        download_dataset, build_sqlite, run_profiling_queries
    )
    X, y = download_dataset(cfg["paths"]["raw_data"])
    conn = build_sqlite(X, y)
    run_profiling_queries(conn, cfg["paths"]["reports"])
    logger.info("[1/5] ✓ Done\n")

    # ── Phase 2: EDA ──────────────────────────────────────────────
    logger.info("[2/5] Exploratory data analysis...")
    from src.ingestion.eda import run_eda
    from src.ingestion.load_data import load_full_dataframe
    df = load_full_dataframe(cfg["paths"]["raw_data"])
    run_eda(df, cfg["paths"]["figures"])
    logger.info("[2/5] ✓ Done\n")

    # ── Phase 3: Feature engineering ─────────────────────────────
    logger.info("[3/5] Feature engineering...")
    from src.features.engineer import run_feature_engineering
    run_feature_engineering(cfg)
    logger.info("[3/5] ✓ Done\n")

    # ── Phase 4: Model training & hypothesis testing ──────────────
    logger.info("[4/5] Model training & hypothesis testing...")
    from src.modeling.train import run_training
    run_training(cfg)
    logger.info("[4/5] ✓ Done\n")

    # ── Phase 5: Explainability ───────────────────────────────────
    logger.info("[5/5] SHAP + LIME explainability...")
    from src.explainability.explain import run_explainability
    run_explainability(cfg)
    logger.info("[5/5] ✓ Done\n")

    # ── Phase 6: Clinical PDF report ──────────────────────────────
    logger.info("[6/6] Generating clinical PDF report...")
    from src.reporting.report import build_report
    path = build_report(cfg)
    logger.info(f"[6/6] ✓ Report saved → {path}\n")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE — Total time: {elapsed:.1f}s")
    logger.info("=" * 60)
    print(f"\n✅ Full pipeline complete in {elapsed:.1f}s")
    print(f"   Clinical report → {path}")


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    try:
        run_pipeline(cfg)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

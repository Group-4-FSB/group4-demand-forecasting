"""End-to-end training pipeline entrypoint:
ingest -> validate -> feature engineering -> train (+ tune, CV, MLflow log)
-> save reference snapshot for the serving API.

Usage:
    python scripts/run_pipeline.py [--n-param-samples 6] [--n-cv-splits 3]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecast.config import settings  # noqa: E402
from demand_forecast.data.features import build_features  # noqa: E402
from demand_forecast.data.ingest import load_raw_tables, merge_dataset  # noqa: E402
from demand_forecast.data.snapshot import save_reference_artifacts  # noqa: E402
from demand_forecast.data.validate import validate_sales_table  # noqa: E402
from demand_forecast.models.train import train_and_log  # noqa: E402
from demand_forecast.reporting import generate_responsible_ai_report  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-param-samples", type=int, default=6)
    parser.add_argument("--n-cv-splits", type=int, default=settings.n_cv_splits)
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow model registry")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("1/4 Loading raw tables from %s", settings.data_raw_dir)
    tables = load_raw_tables(settings.data_raw_dir)

    logger.info("2/4 Merging + validating training data")
    merged = merge_dataset(tables, split="train")
    report = validate_sales_table(merged, tables["stores"])
    for w in report.warnings:
        logger.warning("Data quality warning: %s", w)
    report.raise_if_failed()

    logger.info("3/4 Feature engineering")
    features_df = build_features(merged)

    logger.info("4/4 Training + hyperparameter search + MLflow logging")
    summary = train_and_log(
        features_df,
        n_cv_splits=args.n_cv_splits,
        n_param_samples=args.n_param_samples,
        register_model=not args.no_register,
    )

    save_reference_artifacts(
        features_df, tables["stores"], tables["holidays"], settings.data_processed_dir
    )
    logger.info("Saved reference snapshot to %s", settings.data_processed_dir)

    logger.info("5/5 Generating Responsible AI report (fairness + explainability)")
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    generate_responsible_ai_report(
        model=summary["model"],
        df=features_df,
        feature_cols=summary["feature_columns"],
        output_dir=reports_dir,
        mlflow_run_id=summary["run_id"],
    )
    logger.info("Saved Responsible AI report to %s", reports_dir)

    print("=" * 60)
    print(f"Baseline RMSLE      : {summary['baseline_metrics']['rmsle']:.4f}")
    print(f"Best LightGBM RMSLE : {summary['best_cv_rmsle']:.4f}")
    print(f"Best params         : {summary['best_params']}")
    print(f"MLflow run id       : {summary['run_id']}")
    print(f"MLflow model URI    : {summary['model_uri']}")
    print(f"Reports directory   : {reports_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

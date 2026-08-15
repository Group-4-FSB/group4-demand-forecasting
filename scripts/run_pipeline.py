"""End-to-end training pipeline entrypoint:
ingest -> validate -> feature engineering -> train (+ tune, CV, held-out test,
quality gate, MLflow log) -> save reference snapshot for the serving API.

The most recent `--test-weeks` weeks are held out and never used for training
or hyperparameter selection — see `models.train.train_and_log` for the full
three-stage split (pool for CV -> honest test eval -> refit on everything).
The resulting candidate is only promoted to the `production` alias if it is
at least as good, on that held-out test set, as whatever is already
production — see the "quality gate" section of `train_and_log`.

`run_training_pipeline()` is the reusable entrypoint (also used by
`scripts/retrain_if_stale.py` for scheduled/automatic retraining); `main()`
below is just its CLI wrapper.

Usage:
    python scripts/run_pipeline.py [--n-param-samples 6] [--n-cv-splits 3] [--test-weeks 8]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecast.config import settings  # noqa: E402
from demand_forecast.data.features import build_features  # noqa: E402
from demand_forecast.data.ingest import (  # noqa: E402
    compute_raw_data_fingerprint,
    load_walmart_sales,
)
from demand_forecast.data.snapshot import save_reference_artifacts  # noqa: E402
from demand_forecast.data.validate import validate_sales_table  # noqa: E402
from demand_forecast.models.train import train_and_log  # noqa: E402
from demand_forecast.reporting import generate_responsible_ai_report  # noqa: E402

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def run_training_pipeline(
    n_param_samples: int = 6,
    n_cv_splits: int = settings.n_cv_splits,
    test_weeks: int = settings.test_holdout_weeks,
    register_model: bool = True,
) -> dict[str, Any]:
    """Run the full pipeline once: ingest -> validate -> features -> train
    (CV + held-out test + quality gate) -> reference snapshot -> Responsible
    AI report (only if the new model was actually promoted).

    Returns `train_and_log()`'s summary dict, with `reports_dir` added.
    """
    logger.info("1/5 Loading raw data from %s", settings.data_raw_dir)
    df = load_walmart_sales(settings.data_raw_dir)
    data_fingerprint = compute_raw_data_fingerprint(settings.data_raw_dir, df)
    logger.info(
        "Raw data fingerprint: sha256=%s..., %d rows, %s to %s",
        data_fingerprint["raw_data_sha256"][:12],
        data_fingerprint["raw_data_rows"],
        data_fingerprint["raw_data_min_date"],
        data_fingerprint["raw_data_max_date"],
    )

    logger.info("2/5 Validating training data")
    report = validate_sales_table(df)
    for w in report.warnings:
        logger.warning("Data quality warning: %s", w)
    report.raise_if_failed()

    logger.info("3/5 Feature engineering")
    features_df = build_features(df)

    logger.info("4/5 Training + hyperparameter search + quality gate + MLflow logging")
    summary = train_and_log(
        features_df,
        n_cv_splits=n_cv_splits,
        n_param_samples=n_param_samples,
        register_model=register_model,
        test_weeks=test_weeks,
        data_fingerprint=data_fingerprint,
    )

    if summary["promoted"]:
        save_reference_artifacts(features_df, settings.data_processed_dir)
        logger.info("Saved reference snapshot to %s", settings.data_processed_dir)

        logger.info("5/5 Generating Responsible AI report (fairness + explainability)")
        generate_responsible_ai_report(
            model=summary["model"],
            df=features_df,
            feature_cols=summary["feature_columns"],
            output_dir=REPORTS_DIR,
            fairness_evaluation_df=summary["responsible_ai_eval_df"],
            fairness_reference_df=summary["responsible_ai_reference_df"],
            fairness_evaluation_model=summary["responsible_ai_eval_model"],
            mlflow_run_id=summary["run_id"],
        )
        logger.info("Saved Responsible AI report to %s", REPORTS_DIR)
        summary["reports_dir"] = REPORTS_DIR
    else:
        logger.info(
            "5/5 Skipped reference snapshot + Responsible AI report — new model "
            "was not promoted, so reports/ and data/processed/ still reflect the "
            "current production model."
        )
        summary["reports_dir"] = None

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-param-samples", type=int, default=6)
    parser.add_argument("--n-cv-splits", type=int, default=settings.n_cv_splits)
    parser.add_argument("--test-weeks", type=int, default=settings.test_holdout_weeks)
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow model registry")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")

    try:
        summary = run_training_pipeline(
            n_param_samples=args.n_param_samples,
            n_cv_splits=args.n_cv_splits,
            test_weeks=args.test_weeks,
            register_model=not args.no_register,
        )
    except Exception:
        logger.exception(
            "Training pipeline failed — production model (if any) is unchanged; "
            "see the traceback above for the root cause."
        )
        return 1

    print("=" * 60)
    print(f"Baseline RMSLE (all data)   : {summary['baseline_metrics']['rmsle']:.4f}")
    print(f"CV RMSLE (train+CV pool)    : {summary['best_cv_rmsle']:.4f}")
    print(f"Baseline RMSLE (test weeks) : {summary['baseline_test_metrics']['rmsle']:.4f}")
    print(
        f"Held-out TEST RMSLE         : {summary['test_metrics']['rmsle']:.4f}  "
        f"({summary['test_weeks']} weeks, never trained on)"
    )
    if summary["current_production_test_rmsle"] is not None:
        print(f"Current production RMSLE    : {summary['current_production_test_rmsle']:.4f}")
    print(f"Best params                 : {summary['best_params']}")
    if summary["promoted"]:
        print("Result                       : PROMOTED to production")
        print(f"MLflow run id                : {summary['run_id']}")
        print(f"MLflow model URI             : {summary['model_uri']}")
        print(f"Reports directory            : {summary['reports_dir']}")
    else:
        print("Result                       : REJECTED by quality gate — production unchanged")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

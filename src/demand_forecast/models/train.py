"""Model training: seasonal-naive baseline + LightGBM with time-series CV,
a small hyperparameter search, and full MLflow experiment tracking.

Run directly (`python -m demand_forecast.models.train` or via
`scripts/run_pipeline.py`) to train on the full dataset in data/raw/, or import
`train_and_log()` to run the same logic against any prepared DataFrame (e.g. a
small fixture in tests).
"""

from __future__ import annotations

import itertools
import logging
import random
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from demand_forecast.config import settings
from demand_forecast.data.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    LOG_TARGET,
    TARGET,
    build_features,
)
from demand_forecast.data.ingest import load_raw_tables, merge_dataset
from demand_forecast.data.validate import validate_sales_table
from demand_forecast.models.evaluate import evaluate_all

logger = logging.getLogger(__name__)

DEFAULT_PARAM_GRID = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
    "min_child_samples": [20, 50],
}


def _ensure_experiment(name: str, artifact_root: str) -> None:
    """Set the active MLflow experiment, creating it with an explicit artifact
    location if it doesn't exist yet (avoids artifacts landing in a stray
    ./mlruns relative to whatever the current working directory happens to be)."""
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        Path(artifact_root).mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=Path(artifact_root).as_uri())
    mlflow.set_experiment(name)


def prepare_training_frame(raw_dir: Path) -> pd.DataFrame:
    """Load, validate, and feature-engineer the full training table."""
    tables = load_raw_tables(raw_dir)
    merged = merge_dataset(tables, split="train")
    report = validate_sales_table(merged, tables["stores"])
    if not report.ok:
        raise ValueError(f"Data validation failed: {report.errors}")
    for w in report.warnings:
        logger.warning("Data quality warning: %s", w)
    return build_features(merged)


def time_series_cv_splits(dates: pd.Series, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rolling-origin CV: split on unique sorted dates, map back to row indices
    so every fold's validation window is strictly after its training window."""
    unique_dates = np.sort(dates.unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_date_idx, valid_date_idx in tscv.split(unique_dates):
        train_dates = set(unique_dates[train_date_idx])
        valid_dates = set(unique_dates[valid_date_idx])
        train_idx = np.flatnonzero(dates.isin(train_dates).to_numpy())
        valid_idx = np.flatnonzero(dates.isin(valid_dates).to_numpy())
        splits.append((train_idx, valid_idx))
    return splits


def seasonal_naive_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Baseline: predict this week's sales as last week's same-weekday sales
    (already computed as the `sales_lag_7` feature). No training required."""
    mask = df["sales_lag_7"].notna()
    return evaluate_all(df.loc[mask, TARGET], df.loc[mask, "sales_lag_7"])


def sample_param_grid(
    grid: dict[str, list[Any]], n_samples: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(grid)
    all_combos = list(itertools.product(*grid.values()))
    rng.shuffle(all_combos)
    chosen = all_combos[:n_samples]
    return [dict(zip(keys, combo)) for combo in chosen]


def cv_score_params(
    df: pd.DataFrame, params: dict[str, Any], splits: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[float, list[dict[str, float]]]:
    fold_metrics = []
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    for train_idx, valid_idx in splits:
        train_df, valid_df = df.iloc[train_idx], df.iloc[valid_idx]
        model = lgb.LGBMRegressor(
            objective="regression",
            random_state=settings.random_seed,
            verbosity=-1,
            **params,
        )
        model.fit(
            train_df[feature_cols],
            train_df[LOG_TARGET],
            categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feature_cols],
        )
        preds = np.expm1(model.predict(valid_df[feature_cols]))
        fold_metrics.append(evaluate_all(valid_df[TARGET], preds))
    mean_rmsle = float(np.mean([m["rmsle"] for m in fold_metrics]))
    return mean_rmsle, fold_metrics


def train_and_log(
    df: pd.DataFrame,
    n_cv_splits: int = settings.n_cv_splits,
    n_param_samples: int = 6,
    param_grid: dict[str, list[Any]] | None = None,
    register_model: bool = True,
) -> dict[str, Any]:
    """Run baseline + tuned LightGBM training, logging everything to MLflow.

    Returns a summary dict with baseline metrics, best params, best CV metrics,
    and the MLflow run id of the final (best) model.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    _ensure_experiment(settings.mlflow_experiment_name, settings.mlflow_artifact_root)

    splits = time_series_cv_splits(df["date"], n_cv_splits)
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    # 1. Baseline run — logged for comparison ("multiple experiments").
    logger.info("Computing seasonal-naive baseline on %d rows...", len(df))
    baseline_metrics = seasonal_naive_baseline(df)
    logger.info("Baseline RMSLE: %.4f", baseline_metrics["rmsle"])
    with mlflow.start_run(run_name="baseline_seasonal_naive"):
        mlflow.set_tag("model_type", "baseline")
        mlflow.log_metrics({f"holdout_{k}": v for k, v in baseline_metrics.items()})

    # 2. Small random hyperparameter search over LightGBM, each combo its own run.
    grid = param_grid or DEFAULT_PARAM_GRID
    candidates = sample_param_grid(grid, n_param_samples, seed=settings.random_seed)
    logger.info(
        "Starting hyperparameter search: %d candidates x %d CV folds = %d LightGBM fits "
        "on %d rows each (this is the slow part — no MLflow output happens between here "
        "and the next log line per candidate, that's expected, not a hang).",
        len(candidates),
        n_cv_splits,
        len(candidates) * n_cv_splits,
        len(df),
    )

    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    best_fold_metrics: list[dict[str, float]] = []
    for i, params in enumerate(candidates, start=1):
        logger.info("[%d/%d] Fitting LightGBM with params=%s ...", i, len(candidates), params)
        with mlflow.start_run(run_name=f"lgbm_{'_'.join(str(v) for v in params.values())}"):
            mlflow.set_tag("model_type", "lightgbm")
            mlflow.log_params(params)
            mean_rmsle, fold_metrics = cv_score_params(df, params, splits)
            for j, fm in enumerate(fold_metrics):
                mlflow.log_metrics({f"fold{j}_{k}": v for k, v in fm.items()})
            mlflow.log_metric("cv_mean_rmsle", mean_rmsle)
        logger.info("[%d/%d] cv_mean_rmsle=%.4f", i, len(candidates), mean_rmsle)
        if mean_rmsle < best_score:
            best_score = mean_rmsle
            best_params = params
            best_fold_metrics = fold_metrics

    assert best_params is not None
    logger.info("Best params so far: %s (cv_mean_rmsle=%.4f)", best_params, best_score)

    # 3. Retrain best params on the full dataset and register the final model.
    logger.info("Retraining best model on the full dataset and logging to MLflow...")
    final_model = lgb.LGBMRegressor(
        objective="regression",
        random_state=settings.random_seed,
        verbosity=-1,
        **best_params,
    )
    final_model.fit(
        df[feature_cols],
        df[LOG_TARGET],
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in feature_cols],
    )
    logger.info("Final model trained. Logging model + artifacts to MLflow...")

    with mlflow.start_run(run_name="best_lightgbm_final") as run:
        mlflow.set_tag("model_type", "lightgbm_final")
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_mean_rmsle", best_score)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        mlflow.log_dict({"feature_columns": feature_cols}, "feature_columns.json")

        importances = pd.Series(
            final_model.booster_.feature_importance(importance_type="gain"), index=feature_cols
        ).sort_values(ascending=False)
        mlflow.log_dict(importances.to_dict(), "feature_importance_gain.json")

        model_info = mlflow.lightgbm.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=settings.mlflow_model_name if register_model else None,
        )
        final_run_id = run.info.run_id

    if register_model:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"run_id='{final_run_id}'")
        if versions:
            client.set_registered_model_alias(
                name=settings.mlflow_model_name,
                alias=settings.mlflow_model_alias,
                version=versions[0].version,
            )
            logger.info(
                "Registered %s v%s and aliased '%s'.",
                settings.mlflow_model_name,
                versions[0].version,
                settings.mlflow_model_alias,
            )

    return {
        "baseline_metrics": baseline_metrics,
        "best_params": best_params,
        "best_cv_rmsle": best_score,
        "best_fold_metrics": best_fold_metrics,
        "run_id": final_run_id,
        "model_uri": model_info.model_uri,
        "model": final_model,
        "feature_columns": feature_cols,
    }


def main() -> None:
    logging.basicConfig(level=settings.log_level)
    df = prepare_training_frame(settings.data_raw_dir)
    summary = train_and_log(df)
    logger.info("Training complete: %s", summary)
    print(
        f"Baseline RMSLE={summary['baseline_metrics']['rmsle']:.4f} | "
        f"Best LightGBM CV RMSLE={summary['best_cv_rmsle']:.4f} "
        f"(params={summary['best_params']})"
    )


if __name__ == "__main__":
    main()

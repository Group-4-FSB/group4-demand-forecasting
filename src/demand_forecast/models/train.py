"""Model training: naive (previous-week) baseline + LightGBM with
time-series CV, a small hyperparameter search, and full MLflow experiment
tracking.

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
from demand_forecast.data.ingest import load_walmart_sales
from demand_forecast.data.validate import validate_sales_table
from demand_forecast.models.evaluate import evaluate_all

logger = logging.getLogger(__name__)

# Small-data-friendly grid: the full dataset is only ~6.4K rows (45 stores x
# 143 weeks), a fraction of what the original daily-grain dataset had, so
# num_leaves/min_child_samples are kept low to reduce overfitting risk.
DEFAULT_PARAM_GRID = {
    "num_leaves": [7, 15, 31],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [100, 200],
    "min_child_samples": [5, 10, 20],
}


def _is_remote_tracking_uri(tracking_uri: str) -> bool:
    """True for an MLflow tracking *server* (http/https — e.g. the Dockerized
    mlflow service), False for a local backend (sqlite:/// etc.)."""
    return tracking_uri.startswith("http://") or tracking_uri.startswith("https://")


def _ensure_experiment(name: str, artifact_root: str) -> None:
    """Set the active MLflow experiment, creating it if it doesn't exist yet.

    Local tracking (sqlite:///...): pass an explicit local artifact_location
    so artifacts don't land in a stray ./mlruns relative to whatever the
    current working directory happens to be.

    Remote tracking (http(s)://... — e.g. the Dockerized mlflow service):
    do NOT force a local filesystem path as artifact_location. The training
    client (running on the host) can write to a local Windows/Mac/Linux path
    just fine, but any *other* client reading the model later — e.g. the API
    container, which only has the network, not that filesystem — can't. Let
    the server pick its own artifact storage (mlflow-artifacts:/, proxied
    over the same HTTP connection every client already uses) instead.
    """
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        if _is_remote_tracking_uri(mlflow.get_tracking_uri()):
            mlflow.create_experiment(name)
        else:
            Path(artifact_root).mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(name, artifact_location=Path(artifact_root).as_uri())
    mlflow.set_experiment(name)


def prepare_training_frame(raw_dir: Path) -> pd.DataFrame:
    """Load, validate, and feature-engineer the full training table."""
    df = load_walmart_sales(raw_dir)
    report = validate_sales_table(df)
    if not report.ok:
        raise ValueError(f"Data validation failed: {report.errors}")
    for w in report.warnings:
        logger.warning("Data quality warning: %s", w)
    return build_features(df)


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


def chronological_holdout_split(
    df: pd.DataFrame, test_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date into (pool, test): `pool` is every week except the most
    recent `test_weeks`, `test` is those held-out weeks.

    `pool` is the only data allowed into training and hyperparameter-search
    CV. `test` is never trained on or used to pick hyperparameters — it
    exists purely for a one-time, honest accuracy check (see `train_and_log`)
    before the final production model is refit on 100% of the data. Splitting
    happens *after* feature engineering, so `test` rows keep correct
    backward-looking lag/rolling values computed from real history — nothing
    here can leak future information into the past.
    """
    unique_dates = np.sort(df["date"].unique())
    if test_weeks <= 0 or test_weeks >= len(unique_dates):
        raise ValueError(
            f"test_weeks={test_weeks} must be between 1 and {len(unique_dates) - 1} "
            f"(dataset has {len(unique_dates)} unique weeks)"
        )
    cutoff_date = unique_dates[-test_weeks]
    pool = df.loc[df["date"] < cutoff_date].copy()
    test = df.loc[df["date"] >= cutoff_date].copy()
    return pool, test


def naive_baseline(df: pd.DataFrame) -> dict[str, float]:
    """Baseline: predict this week's sales as last week's sales for the same
    store (already computed as the `sales_lag_1` feature). No training
    required — data is weekly, so lag_1 is the simplest persistence baseline."""
    mask = df["sales_lag_1"].notna()
    return evaluate_all(df.loc[mask, TARGET], df.loc[mask, "sales_lag_1"])


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


def _fit_lgbm(train_df: pd.DataFrame, feature_cols: list[str], params: dict[str, Any]):
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
    return model


def train_and_log(
    df: pd.DataFrame,
    n_cv_splits: int = settings.n_cv_splits,
    n_param_samples: int = 6,
    param_grid: dict[str, list[Any]] | None = None,
    register_model: bool = True,
    test_weeks: int = settings.test_holdout_weeks,
) -> dict[str, Any]:
    """Run baseline + tuned LightGBM training, logging everything to MLflow.

    Three-stage split (chronological, never random — see `docs/USER_GUIDE.md`
    "Train / CV / Test" section for the full walkthrough with real dates):

    1. The most recent `test_weeks` weeks are held out as `test` and are
       *never* seen during hyperparameter search or training below —
       `chronological_holdout_split` carves them off first.
    2. Everything else (`pool`) is used for the baseline + the rolling-origin
       CV hyperparameter search (`time_series_cv_splits`), exactly as before.
    3. The best hyperparameters are (a) fit on `pool` alone and scored once
       on `test` for an honest, never-trained-on accuracy number
       (`test_metrics`), then (b) refit on the *full* `df` (pool + test) as
       the model that actually gets registered — recent data matters for a
       time series, so the production model shouldn't discard it once the
       honest test score has been recorded.

    Returns a summary dict with baseline metrics, best params, CV metrics,
    held-out test metrics, and the MLflow run id of the final (registered)
    model.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    _ensure_experiment(settings.mlflow_experiment_name, settings.mlflow_artifact_root)

    pool_df, test_df = chronological_holdout_split(df, test_weeks)
    logger.info(
        "Chronological split: %d weeks in train+CV pool (%s -> %s), "
        "%d weeks held out as test (%s -> %s, never trained on until step 3b)",
        pool_df["date"].nunique(),
        pool_df["date"].min().date(),
        pool_df["date"].max().date(),
        test_df["date"].nunique(),
        test_df["date"].min().date(),
        test_df["date"].max().date(),
    )

    splits = time_series_cv_splits(pool_df["date"], n_cv_splits)
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    # 1. Baseline — reference number across the whole dataset ("multiple
    #    experiments" for comparison), plus the same baseline restricted to
    #    the held-out test weeks for a fair side-by-side with test_metrics.
    logger.info("Computing naive (previous-week) baseline on %d rows...", len(df))
    baseline_metrics = naive_baseline(df)
    baseline_test_metrics = naive_baseline(test_df)
    logger.info(
        "Baseline RMSLE: %.4f (all data) / %.4f (test weeks only)",
        baseline_metrics["rmsle"],
        baseline_test_metrics["rmsle"],
    )
    with mlflow.start_run(run_name="baseline_naive_previous_week"):
        mlflow.set_tag("model_type", "baseline")
        mlflow.log_metrics({f"holdout_{k}": v for k, v in baseline_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in baseline_test_metrics.items()})

    # 2. Small random hyperparameter search over LightGBM on the pool only,
    #    each combo its own run.
    grid = param_grid or DEFAULT_PARAM_GRID
    candidates = sample_param_grid(grid, n_param_samples, seed=settings.random_seed)
    logger.info(
        "Starting hyperparameter search: %d candidates x %d CV folds = %d LightGBM fits "
        "on %d rows each (this is the slow part — no MLflow output happens between here "
        "and the next log line per candidate, that's expected, not a hang).",
        len(candidates),
        n_cv_splits,
        len(candidates) * n_cv_splits,
        len(pool_df),
    )

    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    best_fold_metrics: list[dict[str, float]] = []
    for i, params in enumerate(candidates, start=1):
        logger.info("[%d/%d] Fitting LightGBM with params=%s ...", i, len(candidates), params)
        with mlflow.start_run(run_name=f"lgbm_{'_'.join(str(v) for v in params.values())}"):
            mlflow.set_tag("model_type", "lightgbm")
            mlflow.log_params(params)
            mean_rmsle, fold_metrics = cv_score_params(pool_df, params, splits)
            for j, fm in enumerate(fold_metrics):
                mlflow.log_metrics({f"fold{j}_{k}": v for k, v in fm.items()})
            mlflow.log_metric("cv_mean_rmsle", mean_rmsle)
        logger.info("[%d/%d] cv_mean_rmsle=%.4f", i, len(candidates), mean_rmsle)
        if mean_rmsle < best_score:
            best_score = mean_rmsle
            best_params = params
            best_fold_metrics = fold_metrics

    assert best_params is not None
    logger.info("Best params from CV: %s (cv_mean_rmsle=%.4f)", best_params, best_score)

    # 3a. Honest held-out evaluation: fit ONLY on the pool, score ONCE on the
    #     test weeks the model has never seen in any form up to this point.
    logger.info(
        "Evaluating best params on the %d held-out test week(s) (never used for "
        "training or hyperparameter selection)...",
        test_df["date"].nunique(),
    )
    eval_model = _fit_lgbm(pool_df, feature_cols, best_params)
    test_preds = np.expm1(eval_model.predict(test_df[feature_cols]))
    test_metrics = evaluate_all(test_df[TARGET], test_preds)
    logger.info(
        "Held-out test RMSLE: %.4f (naive baseline on the same weeks: %.4f)",
        test_metrics["rmsle"],
        baseline_test_metrics["rmsle"],
    )
    with mlflow.start_run(run_name="holdout_test_evaluation"):
        mlflow.set_tag("model_type", "lightgbm_holdout_eval")
        mlflow.log_params(best_params)
        mlflow.log_param("test_weeks", test_weeks)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"baseline_test_{k}": v for k, v in baseline_test_metrics.items()})

    # 3b. Refit on the FULL dataset (pool + test) with the same best params —
    #     this is the model that actually gets logged/registered for serving.
    logger.info("Retraining the production model on the full dataset (pool + test)...")
    final_model = _fit_lgbm(df, feature_cols, best_params)
    logger.info("Final model trained. Logging model + artifacts to MLflow...")

    with mlflow.start_run(run_name="best_lightgbm_final") as run:
        mlflow.set_tag("model_type", "lightgbm_final")
        mlflow.log_params(best_params)
        mlflow.log_param("test_weeks", test_weeks)
        mlflow.log_metric("cv_mean_rmsle", best_score)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
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
        "baseline_test_metrics": baseline_test_metrics,
        "best_params": best_params,
        "best_cv_rmsle": best_score,
        "best_fold_metrics": best_fold_metrics,
        "test_metrics": test_metrics,
        "test_weeks": test_weeks,
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
        f"Best LightGBM CV RMSLE={summary['best_cv_rmsle']:.4f} | "
        f"Held-out test RMSLE={summary['test_metrics']['rmsle']:.4f} "
        f"(vs baseline {summary['baseline_test_metrics']['rmsle']:.4f} on same weeks) "
        f"(params={summary['best_params']})"
    )


if __name__ == "__main__":
    main()

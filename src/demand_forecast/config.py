"""Centralized configuration, loaded from environment variables (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    # Paths
    data_raw_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / _env("DATA_RAW_DIR", "data/raw")
    )
    data_processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / _env("DATA_PROCESSED_DIR", "data/processed")
    )
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")

    # MLflow
    mlflow_tracking_uri: str = field(
        default_factory=lambda: _env("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT}/mlflow.db")
    )
    mlflow_experiment_name: str = field(
        default_factory=lambda: _env("MLFLOW_EXPERIMENT_NAME", "demand-forecast")
    )
    mlflow_artifact_root: str = field(
        default_factory=lambda: _env(
            "MLFLOW_ARTIFACT_ROOT", (PROJECT_ROOT / "mlartifacts").as_posix()
        )
    )
    mlflow_model_name: str = field(
        default_factory=lambda: _env("MLFLOW_MODEL_NAME", "demand-forecast-lgbm")
    )
    mlflow_model_alias: str = field(
        default_factory=lambda: _env("MLFLOW_MODEL_ALIAS", "production")
    )

    # API
    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(_env("API_PORT", "8000")))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # Modeling
    random_seed: int = 42
    n_cv_splits: int = 3
    # Most recent N weeks held out as a true, never-trained-on test set for a
    # one-time honest accuracy check before the final model is refit on 100%
    # of the data and promoted — see models/train.py chronological_holdout_split.
    test_holdout_weeks: int = field(default_factory=lambda: int(_env("TEST_HOLDOUT_WEEKS", "8")))
    # scripts/retrain_if_stale.py: force a retrain attempt once the current
    # production model is at least this many days old.
    retrain_max_age_days: int = field(
        default_factory=lambda: int(_env("RETRAIN_MAX_AGE_DAYS", "7"))
    )
    # scripts/retrain_if_stale.py: start logging a heads-up WARNING once the
    # production model is within this many days of retrain_max_age_days, e.g.
    # default 7/2 -> a warning from day 5 ("2 day(s) left") through day 7.
    retrain_warning_lead_days: int = field(
        default_factory=lambda: int(_env("RETRAIN_WARNING_LEAD_DAYS", "2"))
    )


settings = Settings()

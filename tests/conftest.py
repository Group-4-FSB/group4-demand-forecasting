"""Shared pytest fixtures.

IMPORTANT: environment variables that `demand_forecast.config.Settings` reads
at import time are set here, at module scope, *before* anything under
`demand_forecast` is imported — conftest.py is always collected first, so
this reliably redirects MLflow / data paths to an isolated test-artifacts
directory instead of whatever the developer has configured for local dev.
"""

from __future__ import annotations

import os
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
ARTIFACTS_DIR = TESTS_DIR / ".test_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

_mlflow_db_path = (ARTIFACTS_DIR / "mlflow.db").as_posix()
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_mlflow_db_path}")
os.environ.setdefault("MLFLOW_ARTIFACT_ROOT", (ARTIFACTS_DIR / "mlartifacts").as_posix())
os.environ.setdefault("DATA_PROCESSED_DIR", str(ARTIFACTS_DIR / "processed"))
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "demand-forecast-test")
os.environ.setdefault("MLFLOW_MODEL_NAME", "demand-forecast-lgbm-test")

import pytest  # noqa: E402
from demand_forecast.data.features import build_features  # noqa: E402
from demand_forecast.data.ingest import load_walmart_sales  # noqa: E402
from demand_forecast.data.snapshot import save_reference_artifacts  # noqa: E402
from demand_forecast.models.train import train_and_log  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return load_walmart_sales(FIXTURES_DIR)


@pytest.fixture(scope="session")
def features_df(raw_df):
    return build_features(raw_df)


@pytest.fixture(scope="session")
def trained_summary(features_df):
    """Train + register one small model per test session (shared across tests
    that need a real fitted model, instead of retraining for each one)."""
    return train_and_log(features_df, n_cv_splits=2, n_param_samples=2, register_model=True)


@pytest.fixture(scope="session")
def reference_snapshot_dir(features_df, trained_summary):
    from demand_forecast.config import settings

    save_reference_artifacts(features_df, settings.data_processed_dir)
    return settings.data_processed_dir


@pytest.fixture(scope="session")
def api_client(trained_summary, reference_snapshot_dir):
    from demand_forecast.api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client

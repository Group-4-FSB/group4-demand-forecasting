"""Tests for the promotion quality gate in models/train.py.

These use their own isolated `mlflow_model_name` (via monkeypatching the
`settings` object `train.py` sees) so they never touch the shared
`demand-forecast-lgbm-test` model that the session-scoped `trained_summary`
fixture and other tests depend on.
"""

from __future__ import annotations

import dataclasses

import demand_forecast.models.train as train_module
import mlflow
import pytest
from demand_forecast.config import settings as global_settings


@pytest.fixture()
def isolated_train_settings(monkeypatch, request):
    """Point train.py at a throwaway, per-test-unique model name so tests in
    this file never see each other's registered versions/aliases regardless
    of execution order."""
    model_name = f"{global_settings.mlflow_model_name}-gate-{request.node.name}"[:200]
    isolated = dataclasses.replace(global_settings, mlflow_model_name=model_name)
    monkeypatch.setattr(train_module, "settings", isolated)
    return isolated


def test_gate_promotes_when_no_prior_production_exists(features_df, isolated_train_settings):
    summary = train_module.train_and_log(
        features_df, n_cv_splits=2, n_param_samples=2, register_model=True
    )
    assert summary["promoted"] is True
    assert summary["current_production_test_rmsle"] is None
    assert summary["run_id"] is not None
    assert summary["model_uri"] is not None


def test_gate_rejects_a_worse_candidate_and_leaves_alias_unchanged(
    features_df, isolated_train_settings
):
    client = mlflow.MlflowClient()

    # First registration always promotes (nothing to compare against yet).
    summary1 = train_module.train_and_log(
        features_df, n_cv_splits=2, n_param_samples=2, register_model=True
    )
    assert summary1["promoted"] is True

    # Artificially make the just-registered production version "unbeatable"
    # so the next real training attempt is guaranteed to look worse.
    client.log_metric(summary1["run_id"], "test_rmsle", 1e-9)

    summary2 = train_module.train_and_log(
        features_df, n_cv_splits=2, n_param_samples=2, register_model=True
    )

    assert summary2["promoted"] is False
    assert summary2["run_id"] is None
    assert summary2["model_uri"] is None
    assert summary2["current_production_test_rmsle"] == pytest.approx(1e-9)
    # the rejected attempt's own test_metrics are still returned, for logging
    assert summary2["test_metrics"]["rmsle"] > summary2["current_production_test_rmsle"]

    # The alias must still point at version 1 — nothing new was promoted.
    mv = client.get_model_version_by_alias(
        isolated_train_settings.mlflow_model_name, isolated_train_settings.mlflow_model_alias
    )
    assert str(mv.version) == "1"


def test_gate_promotes_a_tying_candidate(features_df, isolated_train_settings):
    """A same-quality retrain (e.g. identical data + seed) should still be
    promoted — the gate only blocks strictly worse candidates."""
    client = mlflow.MlflowClient()

    summary1 = train_module.train_and_log(
        features_df, n_cv_splits=2, n_param_samples=2, register_model=True
    )
    assert summary1["promoted"] is True

    # Same features_df + fixed random_seed -> deterministic identical result.
    summary2 = train_module.train_and_log(
        features_df, n_cv_splits=2, n_param_samples=2, register_model=True
    )
    assert summary2["promoted"] is True
    assert summary2["test_metrics"]["rmsle"] == pytest.approx(summary1["test_metrics"]["rmsle"])

    mv = client.get_model_version_by_alias(
        isolated_train_settings.mlflow_model_name, isolated_train_settings.mlflow_model_alias
    )
    assert str(mv.version) == "2"


def test_gate_disabled_when_register_model_is_false(features_df, isolated_train_settings):
    summary = train_module.train_and_log(
        features_df, n_cv_splits=2, n_param_samples=2, register_model=False
    )
    # Nothing to gate against since nothing gets registered either way — the
    # model artifact is still logged to its run (model_uri), just never
    # turned into a named registry version or aliased.
    assert summary["promoted"] is True
    assert summary["current_production_test_rmsle"] is None
    assert summary["run_id"] is not None
    assert summary["model_uri"] is not None

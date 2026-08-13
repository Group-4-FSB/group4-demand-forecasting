"""Tests for scripts/retrain_if_stale.py — the weekly-retrain business rule.

scripts/ isn't part of the installed package, so it's added to sys.path here
the same way the scripts themselves add `src/` to theirs.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mlflow  # noqa: E402
import retrain_if_stale as scheduler  # noqa: E402

# ---- is_retrain_due: pure logic, no I/O ----------------------------------


def test_is_retrain_due_when_no_existing_model():
    assert scheduler.is_retrain_due(None, max_age_days=7) is True


def test_is_retrain_due_when_fresh():
    assert scheduler.is_retrain_due(timedelta(hours=1), max_age_days=7) is False


def test_is_retrain_due_when_stale():
    assert scheduler.is_retrain_due(timedelta(days=8), max_age_days=7) is True


def test_is_retrain_due_at_exact_boundary():
    # exactly N days old counts as due today, not "one day too early"
    assert scheduler.is_retrain_due(timedelta(days=7), max_age_days=7) is True


def test_is_retrain_due_just_under_boundary():
    assert scheduler.is_retrain_due(timedelta(days=6, hours=23), max_age_days=7) is False


# ---- is_approaching_staleness: pure logic, no I/O -------------------------


def test_not_approaching_when_no_existing_model():
    # "due" (always retrain), not "approaching" — these are mutually exclusive.
    assert scheduler.is_approaching_staleness(None, max_age_days=7, warning_lead_days=2) is False


def test_not_approaching_when_well_within_limit():
    assert (
        scheduler.is_approaching_staleness(timedelta(days=1), max_age_days=7, warning_lead_days=2)
        is False
    )


def test_approaching_at_lead_boundary():
    # day 5 of 7, with a 2-day lead -> warning starts exactly here
    assert (
        scheduler.is_approaching_staleness(timedelta(days=5), max_age_days=7, warning_lead_days=2)
        is True
    )


def test_not_approaching_just_under_lead_boundary():
    assert (
        scheduler.is_approaching_staleness(
            timedelta(days=4, hours=23), max_age_days=7, warning_lead_days=2
        )
        is False
    )


def test_not_approaching_once_already_stale():
    # already due -> is_retrain_due owns this case, not the "approaching" one
    assert (
        scheduler.is_approaching_staleness(timedelta(days=8), max_age_days=7, warning_lead_days=2)
        is False
    )


# ---- get_production_model_age: real MLflow lookup ------------------------


def test_get_production_model_age_matches_a_just_registered_model(trained_summary):
    client = mlflow.MlflowClient()
    from demand_forecast.config import settings

    age = scheduler.get_production_model_age(
        client, settings.mlflow_model_name, settings.mlflow_model_alias
    )
    assert age is not None
    assert timedelta(0) <= age < timedelta(minutes=10)


def test_get_production_model_age_none_for_unknown_model():
    client = mlflow.MlflowClient()
    age = scheduler.get_production_model_age(client, "no-such-model-xyz", "production")
    assert age is None


# ---- check_and_retrain: orchestration, with run_training_pipeline mocked -


def test_check_and_retrain_skips_when_not_due(monkeypatch):
    calls = {"n": 0}

    def fake_run_training_pipeline(**kwargs):
        calls["n"] += 1
        return {"promoted": True}

    monkeypatch.setattr(scheduler, "run_training_pipeline", fake_run_training_pipeline)
    monkeypatch.setattr(scheduler, "get_production_model_age", lambda *a, **k: timedelta(hours=1))

    result = scheduler.check_and_retrain(max_age_days=7)

    assert result is False
    assert calls["n"] == 0


def test_check_and_retrain_retrains_when_due_and_promoted(monkeypatch):
    calls = {"n": 0}

    def fake_run_training_pipeline(**kwargs):
        calls["n"] += 1
        return {
            "promoted": True,
            "test_metrics": {"rmsle": 0.05},
            "current_production_test_rmsle": None,
        }

    monkeypatch.setattr(scheduler, "run_training_pipeline", fake_run_training_pipeline)
    monkeypatch.setattr(scheduler, "get_production_model_age", lambda *a, **k: None)

    result = scheduler.check_and_retrain(max_age_days=7)

    assert result is True
    assert calls["n"] == 1


def test_check_and_retrain_warns_when_approaching_staleness(monkeypatch, caplog):
    monkeypatch.setattr(scheduler, "get_production_model_age", lambda *a, **k: timedelta(days=6))

    with caplog.at_level("WARNING", logger=scheduler.logger.name):
        result = scheduler.check_and_retrain(max_age_days=7, warning_lead_days=2)

    assert result is False
    assert any("approaching staleness" in r.message for r in caplog.records)


def test_check_and_retrain_stays_info_when_comfortably_fresh(monkeypatch, caplog):
    monkeypatch.setattr(scheduler, "get_production_model_age", lambda *a, **k: timedelta(days=1))

    with caplog.at_level("INFO", logger=scheduler.logger.name):
        result = scheduler.check_and_retrain(max_age_days=7, warning_lead_days=2)

    assert result is False
    assert not any(r.levelname == "WARNING" for r in caplog.records)
    assert any("Nothing to do" in r.message for r in caplog.records)


def test_check_and_retrain_retrains_but_gate_rejects(monkeypatch, caplog):
    def fake_run_training_pipeline(**kwargs):
        return {
            "promoted": False,
            "test_metrics": {"rmsle": 0.09},
            "current_production_test_rmsle": 0.04,
        }

    monkeypatch.setattr(scheduler, "run_training_pipeline", fake_run_training_pipeline)
    monkeypatch.setattr(scheduler, "get_production_model_age", lambda *a, **k: timedelta(days=8))

    result = scheduler.check_and_retrain(max_age_days=7)

    # A retrain WAS attempted (that's what this function is responsible for)
    # even though the quality gate then rejected the candidate.
    assert result is True

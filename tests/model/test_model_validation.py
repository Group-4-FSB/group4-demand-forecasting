from __future__ import annotations

import numpy as np
from demand_forecast.data.features import ALL_FEATURES
from demand_forecast.models.predict import PredictionService, UnknownStoreError
from demand_forecast.models.train import naive_baseline

# Loose sanity bound, not a tight accuracy target: catches "something is
# fundamentally broken" (e.g. predicting on the wrong scale) without being
# flaky on the small fixture dataset used for fast CI runs.
SANITY_RMSLE_UPPER_BOUND = 1.0


def test_baseline_metrics_are_finite_and_reasonable(features_df):
    metrics = naive_baseline(features_df)
    assert set(metrics) == {"rmsle", "mae", "rmse"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["rmsle"] < SANITY_RMSLE_UPPER_BOUND


def test_trained_model_cv_rmsle_within_sanity_bound(trained_summary):
    assert trained_summary["best_cv_rmsle"] < SANITY_RMSLE_UPPER_BOUND
    for fold_metrics in trained_summary["best_fold_metrics"]:
        assert fold_metrics["rmsle"] < SANITY_RMSLE_UPPER_BOUND


def test_trained_model_has_honest_holdout_test_metrics(trained_summary):
    """The registered model must come with a held-out test score computed by
    a model that never saw those weeks — see train.py's chronological split."""
    assert trained_summary["test_weeks"] > 0
    test_metrics = trained_summary["test_metrics"]
    assert set(test_metrics) == {"rmsle", "mae", "rmse"}
    assert all(np.isfinite(v) for v in test_metrics.values())
    assert test_metrics["rmsle"] < SANITY_RMSLE_UPPER_BOUND
    # baseline_test_metrics is the naive baseline on the SAME held-out weeks,
    # for a fair apples-to-apples comparison against test_metrics.
    assert trained_summary["baseline_test_metrics"]["rmsle"] > 0


def test_first_ever_registration_is_always_promoted(trained_summary):
    # trained_summary is this test session's first (and only, per session)
    # training run for its isolated model name, so there is nothing to gate
    # against yet — the quality gate must let it through unconditionally.
    assert trained_summary["promoted"] is True
    assert trained_summary["current_production_test_rmsle"] is None
    assert trained_summary["run_id"] is not None
    assert trained_summary["model_uri"] is not None


def test_trained_model_predictions_are_non_negative(trained_summary, features_df):
    model = trained_summary["model"]
    feature_cols = trained_summary["feature_columns"]
    preds = np.expm1(model.predict(features_df[feature_cols]))
    assert (preds >= 0).all()


def test_trained_model_uses_expected_feature_set(trained_summary):
    assert set(trained_summary["feature_columns"]) <= set(ALL_FEATURES)
    assert len(trained_summary["feature_columns"]) > 0


def test_prediction_service_predicts_known_store(reference_snapshot_dir, trained_summary):
    service = PredictionService.load(model_uri=trained_summary["model_uri"])
    value = service.predict(store_nbr=1, date="2012-11-02")
    assert isinstance(value, float)
    assert value >= 0.0


def test_prediction_service_auto_detects_holiday_week(reference_snapshot_dir, trained_summary):
    service = PredictionService.load(model_uri=trained_summary["model_uri"])
    row = service.build_feature_row(store_nbr=1, date="2012-12-28")  # Christmas week Friday
    assert row["holiday_flag"].iloc[0] == 1
    non_holiday_row = service.build_feature_row(store_nbr=1, date="2012-11-02")
    assert non_holiday_row["holiday_flag"].iloc[0] == 0


def test_prediction_service_respects_explicit_overrides(reference_snapshot_dir, trained_summary):
    service = PredictionService.load(model_uri=trained_summary["model_uri"])
    row = service.build_feature_row(store_nbr=1, date="2012-11-02", temperature=12.3)
    assert row["temperature"].iloc[0] == 12.3


def test_prediction_service_rejects_unknown_store(reference_snapshot_dir, trained_summary):
    service = PredictionService.load(model_uri=trained_summary["model_uri"])
    try:
        service.predict(store_nbr=987654, date="2012-11-02")
        assert False, "expected UnknownStoreError"
    except UnknownStoreError:
        pass

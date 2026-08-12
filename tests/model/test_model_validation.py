from __future__ import annotations

import numpy as np
from demand_forecast.data.features import ALL_FEATURES
from demand_forecast.models.predict import PredictionService, UnknownStoreFamilyError
from demand_forecast.models.train import seasonal_naive_baseline

# Loose sanity bound, not a tight accuracy target: catches "something is
# fundamentally broken" (e.g. predicting on the wrong scale) without being
# flaky on the small fixture dataset used for fast CI runs.
SANITY_RMSLE_UPPER_BOUND = 1.0


def test_baseline_metrics_are_finite_and_reasonable(features_df):
    metrics = seasonal_naive_baseline(features_df)
    assert set(metrics) == {"rmsle", "mae", "rmse"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["rmsle"] < SANITY_RMSLE_UPPER_BOUND


def test_trained_model_cv_rmsle_within_sanity_bound(trained_summary):
    assert trained_summary["best_cv_rmsle"] < SANITY_RMSLE_UPPER_BOUND
    for fold_metrics in trained_summary["best_fold_metrics"]:
        assert fold_metrics["rmsle"] < SANITY_RMSLE_UPPER_BOUND


def test_trained_model_predictions_are_non_negative(trained_summary, features_df):
    model = trained_summary["model"]
    feature_cols = trained_summary["feature_columns"]
    preds = np.expm1(model.predict(features_df[feature_cols]))
    assert (preds >= 0).all()


def test_trained_model_uses_expected_feature_set(trained_summary):
    assert set(trained_summary["feature_columns"]) <= set(ALL_FEATURES)
    assert len(trained_summary["feature_columns"]) > 0


def test_prediction_service_predicts_known_combo(reference_snapshot_dir, trained_summary):
    service = PredictionService.load(model_uri=trained_summary["model_uri"])
    value = service.predict(store_nbr=1, family="DAIRY", date="2017-08-20", onpromotion=0)
    assert isinstance(value, float)
    assert value >= 0.0


def test_prediction_service_rejects_unknown_store(reference_snapshot_dir, trained_summary):
    service = PredictionService.load(model_uri=trained_summary["model_uri"])
    try:
        service.predict(store_nbr=99999, family="DAIRY", date="2017-08-20", onpromotion=0)
        assert False, "expected UnknownStoreFamilyError"
    except UnknownStoreFamilyError:
        pass

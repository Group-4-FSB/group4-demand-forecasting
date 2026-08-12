from __future__ import annotations

import numpy as np
import pytest
from demand_forecast.models.evaluate import evaluate_all, mae, rmse, rmsle


def test_rmsle_zero_when_predictions_are_perfect():
    y = [1.0, 5.0, 100.0, 0.0]
    assert rmsle(y, y) == pytest.approx(0.0, abs=1e-9)


def test_rmsle_known_value():
    y_true = [np.e - 1]  # log1p(y_true) == 1
    y_pred = [np.e**2 - 1]  # log1p(y_pred) == 2
    assert rmsle(y_true, y_pred) == pytest.approx(1.0, rel=1e-6)


def test_rmsle_clips_negative_values():
    # negative predictions/targets should not raise (log1p of negative would NaN)
    result = rmsle([-5.0, 10.0], [-3.0, 12.0])
    assert np.isfinite(result)


def test_mae_basic():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert mae([0, 0, 0], [1, 2, 3]) == pytest.approx(2.0)


def test_rmse_basic():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_evaluate_all_returns_expected_keys():
    result = evaluate_all([1, 2, 3], [1, 2, 4])
    assert set(result.keys()) == {"rmsle", "mae", "rmse"}
    assert all(isinstance(v, float) for v in result.values())

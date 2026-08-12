from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from demand_forecast.data.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    LOG_TARGET,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    select_model_columns,
)


def _toy_frame() -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": list(dates) * 1,
            "store_nbr": [1] * 10,
            "family": ["DAIRY"] * 10,
            "sales": np.arange(10, dtype=float),
        }
    )


def test_add_calendar_features_known_date():
    df = pd.DataFrame({"date": [pd.Timestamp("2017-08-20")]})  # a Sunday
    out = add_calendar_features(df)
    assert out.loc[0, "day_of_week"] == 6
    assert out.loc[0, "is_weekend"] == 1
    assert out.loc[0, "month"] == 8
    assert out.loc[0, "day_of_month"] == 20


def test_add_calendar_features_weekday_is_not_weekend():
    df = pd.DataFrame({"date": [pd.Timestamp("2017-08-16")]})  # a Wednesday
    out = add_calendar_features(df)
    assert out.loc[0, "day_of_week"] == 2
    assert out.loc[0, "is_weekend"] == 0


def test_add_lag_features_shifts_by_group():
    df = _toy_frame()
    out = add_lag_features(df, lags=(1,))
    # sales_lag_1 on day i should equal sales on day i-1, within the same group
    expected = [np.nan] + list(range(9))
    np.testing.assert_allclose(out["sales_lag_1"].to_numpy(), expected)


def test_add_lag_features_does_not_leak_across_groups():
    df = pd.concat([_toy_frame(), _toy_frame().assign(store_nbr=2)], ignore_index=True)
    out = add_lag_features(df, lags=(1,))
    # first row of the second group must not see the last row of the first group
    second_group_first_row = out[out["store_nbr"] == 2].iloc[0]
    assert np.isnan(second_group_first_row["sales_lag_1"])


def test_add_rolling_features_adds_expected_columns():
    df = _toy_frame()
    out = add_rolling_features(df, windows=(3,))
    assert "sales_roll_mean_3" in out.columns
    assert "sales_roll_std_7" in out.columns


def test_build_features_on_merged_fixture_has_expected_columns(merged_df):
    out = build_features(merged_df)
    for col in ["day_of_week", "is_weekend", "sales_lag_7", "sales_roll_mean_7", LOG_TARGET]:
        assert col in out.columns
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            assert str(out[col].dtype) == "category"


def test_build_features_log_target_matches_log1p(merged_df):
    out = build_features(merged_df)
    np.testing.assert_allclose(out[LOG_TARGET], np.log1p(out["sales"].clip(lower=0)))


def test_select_model_columns_only_returns_known_features(merged_df):
    out = build_features(merged_df)
    selected = select_model_columns(out)
    assert set(selected.columns) <= set(ALL_FEATURES)
    assert "sales" not in selected.columns


@pytest.mark.parametrize("lag", [7, 14, 28])
def test_lag_features_present_for_supported_lags(lag):
    df = _toy_frame()
    # extend enough rows so the lag has data to reference
    dates = pd.date_range("2017-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {"date": dates, "store_nbr": 1, "family": "DAIRY", "sales": np.arange(40, dtype=float)}
    )
    out = add_lag_features(df, lags=(lag,))
    assert f"sales_lag_{lag}" in out.columns
    assert out[f"sales_lag_{lag}"].notna().sum() == 40 - lag

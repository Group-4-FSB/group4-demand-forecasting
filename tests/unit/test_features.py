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
    is_major_holiday_week,
    select_model_columns,
)

# The 10 Fridays actually flagged holiday_flag=1 in the source dataset
# (within its 2010-02-05..2012-10-26 range) — the ground truth this
# project's rule-based heuristic must reproduce exactly.
ACTUAL_HOLIDAY_FRIDAYS = [
    "2010-02-12",
    "2010-09-10",
    "2010-11-26",
    "2010-12-31",
    "2011-02-11",
    "2011-09-09",
    "2011-11-25",
    "2011-12-30",
    "2012-02-10",
    "2012-09-07",
]

# A handful of ordinary (non-holiday) Fridays from the same years, to check
# for false positives too.
ORDINARY_FRIDAYS = ["2010-03-05", "2011-06-17", "2012-04-06", "2012-10-26"]


def _toy_frame(n_weeks: int = 10, store_nbr: int = 1) -> pd.DataFrame:
    dates = pd.date_range("2010-02-05", periods=n_weeks, freq="7D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_nbr": store_nbr,
            "sales": np.arange(n_weeks, dtype=float) * 1000,
            "holiday_flag": 0,
            "temperature": 50.0,
            "fuel_price": 2.5,
            "cpi": 200.0,
            "unemployment": 8.0,
        }
    )


@pytest.mark.parametrize("date_str", ACTUAL_HOLIDAY_FRIDAYS)
def test_is_major_holiday_week_matches_real_dataset_positives(date_str):
    assert is_major_holiday_week(pd.Timestamp(date_str)) == 1


@pytest.mark.parametrize("date_str", ORDINARY_FRIDAYS)
def test_is_major_holiday_week_matches_real_dataset_negatives(date_str):
    assert is_major_holiday_week(pd.Timestamp(date_str)) == 0


def test_is_major_holiday_week_generalizes_to_future_years():
    # 2013 Thanksgiving week Friday (4th Thursday of Nov 2013 = Nov 28 -> Fri Nov 29)
    assert is_major_holiday_week(pd.Timestamp("2013-11-29")) == 1
    assert is_major_holiday_week(pd.Timestamp("2013-07-05")) == 0


def test_add_calendar_features_adds_expected_columns():
    df = _toy_frame()
    out = add_calendar_features(df)
    assert out.loc[0, "month"] == 2
    assert out.loc[0, "year"] == 2010
    assert "week_of_year" in out.columns


def test_add_lag_features_shifts_by_week_within_store():
    df = _toy_frame()
    out = add_lag_features(df, lags=(1,))
    expected = [np.nan] + list(np.arange(9) * 1000.0)
    np.testing.assert_allclose(out["sales_lag_1"].to_numpy(), expected)


def test_add_lag_features_does_not_leak_across_stores():
    df = pd.concat([_toy_frame(store_nbr=1), _toy_frame(store_nbr=2)], ignore_index=True)
    out = add_lag_features(df, lags=(1,))
    second_store_first_row = out[out["store_nbr"] == 2].iloc[0]
    assert np.isnan(second_store_first_row["sales_lag_1"])


def test_add_rolling_features_adds_expected_columns():
    df = _toy_frame()
    out = add_rolling_features(df, windows=(4,))
    assert "sales_roll_mean_4" in out.columns
    assert "sales_roll_std_4" in out.columns


def test_build_features_on_fixture_has_expected_columns(raw_df):
    out = build_features(raw_df)
    for col in [
        "month",
        "week_of_year",
        "year",
        "sales_lag_1",
        "sales_lag_4",
        "sales_lag_52",
        "sales_roll_mean_4",
        LOG_TARGET,
    ]:
        assert col in out.columns
    for col in CATEGORICAL_FEATURES:
        assert str(out[col].dtype) == "category"


def test_build_features_log_target_matches_log1p(raw_df):
    out = build_features(raw_df)
    np.testing.assert_allclose(out[LOG_TARGET], np.log1p(out["sales"].clip(lower=0)))


def test_select_model_columns_only_returns_known_features(raw_df):
    out = build_features(raw_df)
    selected = select_model_columns(out)
    assert set(selected.columns) <= set(ALL_FEATURES)
    assert "sales" not in selected.columns


def test_lag_52_mostly_null_within_first_year(raw_df):
    # only ~2.7 years of history exist per store, so lag_52 (year-over-year)
    # is NaN for each store's first 52 weeks by construction.
    out = build_features(raw_df)
    first_year = out.groupby("store_nbr", observed=True).head(52)
    assert first_year["sales_lag_52"].isna().all()

from __future__ import annotations

import pytest
from demand_forecast.data.ingest import load_walmart_sales


def test_load_walmart_sales_returns_expected_columns(raw_df):
    expected = {
        "store_nbr",
        "date",
        "sales",
        "holiday_flag",
        "temperature",
        "fuel_price",
        "cpi",
        "unemployment",
    }
    assert expected <= set(raw_df.columns)
    assert len(raw_df) > 0


def test_load_walmart_sales_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_walmart_sales(tmp_path)


def test_load_walmart_sales_parses_dates_and_sorts(raw_df):
    assert str(raw_df["date"].dtype).startswith("datetime64")
    # sorted by store_nbr then date within each store
    for _, group in raw_df.groupby("store_nbr"):
        assert group["date"].is_monotonic_increasing


def test_load_walmart_sales_holiday_flag_is_binary(raw_df):
    assert raw_df["holiday_flag"].isin([0, 1]).all()

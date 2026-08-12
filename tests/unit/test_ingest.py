from __future__ import annotations

import pandas as pd
import pytest
from demand_forecast.data.ingest import build_holiday_flags, load_raw_tables, merge_dataset


def test_load_raw_tables_returns_all_expected_keys(raw_tables):
    assert set(raw_tables) == {"train", "test", "stores", "oil", "holidays", "transactions"}
    for df in raw_tables.values():
        assert len(df) > 0


def test_load_raw_tables_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_tables(tmp_path)


def test_merge_dataset_invalid_split_raises(raw_tables):
    with pytest.raises(ValueError):
        merge_dataset(raw_tables, split="not_a_split")


def test_merge_dataset_train_has_sales_column(raw_tables):
    merged = merge_dataset(raw_tables, split="train")
    assert "sales" in merged.columns
    assert "oil_price" in merged.columns
    assert "is_holiday" in merged.columns


def test_merge_dataset_test_has_no_sales_column(raw_tables):
    merged = merge_dataset(raw_tables, split="test")
    assert "sales" not in merged.columns


def test_build_holiday_flags_excludes_transferred_holidays():
    holidays = pd.DataFrame(
        {
            "date": [pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-02")],
            "type": ["Holiday", "Event"],
            "locale": ["National", "National"],
            "locale_name": ["Ecuador", "Ecuador"],
            "description": ["x", "y"],
            "transferred": [True, False],
        }
    )
    daily = build_holiday_flags(holidays)
    # the transferred Holiday row should be dropped entirely
    assert pd.Timestamp("2017-01-01") not in daily["date"].to_numpy()
    row = daily.loc[daily["date"] == pd.Timestamp("2017-01-02")].iloc[0]
    assert row["is_event"] == 1
    assert row["is_holiday"] == 0

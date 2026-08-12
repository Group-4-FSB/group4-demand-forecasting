from __future__ import annotations

import pandas as pd
import pytest
from demand_forecast.data.validate import (
    ValidationReport,
    validate_duplicates,
    validate_nulls,
    validate_panel_balance,
    validate_ranges,
    validate_sales_table,
    validate_schema,
)


def test_real_fixture_passes_full_validation(raw_df):
    report = validate_sales_table(raw_df)
    assert report.ok, report.errors


def test_schema_detects_missing_column():
    df = pd.DataFrame({"date": [pd.Timestamp("2010-02-05")], "store_nbr": [1]})
    report = ValidationReport()
    validate_schema(df, ["date", "store_nbr", "sales"], report)
    assert not report.ok
    assert "sales" in report.errors[0]


def test_nulls_detected():
    df = pd.DataFrame({"date": [pd.Timestamp("2010-02-05"), None], "store_nbr": [1, 2]})
    report = ValidationReport()
    validate_nulls(df, ["date"], report)
    assert not report.ok
    assert "date" in report.errors[0]


def test_negative_sales_detected():
    df = pd.DataFrame(
        {
            "sales": [10.0, -5.0, 3.0],
            "holiday_flag": [0, 0, 0],
            "unemployment": [8.0, 8.0, 8.0],
            "fuel_price": [2.5, 2.5, 2.5],
            "cpi": [200.0, 200.0, 200.0],
            "store_nbr": [1, 1, 1],
        }
    )
    report = ValidationReport()
    validate_ranges(df, report)
    assert not report.ok
    assert any("sales" in e for e in report.errors)


def test_holiday_flag_outside_binary_detected():
    df = pd.DataFrame(
        {
            "sales": [10.0],
            "holiday_flag": [2],
            "unemployment": [8.0],
            "fuel_price": [2.5],
            "cpi": [200.0],
            "store_nbr": [1],
        }
    )
    report = ValidationReport()
    validate_ranges(df, report)
    assert not report.ok
    assert any("holiday_flag" in e for e in report.errors)


def test_negative_unemployment_detected():
    df = pd.DataFrame(
        {
            "sales": [10.0],
            "holiday_flag": [0],
            "unemployment": [-1.0],
            "fuel_price": [2.5],
            "cpi": [200.0],
            "store_nbr": [1],
        }
    )
    report = ValidationReport()
    validate_ranges(df, report)
    assert not report.ok
    assert any("unemployment" in e for e in report.errors)


def test_non_positive_fuel_price_detected():
    df = pd.DataFrame(
        {
            "sales": [10.0],
            "holiday_flag": [0],
            "unemployment": [8.0],
            "fuel_price": [0.0],
            "cpi": [200.0],
            "store_nbr": [1],
        }
    )
    report = ValidationReport()
    validate_ranges(df, report)
    assert not report.ok
    assert any("fuel_price" in e for e in report.errors)


def test_non_positive_cpi_detected():
    df = pd.DataFrame(
        {
            "sales": [10.0],
            "holiday_flag": [0],
            "unemployment": [8.0],
            "fuel_price": [2.5],
            "cpi": [-5.0],
            "store_nbr": [1],
        }
    )
    report = ValidationReport()
    validate_ranges(df, report)
    assert not report.ok
    assert any("cpi" in e for e in report.errors)


def test_store_nbr_below_one_detected():
    df = pd.DataFrame(
        {
            "sales": [10.0],
            "holiday_flag": [0],
            "unemployment": [8.0],
            "fuel_price": [2.5],
            "cpi": [200.0],
            "store_nbr": [0],
        }
    )
    report = ValidationReport()
    validate_ranges(df, report)
    assert not report.ok
    assert any("store_nbr" in e for e in report.errors)


def test_implausible_temperature_warns_not_errors():
    df = pd.DataFrame(
        {
            "sales": [10.0],
            "holiday_flag": [0],
            "unemployment": [8.0],
            "fuel_price": [2.5],
            "cpi": [200.0],
            "store_nbr": [1],
            "temperature": [200.0],
        }
    )
    report = ValidationReport()
    validate_ranges(df, report)
    assert report.ok  # warning, not a hard failure
    assert report.warnings


def test_valid_ranges_pass():
    df = pd.DataFrame(
        {
            "sales": [0.0, 10.0],
            "holiday_flag": [0, 1],
            "unemployment": [7.5, 8.2],
            "fuel_price": [2.5, 2.6],
            "cpi": [200.0, 201.0],
            "store_nbr": [1, 2],
            "temperature": [45.0, 90.0],
        }
    )
    report = ValidationReport()
    validate_ranges(df, report)
    assert report.ok


def test_duplicates_detected():
    df = pd.DataFrame({"date": [pd.Timestamp("2010-02-05")] * 2, "store_nbr": [1, 1]})
    report = ValidationReport()
    validate_duplicates(df, ["date", "store_nbr"], report)
    assert not report.ok


def test_panel_balance_warns_on_uneven_weeks():
    df = pd.DataFrame(
        {
            "store_nbr": [1, 1, 2],
            "date": pd.to_datetime(["2010-02-05", "2010-02-12", "2010-02-05"]),
        }
    )
    report = ValidationReport()
    validate_panel_balance(df, report)
    assert not report.errors  # imbalance is a warning, not a hard failure
    assert report.warnings


def test_validation_error_raised_when_report_has_errors():
    report = ValidationReport(errors=["something is wrong"])
    with pytest.raises(Exception):
        report.raise_if_failed()


def test_validation_no_raise_when_report_ok():
    report = ValidationReport()
    report.raise_if_failed()  # should not raise

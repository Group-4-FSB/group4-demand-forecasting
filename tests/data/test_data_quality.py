from __future__ import annotations

import pandas as pd
import pytest
from demand_forecast.data.validate import (
    ValidationReport,
    validate_duplicates,
    validate_nulls,
    validate_ranges,
    validate_referential_integrity,
    validate_sales_table,
    validate_schema,
)


def test_real_fixture_passes_full_validation(merged_df, raw_tables):
    report = validate_sales_table(merged_df, raw_tables["stores"])
    assert report.ok, report.errors


def test_schema_detects_missing_column():
    df = pd.DataFrame({"date": [pd.Timestamp("2017-01-01")], "store_nbr": [1]})
    report = ValidationReport()
    validate_schema(df, {"date": "x", "family": "x", "sales": "x"}, report)
    assert not report.ok
    assert "family" in report.errors[0]


def test_nulls_detected():
    df = pd.DataFrame({"date": [pd.Timestamp("2017-01-01"), None], "store_nbr": [1, 2]})
    report = ValidationReport()
    validate_nulls(df, ["date"], report)
    assert not report.ok
    assert "date" in report.errors[0]


def test_negative_sales_detected():
    df = pd.DataFrame({"sales": [10.0, -5.0, 3.0], "onpromotion": [0, 0, 0]})
    report = ValidationReport()
    validate_ranges(df, report)
    assert not report.ok
    assert "sales" in report.errors[0]


def test_negative_onpromotion_detected():
    df = pd.DataFrame({"sales": [1.0], "onpromotion": [-1]})
    report = ValidationReport()
    validate_ranges(df, report)
    assert not report.ok


def test_valid_ranges_pass():
    df = pd.DataFrame({"sales": [0.0, 10.0], "onpromotion": [0, 3]})
    report = ValidationReport()
    validate_ranges(df, report)
    assert report.ok


def test_referential_integrity_detects_unknown_store():
    df = pd.DataFrame({"store_nbr": [1, 2, 999]})
    stores = pd.DataFrame({"store_nbr": [1, 2, 3]})
    report = ValidationReport()
    validate_referential_integrity(df, stores, report)
    assert not report.ok
    assert "999" in report.errors[0]


def test_duplicates_detected():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2017-01-01")] * 2,
            "store_nbr": [1, 1],
            "family": ["DAIRY", "DAIRY"],
        }
    )
    report = ValidationReport()
    validate_duplicates(df, ["date", "store_nbr", "family"], report)
    assert not report.ok


def test_validation_error_raised_when_report_has_errors():
    report = ValidationReport(errors=["something is wrong"])
    with pytest.raises(Exception):
        report.raise_if_failed()


def test_validation_no_raise_when_report_ok():
    report = ValidationReport()
    report.raise_if_failed()  # should not raise

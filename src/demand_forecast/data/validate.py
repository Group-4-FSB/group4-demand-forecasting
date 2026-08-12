"""Lightweight data-quality validation (schema, nulls, ranges, duplicates).

Deliberately dependency-free (plain pandas) to keep the pipeline lean — no
extra framework such as Great Expectations is required to express these
checks clearly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class DataValidationError(Exception):
    """Raised when one or more required (non-negotiable) checks fail."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise DataValidationError("; ".join(self.errors))


REQUIRED_COLUMNS = [
    "date",
    "store_nbr",
    "sales",
    "holiday_flag",
    "temperature",
    "fuel_price",
    "cpi",
    "unemployment",
]


def validate_schema(df: pd.DataFrame, required: list[str], report: ValidationReport) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required columns: {missing}")


def validate_nulls(df: pd.DataFrame, critical_cols: list[str], report: ValidationReport) -> None:
    for col in critical_cols:
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            report.errors.append(f"Column '{col}' has {n_null} null value(s)")


def validate_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    if "sales" in df.columns and (df["sales"] < 0).any():
        n_bad = int((df["sales"] < 0).sum())
        report.errors.append(f"'sales' has {n_bad} negative value(s)")
    if "holiday_flag" in df.columns and not df["holiday_flag"].isin([0, 1]).all():
        n_bad = int((~df["holiday_flag"].isin([0, 1])).sum())
        report.errors.append(f"'holiday_flag' has {n_bad} value(s) outside {{0, 1}}")
    if "unemployment" in df.columns and (df["unemployment"] < 0).any():
        n_bad = int((df["unemployment"] < 0).sum())
        report.errors.append(f"'unemployment' has {n_bad} negative value(s)")
    if "fuel_price" in df.columns and (df["fuel_price"] <= 0).any():
        n_bad = int((df["fuel_price"] <= 0).sum())
        report.errors.append(f"'fuel_price' has {n_bad} non-positive value(s)")
    if "cpi" in df.columns and (df["cpi"] <= 0).any():
        n_bad = int((df["cpi"] <= 0).sum())
        report.errors.append(f"'cpi' has {n_bad} non-positive value(s)")
    if "store_nbr" in df.columns and (df["store_nbr"] < 1).any():
        n_bad = int((df["store_nbr"] < 1).sum())
        report.errors.append(f"'store_nbr' has {n_bad} value(s) below 1")
    # Temperature (Fahrenheit) can legitimately be negative in winter — no
    # lower-bound check — but flag physically implausible extremes.
    if "temperature" in df.columns:
        implausible = ((df["temperature"] < -30) | (df["temperature"] > 130)).sum()
        if implausible:
            report.warnings.append(f"'temperature' has {implausible} implausible value(s)")


def validate_duplicates(df: pd.DataFrame, key_cols: list[str], report: ValidationReport) -> None:
    n_dupes = int(df.duplicated(subset=key_cols).sum())
    if n_dupes > 0:
        report.errors.append(f"{n_dupes} duplicate row(s) on key {key_cols}")


def validate_panel_balance(df: pd.DataFrame, report: ValidationReport) -> None:
    """Warn (not fail) if stores don't all share the same set of weeks —
    the source dataset is a fully balanced panel (45 stores x 143 weeks
    each); a gap usually signals an upstream extraction problem."""
    counts = df.groupby("store_nbr")["date"].nunique()
    if counts.nunique() > 1:
        report.warnings.append(
            f"Uneven number of weeks per store (min={counts.min()}, max={counts.max()}); "
            "expected a balanced panel."
        )


def validate_sales_table(df: pd.DataFrame) -> ValidationReport:
    """Run the full data-quality suite on the Walmart weekly sales table."""
    report = ValidationReport()
    validate_schema(df, REQUIRED_COLUMNS, report)
    validate_nulls(df, REQUIRED_COLUMNS, report)
    validate_ranges(df, report)
    validate_duplicates(df, ["date", "store_nbr"], report)
    validate_panel_balance(df, report)
    return report

"""Lightweight data-quality validation (schema, nulls, ranges, referential integrity).

Deliberately dependency-free (plain pandas) to keep the pipeline lean — no extra
framework such as Great Expectations is required to express these checks clearly.
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


REQUIRED_SALES_COLUMNS = {
    "date": "datetime64[ns]",
    "store_nbr": "int",
    "family": "object",
    "sales": "float",
    "onpromotion": "int",
}


def validate_schema(df: pd.DataFrame, required: dict[str, str], report: ValidationReport) -> None:
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
    if "onpromotion" in df.columns and (df["onpromotion"] < 0).any():
        n_bad = int((df["onpromotion"] < 0).sum())
        report.errors.append(f"'onpromotion' has {n_bad} negative value(s)")
    if "oil_price" in df.columns:
        out_of_range = df["oil_price"].dropna().pipe(lambda s: ((s < 0) | (s > 300)).sum())
        if out_of_range:
            report.warnings.append(f"'oil_price' has {out_of_range} implausible value(s)")


def validate_referential_integrity(
    df: pd.DataFrame, stores: pd.DataFrame, report: ValidationReport
) -> None:
    unknown_stores = set(df["store_nbr"].unique()) - set(stores["store_nbr"].unique())
    if unknown_stores:
        report.errors.append(f"store_nbr values not found in stores.csv: {sorted(unknown_stores)}")


def validate_duplicates(df: pd.DataFrame, key_cols: list[str], report: ValidationReport) -> None:
    n_dupes = int(df.duplicated(subset=key_cols).sum())
    if n_dupes > 0:
        report.errors.append(f"{n_dupes} duplicate row(s) on key {key_cols}")


def validate_sales_table(df: pd.DataFrame, stores: pd.DataFrame) -> ValidationReport:
    """Run the full data-quality suite on a merged train/test sales table."""
    report = ValidationReport()
    required = {k: v for k, v in REQUIRED_SALES_COLUMNS.items() if k in df.columns or k != "sales"}
    validate_schema(df, required, report)
    validate_nulls(df, ["date", "store_nbr", "family"], report)
    validate_ranges(df, report)
    validate_referential_integrity(df, stores, report)
    validate_duplicates(df, ["date", "store_nbr", "family"], report)
    return report

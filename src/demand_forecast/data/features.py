"""Feature engineering for the Walmart weekly demand forecasting model.

All features are computed causally (only using information available at or
before prediction time) so the same function can be applied to train and
inference data without leakage. The source data is a weekly, Friday-anchored
panel (45 stores x 143 weeks, no gaps) — day-of-week/day-of-month features
from the original daily-grain design are meaningless here (every row is a
Friday) and have been dropped accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["store_nbr"]
NUMERIC_FEATURES = [
    "holiday_flag",
    "temperature",
    "fuel_price",
    "cpi",
    "unemployment",
    "month",
    "week_of_year",
    "year",
    "sales_lag_1",
    "sales_lag_4",
    "sales_lag_52",
    "sales_roll_mean_4",
    "sales_roll_mean_8",
    "sales_roll_std_4",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "sales"
LOG_TARGET = "log_sales"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    return df


def add_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = (1, 4, 52)) -> pd.DataFrame:
    """Add per-store lagged sales. The panel is one row per store per week
    with no gaps, so an N-row shift is exactly an N-week lag.

    lag_1  = last week (short-term momentum)
    lag_4  = ~1 month ago
    lag_52 = same week last year (holiday/seasonal year-over-year signal;
             only ~2.7 years of history exist, so this is NaN for each
             store's first 52 weeks — LightGBM handles that natively).
    """
    df = df.sort_values(["store_nbr", "date"]).copy()
    group = df.groupby("store_nbr", observed=True)["sales"]
    for lag in lags:
        df[f"sales_lag_{lag}"] = group.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: tuple[int, ...] = (4, 8)) -> pd.DataFrame:
    """Rolling mean/std computed on the lag-1 series so the window never
    touches the target week itself (avoids leakage while remaining simple)."""
    df = df.sort_values(["store_nbr", "date"]).copy()
    if "sales_lag_1" not in df.columns:
        df = add_lag_features(df, lags=(1,))
    base = df.groupby("store_nbr", observed=True)["sales_lag_1"]
    for window in windows:
        df[f"sales_roll_mean_{window}"] = base.transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )
    df["sales_roll_std_4"] = base.transform(lambda s: s.rolling(4, min_periods=2).std())
    return df


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """The n-th occurrence of `weekday` (Mon=0..Sun=6) in `year`-`month`."""
    first = pd.Timestamp(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset) + pd.Timedelta(weeks=n - 1)


def _week_friday(anchor: pd.Timestamp) -> pd.Timestamp:
    """Friday of the Sat-Fri week containing `anchor` (matches this dataset's
    week-ending-Friday convention)."""
    days_since_saturday = (anchor.weekday() - 5) % 7
    saturday = anchor - pd.Timedelta(days=days_since_saturday)
    return saturday + pd.Timedelta(days=6)


def is_major_holiday_week(date: pd.Timestamp) -> int:
    """Rule-based approximation of `holiday_flag` for dates beyond the
    training data's range (2010-02-05 .. 2012-10-26), using each holiday's
    standard US scheduling rule rather than a hardcoded per-year table.

    Verified to reproduce all 10 `holiday_flag=1` Fridays actually present
    in the source dataset (Super Bowl / Labor Day / Thanksgiving / Christmas
    weeks within its 2010-02-05..2012-10-26 range) — see
    tests/unit/test_features.py.
    """
    date = pd.Timestamp(date)
    year = date.year
    anchors = [
        _nth_weekday(year, 2, 6, 1),  # Super Bowl: 1st Sunday of February
        _nth_weekday(year, 9, 0, 1),  # Labor Day: 1st Monday of September
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving: 4th Thursday of November
        pd.Timestamp(year, 12, 25),  # Christmas
    ]
    holiday_fridays = {_week_friday(a) for a in anchors}
    return int(_week_friday(date) in holiday_fridays)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline used identically for training and inference batches."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if TARGET in df.columns:
        df[LOG_TARGET] = np.log1p(df[TARGET].clip(lower=0))

    return df


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns the model consumes (features present in df)."""
    cols = [c for c in ALL_FEATURES if c in df.columns]
    return df[cols]

"""Feature engineering for the demand forecasting model.

All features are computed causally (only using information available at or
before prediction time) so the same function can be applied to train and
inference data without leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
NUMERIC_FEATURES = [
    "onpromotion",
    "oil_price",
    "is_holiday",
    "is_event",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll_mean_7",
    "sales_roll_mean_28",
    "sales_roll_std_7",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "sales"
LOG_TARGET = "log_sales"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = (7, 14, 28)) -> pd.DataFrame:
    """Add per (store, family) lagged sales — requires df sorted by store/family/date
    and a 'sales' column that is NaN for rows to forecast (test rows)."""
    df = df.sort_values(["store_nbr", "family", "date"]).copy()
    group = df.groupby(["store_nbr", "family"], observed=True)["sales"]
    for lag in lags:
        df[f"sales_lag_{lag}"] = group.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: tuple[int, ...] = (7, 28)) -> pd.DataFrame:
    """Rolling mean/std computed on the lag-7 series so windows never touch the
    target day itself (avoids leakage while remaining simple)."""
    df = df.sort_values(["store_nbr", "family", "date"]).copy()
    if "sales_lag_7" not in df.columns:
        df = add_lag_features(df, lags=(7,))
    base = df.groupby(["store_nbr", "family"], observed=True)["sales_lag_7"]
    for window in windows:
        df[f"sales_roll_mean_{window}"] = base.transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )
    df["sales_roll_std_7"] = base.transform(lambda s: s.rolling(7, min_periods=2).std())
    return df


def rename_store_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"type": "store_type"})


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline used identically for training and inference batches."""
    df = rename_store_columns(df)
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

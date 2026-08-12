"""Reference feature snapshot used at serving time.

The API is a lightweight service — it should not need to recompute
historical lag/rolling aggregates on every request. Instead, at the end of
training we persist, per store, the most recent known lag/rolling sales
statistics and economic indicators (temperature, fuel price, CPI,
unemployment). At inference time these are combined with calendar/holiday
features computed fresh for the requested date, with an option to override
any of the economic indicators if the caller has an actual forecast for them.

This is a deliberate, documented simplification for course scope (see
ARCHITECTURE.md trade-offs): a production system would instead maintain a
real-time feature store / pull live economic indicator forecasts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_COLUMNS = [
    "store_nbr",
    "temperature",
    "fuel_price",
    "cpi",
    "unemployment",
    "sales_lag_1",
    "sales_lag_4",
    "sales_lag_52",
    "sales_roll_mean_4",
    "sales_roll_mean_8",
    "sales_roll_std_4",
]


def build_reference_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Take the last known row per store from the feature-engineered
    training frame — i.e. the freshest lag/rolling context available at
    serving time."""
    latest = df.sort_values("date").groupby("store_nbr", observed=True).tail(1)
    return latest[SNAPSHOT_COLUMNS].reset_index(drop=True)


def save_reference_artifacts(train_df: pd.DataFrame, processed_dir: Path) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    build_reference_snapshot(train_df).to_csv(processed_dir / "reference_snapshot.csv", index=False)


def load_reference_artifacts(processed_dir: Path) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    snapshot_path = processed_dir / "reference_snapshot.csv"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Reference snapshot not found in {processed_dir}. Run the training "
            "pipeline first (python scripts/run_pipeline.py)."
        )
    return pd.read_csv(snapshot_path)

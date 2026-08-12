"""Reference feature snapshot used at serving time.

The API is a lightweight service — it should not need to load the full
116MB train.csv or recompute historical lag/rolling aggregates on every
request. Instead, at the end of training we persist, per (store_nbr, family),
the most recent known lag/rolling sales statistics and static store
attributes. At inference time these are combined with calendar/holiday/oil
features computed fresh for the requested date.

This is a deliberate, documented simplification for course scope (see
ARCHITECTURE.md trade-offs): a production system would instead maintain a
real-time feature store that refreshes lag values as new sales land.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_COLUMNS = [
    "store_nbr",
    "family",
    "city",
    "state",
    "store_type",
    "cluster",
    "oil_price",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_roll_mean_7",
    "sales_roll_mean_28",
    "sales_roll_std_7",
]


def build_reference_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Take the last known row per (store_nbr, family) from the feature-engineered
    training frame — i.e. the freshest lag/rolling context available at serving time."""
    latest = df.sort_values("date").groupby(["store_nbr", "family"], observed=True).tail(1)
    return latest[SNAPSHOT_COLUMNS].reset_index(drop=True)


def save_reference_artifacts(
    train_df: pd.DataFrame,
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
    processed_dir: Path,
) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    build_reference_snapshot(train_df).to_csv(processed_dir / "reference_snapshot.csv", index=False)
    stores.to_csv(processed_dir / "stores.csv", index=False)
    holidays.to_csv(processed_dir / "holidays_events.csv", index=False)


def load_reference_artifacts(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    snapshot_path = processed_dir / "reference_snapshot.csv"
    holidays_path = processed_dir / "holidays_events.csv"
    if not snapshot_path.exists() or not holidays_path.exists():
        raise FileNotFoundError(
            f"Reference artifacts not found in {processed_dir}. Run the training "
            "pipeline first (python scripts/run_pipeline.py)."
        )
    snapshot = pd.read_csv(snapshot_path)
    holidays = pd.read_csv(holidays_path, parse_dates=["date"])
    return snapshot, holidays

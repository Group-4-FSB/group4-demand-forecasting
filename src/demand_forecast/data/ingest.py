"""Data ingestion: load the raw Kaggle CSVs and merge them into one modeling table.

Source tables (Corporación Favorita "Store Sales - Time Series Forecasting"):
- train.csv / test.csv : date, store_nbr, family, sales, onpromotion
- stores.csv           : store_nbr, city, state, type, cluster
- oil.csv              : date, dcoilwtico (WTI oil price — proxy for Ecuador's economy)
- holidays_events.csv  : date, type, locale, locale_name, description, transferred
- transactions.csv     : date, store_nbr, transactions
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
}

# Only these tables have a 'date' column to parse; stores.csv is static store metadata.
_TABLES_WITH_DATE = {"train", "test", "oil", "holidays", "transactions"}


def load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every raw CSV into a dict of DataFrames, parsing date columns where present."""
    raw_dir = Path(raw_dir)
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw file {path}. Run `python scripts/setup_data.py` first."
            )
        parse_dates = ["date"] if key in _TABLES_WITH_DATE else None
        df = pd.read_csv(path, parse_dates=parse_dates)
        tables[key] = df
        logger.info("Loaded %s: %s rows, %s cols", filename, len(df), df.shape[1])
    return tables


def build_holiday_flags(holidays: pd.DataFrame) -> pd.DataFrame:
    """Collapse holidays_events into one row per date: is_holiday / is_special_event.

    Transferred holidays are treated as regular working days (per dataset docs),
    and the day they were transferred *to* is treated as the holiday instead.
    """
    h = holidays.copy()
    h = h[~((h["type"] == "Holiday") & (h["transferred"]))]
    h["is_holiday"] = h["type"].isin(["Holiday", "Transfer", "Bridge", "Additional"]).astype(int)
    h["is_event"] = (h["type"] == "Event").astype(int)
    daily = h.groupby("date", as_index=False).agg(
        is_holiday=("is_holiday", "max"), is_event=("is_event", "max")
    )
    return daily


def merge_dataset(tables: dict[str, pd.DataFrame], split: str = "train") -> pd.DataFrame:
    """Merge the requested split (train/test) with stores, oil, holidays, transactions."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    df = tables[split].copy()
    df = df.merge(tables["stores"], on="store_nbr", how="left", validate="many_to_one")

    oil = tables["oil"].rename(columns={"dcoilwtico": "oil_price"}).sort_values("date")
    oil["oil_price"] = oil["oil_price"].ffill().bfill()
    df = df.merge(oil, on="date", how="left", validate="many_to_one")

    holiday_daily = build_holiday_flags(tables["holidays"])
    df = df.merge(holiday_daily, on="date", how="left", validate="many_to_one")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df["is_event"] = df["is_event"].fillna(0).astype(int)

    df = df.merge(
        tables["transactions"], on=["date", "store_nbr"], how="left", validate="many_to_one"
    )

    return df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

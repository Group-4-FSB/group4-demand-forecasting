"""Data ingestion: load the raw Walmart weekly sales CSV and normalize it.

Source: Kaggle "Walmart Sales" (https://www.kaggle.com/datasets/mikhail1681/walmart-sales)
— a single file, one row per (store, week):
    Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment

45 stores x 143 weeks (2010-02-05 .. 2012-10-26), every date a Friday, no
gaps, no nulls. There is no separate store-metadata/holiday-calendar/
promotions file like some other retail datasets — CPI and Unemployment
already vary per store and stand in as a regional-economics proxy (see
ARCHITECTURE.md and docs/RESPONSIBLE_AI.md for how that's used).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

RAW_FILENAME = "Walmart_Sales.csv"

RAW_COLUMN_RENAME = {
    "Store": "store_nbr",
    "Date": "date",
    "Weekly_Sales": "sales",
    "Holiday_Flag": "holiday_flag",
    "Temperature": "temperature",
    "Fuel_Price": "fuel_price",
    "CPI": "cpi",
    "Unemployment": "unemployment",
}


def load_walmart_sales(raw_dir: Path) -> pd.DataFrame:
    """Load, rename, and type the raw Walmart weekly sales table."""
    raw_dir = Path(raw_dir)
    path = raw_dir / RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw file {path}. Run `python scripts/setup_data.py` first."
        )

    df = pd.read_csv(path)
    df = df.rename(columns=RAW_COLUMN_RENAME)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["holiday_flag"] = df["holiday_flag"].astype(int)
    df = df.sort_values(["store_nbr", "date"]).reset_index(drop=True)

    logger.info(
        "Loaded %s: %d rows, %d stores, %s to %s",
        RAW_FILENAME,
        len(df),
        df["store_nbr"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def compute_raw_data_fingerprint(raw_dir: Path, df: pd.DataFrame) -> dict[str, Any]:
    """A data-version fingerprint for the raw source file, meant to be logged
    onto every training run (see `models/train.py`).

    A run's own MLflow `start_time` only answers "when did training happen" —
    it says nothing about which content of `data/raw/Walmart_Sales.csv`
    produced the model, since that file has no version control of its own
    (gitignored — see README). SHA256 pins the exact byte content (independent
    of any ingest-time parsing/renaming, which is a *code* concern, not a
    *data* one); the row/date-range stats answer the separate, and often more
    useful, "what week is this model's data current as of" question.
    """
    path = Path(raw_dir) / RAW_FILENAME
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "raw_data_sha256": digest,
        "raw_data_rows": len(df),
        "raw_data_min_date": str(df["date"].min().date()),
        "raw_data_max_date": str(df["date"].max().date()),
        "raw_data_n_stores": int(df["store_nbr"].nunique()),
    }

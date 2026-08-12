"""One-off dev script: build small, deterministic CSV fixtures under tests/fixtures/
from the full raw dataset, for fast unit/integration/CI tests. Not part of the
runtime pipeline — run manually only if fixtures need to be regenerated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIXTURES = ROOT / "tests" / "fixtures"

STORES = [1, 2, 3]
FAMILIES = ["GROCERY I", "BEVERAGES", "DAIRY"]
DATE_START = "2017-06-01"
DATE_END = "2017-08-15"  # train.csv ends 2017-08-15; test.csv starts the next day


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW / "train.csv", parse_dates=["date"])
    mask = (
        train["store_nbr"].isin(STORES)
        & train["family"].isin(FAMILIES)
        & train["date"].between(DATE_START, DATE_END)
    )
    train_fixture = train.loc[mask].reset_index(drop=True)
    train_fixture.to_csv(FIXTURES / "train.csv", index=False)

    test = pd.read_csv(RAW / "test.csv", parse_dates=["date"])
    test_mask = test["store_nbr"].isin(STORES) & test["family"].isin(FAMILIES)
    test_fixture = test.loc[test_mask].reset_index(drop=True)
    test_fixture.to_csv(FIXTURES / "test.csv", index=False)

    stores = pd.read_csv(RAW / "stores.csv")
    stores.loc[stores["store_nbr"].isin(STORES)].to_csv(FIXTURES / "stores.csv", index=False)

    oil = pd.read_csv(RAW / "oil.csv", parse_dates=["date"])
    oil.loc[oil["date"].between(DATE_START, DATE_END)].to_csv(FIXTURES / "oil.csv", index=False)

    holidays = pd.read_csv(RAW / "holidays_events.csv", parse_dates=["date"])
    holidays.loc[holidays["date"].between(DATE_START, DATE_END)].to_csv(
        FIXTURES / "holidays_events.csv", index=False
    )

    transactions = pd.read_csv(RAW / "transactions.csv", parse_dates=["date"])
    tx_mask = transactions["store_nbr"].isin(STORES) & transactions["date"].between(
        DATE_START, DATE_END
    )
    transactions.loc[tx_mask].to_csv(FIXTURES / "transactions.csv", index=False)

    print(f"train_fixture rows: {len(train_fixture)}")
    print(f"test_fixture rows: {len(test_fixture)}")
    print(f"Fixtures written to {FIXTURES}")


if __name__ == "__main__":
    main()

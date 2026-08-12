"""One-off dev script: build a small, deterministic CSV fixture under
tests/fixtures/ from the full raw dataset, for fast unit/integration/CI
tests. Not part of the runtime pipeline — run manually only if the fixture
needs to be regenerated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIXTURES = ROOT / "tests" / "fixtures"

STORES = [1, 2, 3]  # a few stores, full history each — enough weeks per
# store for lag_52 to have some non-null coverage while staying small.


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW / "Walmart_Sales.csv")
    fixture = df.loc[df["Store"].isin(STORES)].reset_index(drop=True)
    fixture.to_csv(FIXTURES / "Walmart_Sales.csv", index=False)

    print(f"fixture rows: {len(fixture)} ({len(STORES)} stores)")
    print(f"Fixture written to {FIXTURES / 'Walmart_Sales.csv'}")


if __name__ == "__main__":
    main()

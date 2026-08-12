"""Extract the Kaggle "Store Sales - Time Series Forecasting" dataset into data/raw/.

Usage:
    python scripts/setup_data.py [--zip PATH] [--force]

By default looks for `store-sales-time-series-forecasting.zip` in the project root
(the file provided with the assignment). If you don't have the zip, download it from
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
(requires a free Kaggle account) and pass its path via --zip.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = PROJECT_ROOT / "store-sales-time-series-forecasting.zip"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
    "sample_submission.csv",
]


def extract(zip_path: Path, dest_dir: Path, force: bool = False) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset zip not found at {zip_path}. Pass --zip <path> to point at it."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    already_present = [f for f in EXPECTED_FILES if (dest_dir / f).exists()]
    if already_present == EXPECTED_FILES and not force:
        print(f"All expected files already present in {dest_dir}. Use --force to re-extract.")
        return

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        missing = [f for f in EXPECTED_FILES if f not in members]
        if missing:
            raise ValueError(f"Zip is missing expected files: {missing}")
        zf.extractall(dest_dir)

    print(f"Extracted {len(EXPECTED_FILES)} files to {dest_dir}")
    for f in EXPECTED_FILES:
        size_mb = (dest_dir / f).stat().st_size / (1024 * 1024)
        print(f"  {f:<28} {size_mb:8.2f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="Path to the dataset zip")
    parser.add_argument("--dest", type=Path, default=RAW_DIR, help="Destination directory")
    parser.add_argument("--force", action="store_true", help="Re-extract even if files exist")
    args = parser.parse_args()

    try:
        extract(args.zip, args.dest, force=args.force)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

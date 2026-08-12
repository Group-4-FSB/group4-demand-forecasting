"""Extract the Kaggle "Walmart Sales" dataset into data/raw/.

Usage:
    python scripts/setup_data.py [--zip PATH] [--force]

By default looks for `walmart_sales.zip` in the project root (the file
provided with the assignment). If you don't have it, download it from
https://www.kaggle.com/datasets/mikhail1681/walmart-sales
(requires a free Kaggle account) and pass its path via --zip.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = PROJECT_ROOT / "walmart_sales.zip"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILE = "Walmart_Sales.csv"


def extract(zip_path: Path, dest_dir: Path, force: bool = False) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset zip not found at {zip_path}. Pass --zip <path> to point at it."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    if (dest_dir / EXPECTED_FILE).exists() and not force:
        print(f"{EXPECTED_FILE} already present in {dest_dir}. Use --force to re-extract.")
        return

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if EXPECTED_FILE not in members:
            raise ValueError(f"Zip is missing expected file: {EXPECTED_FILE} (found: {members})")
        zf.extractall(dest_dir)

    size_mb = (dest_dir / EXPECTED_FILE).stat().st_size / (1024 * 1024)
    print(f"Extracted {EXPECTED_FILE} ({size_mb:.2f} MB) to {dest_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="Path to the dataset zip")
    parser.add_argument("--dest", type=Path, default=RAW_DIR, help="Destination directory")
    parser.add_argument("--force", action="store_true", help="Re-extract even if the file exists")
    args = parser.parse_args()

    try:
        extract(args.zip, args.dest, force=args.force)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

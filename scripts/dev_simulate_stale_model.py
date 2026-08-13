"""DEV/DEMO-ONLY: back-date the `production` model version's registration
timestamp directly in MLflow's SQLite backend store, so you can watch
`demand_forecast_production_model_age_days` and the
`ProductionModelApproachingStaleness` / `ProductionModelStale` Prometheus
alerts (see monitoring/prometheus/alert_rules.yml) actually cross their
thresholds in seconds instead of waiting 5-7 real days.

This is NOT part of the production pipeline and is NOT imported by anything
else in the codebase. It pokes MLflow's internal `model_versions.creation_time`
column directly (there is no public MLflow API to set this — it's normally
assigned once, automatically, at registration). Only works against a
sqlite:/// backend, which both the local dev `mlflow.db` and the Dockerized
MLflow use.

Standalone on purpose (only stdlib `sqlite3`, no `demand_forecast` import):
the Dockerized MLflow's own image never has the `demand_forecast` package
installed (see Dockerfile.mlflow), so this needs to run inside that
container using only what it already has.

Usage — Docker (the common case; run *inside* the mlflow container, which
has the sqlite file + python but not curl/sqlite3-cli):
    docker cp scripts/dev_simulate_stale_model.py demand-forecast-mlflow:/tmp/sim.py
    docker exec demand-forecast-mlflow python /tmp/sim.py --age-days 6
    docker compose restart api   # API only re-reads the timestamp at startup
    # wait ~15-30s for Prometheus's next scrape, then check:
    #   http://localhost:9090/alerts  (ProductionModelApproachingStaleness)
    #   http://localhost:3000 -> "Demand Forecast API" -> "Production model age (days)"

Usage — local dev (mlflow.db in the project root):
    python scripts/dev_simulate_stale_model.py --db-path mlflow.db --age-days 6
    # then restart your local `uvicorn ... --reload` process

To reset: either retrain (`python scripts/run_pipeline.py`, which registers
a brand-new version with a real, current timestamp and makes the backdated
one irrelevant), or re-run this script with `--age-days 0` to make the same
version look freshly registered again. The script also prints the previous
raw `creation_time` (ms since epoch) before overwriting it, in case you want
to restore that exact value by hand.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time


def backdate(db_path: str, model_name: str, alias: str, age_days: float) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT version FROM registered_model_aliases WHERE name = ? AND alias = ?",
            (model_name, alias),
        )
        row = cur.fetchone()
        if row is None:
            print(f"No version aliased '{alias}' for model '{model_name}' in {db_path}.")
            sys.exit(1)
        version = row[0]

        cur.execute(
            "SELECT creation_time FROM model_versions WHERE name = ? AND version = ?",
            (model_name, version),
        )
        (old_creation_time,) = cur.fetchone()

        new_creation_time = int(time.time() * 1000) - int(age_days * 86400 * 1000)
        cur.execute(
            "UPDATE model_versions SET creation_time = ? WHERE name = ? AND version = ?",
            (new_creation_time, model_name, version),
        )
        conn.commit()
        print(
            f"{model_name}@{alias} (version {version}) now looks {age_days:g} day(s) old.\n"
            f"  creation_time: {old_creation_time} -> {new_creation_time}  (ms since epoch; "
            f"save the old value if you want to restore it exactly by hand)"
        )
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db-path",
        default="/mlflow/mlflow.db",
        help="Path to MLflow's sqlite backend store (default: the mlflow container's own path)",
    )
    parser.add_argument("--model-name", default="demand-forecast-lgbm")
    parser.add_argument("--alias", default="production")
    parser.add_argument(
        "--age-days",
        type=float,
        required=True,
        help="How old the model should now appear to be, in days (0 = looks freshly registered)",
    )
    args = parser.parse_args()

    backdate(args.db_path, args.model_name, args.alias, args.age_days)
    print("Now restart the API (docker compose restart api) so it re-reads this timestamp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

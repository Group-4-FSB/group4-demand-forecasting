"""Automatic weekly-retrain trigger.

Business rule: the `production` model must never go more than
`--max-age-days` (default 7, `RETRAIN_MAX_AGE_DAYS`) without an attempted
refresh. This script checks how long ago the model version currently
aliased `production` was registered and, if it's older than the threshold
(or nothing is registered yet), runs the full training pipeline. Starting
`--warning-lead-days` (default 2, `RETRAIN_WARNING_LEAD_DAYS`) before that —
i.e. from day 5 onward for the 7-day default — every check logs a WARNING
("~X day(s) left") instead of the usual INFO line, so staleness is visible
well before it actually happens, not just at the moment it does. The same
signal is also exposed to Prometheus as the
`demand_forecast_production_model_age_days` gauge (see `api/metrics.py`),
with matching `ProductionModelApproachingStaleness` /
`ProductionModelStale` alert rules in `monitoring/prometheus/alert_rules.yml`.

Data freshness is someone else's job: this script always reads whatever is
currently sitting in `data/raw/` — keeping that file up to date is an
external ML-engineering data pipeline's responsibility, out of scope here.

The training pipeline's own quality gate (see `models/train.py`) still
decides whether the resulting candidate actually replaces production — this
script only decides *when* to attempt a retrain, never whether to promote
the result. A rejected candidate still counts as "we tried"; the age clock
only resets when a model is actually promoted (an unchanged `production`
means an unchanged registration timestamp, so the very next check will just
try again — and the warning keeps firing until one finally succeeds).

Usage:
    python scripts/retrain_if_stale.py                          # check once, exit
    python scripts/retrain_if_stale.py --max-age-days 3          # custom threshold
    python scripts/retrain_if_stale.py --warning-lead-days 1     # warn from day 6 instead of 5
    python scripts/retrain_if_stale.py --loop                    # check forever
    python scripts/retrain_if_stale.py --loop --check-interval-hours 6
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts/ imports

import mlflow  # noqa: E402
from demand_forecast.config import settings  # noqa: E402
from demand_forecast.models.train import get_current_production_version  # noqa: E402
from run_pipeline import run_training_pipeline  # noqa: E402

logger = logging.getLogger(__name__)


def get_production_model_age(
    client: mlflow.MlflowClient, model_name: str, alias: str
) -> timedelta | None:
    """How long ago the model version currently aliased `alias` was
    registered. None if no such model/alias exists yet (always due)."""
    mv = get_current_production_version(client, model_name, alias)
    if mv is None:
        return None
    created = datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc)
    return datetime.now(tz=timezone.utc) - created


def is_retrain_due(age: timedelta | None, max_age_days: int) -> bool:
    """No production model at all -> always due. Otherwise due once its age
    reaches the limit (>=, not >, so exactly-N-days-old triggers today)."""
    return age is None or age >= timedelta(days=max_age_days)


def is_approaching_staleness(
    age: timedelta | None, max_age_days: int, warning_lead_days: int
) -> bool:
    """True once the model is within `warning_lead_days` of going stale, but
    not stale yet (that case is `is_retrain_due`, not this). No production
    model at all is already "due", not "approaching" — never true for None."""
    if age is None or is_retrain_due(age, max_age_days):
        return False
    return age >= timedelta(days=max_age_days - warning_lead_days)


def check_and_retrain(max_age_days: int, warning_lead_days: int | None = None) -> bool:
    """Run one staleness check. Returns True if a retrain was attempted
    (regardless of whether the quality gate then promoted it)."""
    if warning_lead_days is None:
        warning_lead_days = settings.retrain_warning_lead_days
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    age = get_production_model_age(client, settings.mlflow_model_name, settings.mlflow_model_alias)

    if not is_retrain_due(age, max_age_days):
        remaining = timedelta(days=max_age_days) - age  # type: ignore[operator]
        remaining_days = remaining.total_seconds() / 86400
        age_days = age.total_seconds() / 86400  # type: ignore[union-attr]
        if is_approaching_staleness(age, max_age_days, warning_lead_days):
            # Loud on purpose: this is the "still X day(s) left" heads-up the
            # user asked for, so it's visible in `docker logs` (or a Prometheus
            # alert — see demand_forecast_production_model_age_days /
            # ProductionModelApproachingStaleness) well before the model
            # actually goes stale, not just at the moment it does.
            logger.warning(
                "Production model is %.1f day(s) old (limit %d) — approaching "
                "staleness: only ~%.1f day(s) left before an automatic retrain "
                "is triggered.",
                age_days,
                max_age_days,
                remaining_days,
            )
        else:
            logger.info(
                "Production model is %.1f day(s) old (limit %d) — next check-in due in "
                "~%.1f day(s). Nothing to do.",
                age_days,
                max_age_days,
                remaining_days,
            )
        return False

    reason = (
        "no production model is registered yet"
        if age is None
        else f"production model is {age.total_seconds() / 86400:.1f} day(s) old "
        f"(limit {max_age_days})"
    )
    logger.info(
        "Retrain due: %s. Running the training pipeline against %s ...",
        reason,
        settings.data_raw_dir,
    )
    summary = run_training_pipeline()

    if summary["promoted"]:
        logger.info(
            "Scheduled retrain PROMOTED a new production model (test_rmsle=%.4f).",
            summary["test_metrics"]["rmsle"],
        )
    else:
        logger.warning(
            "Scheduled retrain ran but did NOT promote — new model failed the quality "
            "gate (test_rmsle=%.4f vs current production %.4f). Production is unchanged; "
            "will try again at the next check.",
            summary["test_metrics"]["rmsle"],
            summary["current_production_test_rmsle"],
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--max-age-days", type=int, default=settings.retrain_max_age_days)
    parser.add_argument(
        "--warning-lead-days",
        type=int,
        default=settings.retrain_warning_lead_days,
        help="Log a heads-up WARNING once the model is within this many days of staleness.",
    )
    parser.add_argument("--loop", action="store_true", help="Run forever, checking periodically")
    parser.add_argument(
        "--check-interval-hours",
        type=float,
        default=24.0,
        help="Only used with --loop. How often to check staleness (not how often to retrain).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if not args.loop:
        check_and_retrain(args.max_age_days, args.warning_lead_days)
        return 0

    logger.info(
        "Retrain scheduler started: checking every %.1fh, %d-day staleness limit "
        "(warns from %d day(s) before).",
        args.check_interval_hours,
        args.max_age_days,
        args.warning_lead_days,
    )
    while True:
        try:
            check_and_retrain(args.max_age_days, args.warning_lead_days)
        except Exception:
            logger.exception("Staleness check failed — will retry at the next interval.")
        time.sleep(args.check_interval_hours * 3600)


if __name__ == "__main__":
    raise SystemExit(main())

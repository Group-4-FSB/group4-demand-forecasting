"""Custom Prometheus metrics for the demand forecasting service.

`prometheus-fastapi-instrumentator` already covers generic HTTP metrics
(request count, latency, in-progress requests) — these are the ML-specific
metrics on top of that: how many predictions were made, how long the model
itself took, what values it's returning, and which model version is loaded.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS_TOTAL = Counter(
    "demand_forecast_predictions_total",
    "Total number of successful predictions served",
    ["store_nbr"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "demand_forecast_prediction_errors_total",
    "Total number of prediction requests that failed",
    ["reason"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "demand_forecast_prediction_latency_seconds",
    "Time spent computing a single model prediction (excludes HTTP overhead)",
)

# Buckets sized for Walmart weekly store sales (observed range ~$260K-$2.1M).
PREDICTED_SALES_VALUE = Histogram(
    "demand_forecast_predicted_sales",
    "Distribution of predicted weekly sales values returned by the model (USD)",
    buckets=(
        0,
        100_000,
        250_000,
        500_000,
        750_000,
        1_000_000,
        1_250_000,
        1_500_000,
        1_750_000,
        2_000_000,
        2_500_000,
        3_000_000,
    ),
)

MODEL_INFO = Gauge(
    "demand_forecast_model_info",
    "Static info about the currently loaded model (value is always 1)",
    ["model_name", "model_alias"],
)

MODEL_LOADED = Gauge(
    "demand_forecast_model_loaded",
    "1 if a production model is currently loaded and serving, 0 if /health "
    "is degraded (e.g. nothing registered yet, or the registry was unreachable "
    "at startup)",
)

# Evaluated lazily at scrape time via .set_function() in api/main.py, so the
# value is always current without polling MLflow on every 15s scrape — see
# the ProductionModelApproachingStaleness / ProductionModelStale alerts in
# monitoring/prometheus/alert_rules.yml, and the day-5/7 warning this backs
# in scripts/retrain_if_stale.py.
PRODUCTION_MODEL_AGE_DAYS = Gauge(
    "demand_forecast_production_model_age_days",
    "Age in days of the model version currently aliased 'production' in "
    "MLflow. NaN if no production model is registered (nothing to compare "
    "against, so the retrain scheduler always promotes the next candidate).",
)

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
    ["family"],
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

PREDICTED_SALES_VALUE = Histogram(
    "demand_forecast_predicted_sales",
    "Distribution of predicted sales values returned by the model",
    buckets=(0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)

MODEL_INFO = Gauge(
    "demand_forecast_model_info",
    "Static info about the currently loaded model (value is always 1)",
    ["model_name", "model_alias"],
)

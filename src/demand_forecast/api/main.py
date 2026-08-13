"""FastAPI application for serving demand forecasts.

Endpoints:
    GET  /health                    liveness/readiness probe
    POST /api/v1/predict            single forecast
    POST /api/v1/predict/batch      batch forecast (up to 500 items)
    GET  /metrics                   Prometheus scrape endpoint (auto + custom)
    GET  /docs                      Swagger UI (auto-generated OpenAPI)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from demand_forecast import __version__
from demand_forecast.api.metrics import (
    MODEL_INFO,
    MODEL_LOADED,
    PREDICTED_SALES_VALUE,
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY_SECONDS,
    PREDICTIONS_TOTAL,
    PRODUCTION_MODEL_AGE_DAYS,
)
from demand_forecast.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from demand_forecast.config import settings
from demand_forecast.models.predict import PredictionService, UnknownStoreError
from demand_forecast.models.train import get_current_production_version

logger = logging.getLogger(__name__)


def _production_model_age_days() -> float:
    """Callback evaluated at Prometheus scrape time (see .set_function()
    below) — pure local arithmetic against the timestamp cached at startup,
    so a scrape never blocks on an MLflow round trip."""
    created_at = getattr(app.state, "production_model_created_at", None)
    if created_at is None:
        return float("nan")
    return (datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.production_model_created_at = None
    try:
        app.state.prediction_service = PredictionService.load()
        MODEL_INFO.labels(
            model_name=settings.mlflow_model_name, model_alias=settings.mlflow_model_alias
        ).set(1)
        MODEL_LOADED.set(1)
        logger.info("Model loaded: %s@%s", settings.mlflow_model_name, settings.mlflow_model_alias)

        try:
            client = mlflow.MlflowClient()
            mv = get_current_production_version(
                client, settings.mlflow_model_name, settings.mlflow_model_alias
            )
            if mv is not None:
                app.state.production_model_created_at = datetime.fromtimestamp(
                    mv.creation_timestamp / 1000, tz=timezone.utc
                )
        except Exception:
            # Non-fatal: the model is already loaded and serving fine, this
            # only affects the age gauge / staleness alerts.
            logger.exception("Failed to fetch production model registration time")
    except Exception:
        # Service starts even if the model isn't trained/registered yet so
        # /health can report the problem instead of the container crash-looping.
        app.state.prediction_service = None
        MODEL_LOADED.set(0)
        logger.exception("Failed to load prediction model at startup")
    yield


app = FastAPI(
    title="Demand Forecast API",
    description="Predicts weekly sales per Walmart store — DDM501 final project.",
    version=__version__,
    lifespan=lifespan,
)

PRODUCTION_MODEL_AGE_DAYS.set_function(_production_model_age_days)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def _service(request: Request) -> PredictionService:
    service = request.app.state.prediction_service
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train and register a model, then restart the service.",
        )
    return service


def _predict_one(service: PredictionService, item: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    try:
        value = service.predict(
            item.store_nbr,
            item.date,
            item.holiday_flag,
            item.temperature,
            item.fuel_price,
            item.cpi,
            item.unemployment,
        )
    except UnknownStoreError as exc:
        PREDICTION_ERRORS_TOTAL.labels(reason="unknown_store").inc()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    PREDICTION_LATENCY_SECONDS.observe(time.perf_counter() - start)
    PREDICTIONS_TOTAL.labels(store_nbr=str(item.store_nbr)).inc()
    PREDICTED_SALES_VALUE.observe(value)
    return PredictResponse(
        store_nbr=item.store_nbr,
        date=item.date,
        predicted_sales=round(value, 2),
        model_name=settings.mlflow_model_name,
        model_alias=settings.mlflow_model_alias,
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    service = request.app.state.prediction_service
    return HealthResponse(
        status="ok" if service is not None else "degraded",
        model_loaded=service is not None,
        model_name=settings.mlflow_model_name if service is not None else None,
        model_alias=settings.mlflow_model_alias if service is not None else None,
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["forecast"])
async def predict(item: PredictRequest, request: Request) -> PredictResponse:
    service = _service(request)
    return _predict_one(service, item)


@app.post("/api/v1/predict/batch", response_model=BatchPredictResponse, tags=["forecast"])
async def predict_batch(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    service = _service(request)
    predictions = [_predict_one(service, item) for item in payload.items]
    return BatchPredictResponse(predictions=predictions)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

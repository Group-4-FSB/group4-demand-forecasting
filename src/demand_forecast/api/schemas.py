"""Pydantic request/response schemas for the demand forecasting API."""

from __future__ import annotations

from datetime import date as date_type

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    store_nbr: int = Field(..., ge=1, le=45, description="Walmart store number (1-45)")
    date: date_type = Field(..., description="Week-ending Friday to forecast sales for")
    holiday_flag: int | None = Field(
        None,
        ge=0,
        le=1,
        description="1 if this week includes a major holiday (Super Bowl/Labor Day/"
        "Thanksgiving/Christmas). Auto-detected from the date if omitted.",
    )
    temperature: float | None = Field(
        None, description="Avg regional temperature (°F). Defaults to the store's last known value."
    )
    fuel_price: float | None = Field(
        None, ge=0, description="Regional fuel price (USD/gallon). Defaults to last known value."
    )
    cpi: float | None = Field(
        None, gt=0, description="Consumer Price Index. Defaults to last known value."
    )
    unemployment: float | None = Field(
        None, ge=0, description="Regional unemployment rate (%). Defaults to last known value."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "store_nbr": 1,
                "date": "2012-12-28",  # a real Christmas-week Friday in the training data
            }
        }
    }


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    store_nbr: int
    date: date_type
    predicted_sales: float
    model_name: str
    model_alias: str


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_name: str | None = None
    model_alias: str | None = None


class ErrorResponse(BaseModel):
    detail: str

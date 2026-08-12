"""Pydantic request/response schemas for the demand forecasting API."""

from __future__ import annotations

from datetime import date as date_type

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    store_nbr: int = Field(..., ge=1, description="Store number, as in stores.csv")
    family: str = Field(..., min_length=1, description="Product family, e.g. 'GROCERY I'")
    date: date_type = Field(..., description="Date to forecast sales for (YYYY-MM-DD)")
    onpromotion: int = Field(0, ge=0, description="Number of items of this family on promotion")

    @field_validator("family")
    @classmethod
    def uppercase_family(cls, v: str) -> str:
        return v.strip().upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "store_nbr": 1,
                "family": "GROCERY I",
                "date": "2017-08-20",
                "onpromotion": 5,
            }
        }
    }


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    store_nbr: int
    family: str
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

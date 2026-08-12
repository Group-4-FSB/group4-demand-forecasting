"""Inference: load the registered MLflow model and score single requests using
the reference feature snapshot (see data/snapshot.py)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from demand_forecast.config import settings
from demand_forecast.data.features import ALL_FEATURES, CATEGORICAL_FEATURES, is_major_holiday_week
from demand_forecast.data.snapshot import load_reference_artifacts


class UnknownStoreError(ValueError):
    """Raised when a store_nbr has no reference snapshot (never seen in training)."""


def load_production_model(model_uri: str | None = None):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = model_uri or f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    return mlflow.pyfunc.load_model(uri)


class PredictionService:
    """Bundles the model + reference data so both API and tests can share one
    instance instead of re-loading MLflow artifacts on every call."""

    def __init__(self, model, snapshot: pd.DataFrame):
        self.model = model
        self.snapshot = snapshot.set_index("store_nbr")

    @classmethod
    def load(
        cls, processed_dir: Path | None = None, model_uri: str | None = None
    ) -> PredictionService:
        processed_dir = processed_dir or settings.data_processed_dir
        snapshot = load_reference_artifacts(processed_dir)
        model = load_production_model(model_uri)
        return cls(model, snapshot)

    def build_feature_row(
        self,
        store_nbr: int,
        date: str | pd.Timestamp,
        holiday_flag: int | None = None,
        temperature: float | None = None,
        fuel_price: float | None = None,
        cpi: float | None = None,
        unemployment: float | None = None,
    ) -> pd.DataFrame:
        date = pd.Timestamp(date)
        try:
            ref = self.snapshot.loc[store_nbr]
        except KeyError as exc:
            raise UnknownStoreError(
                f"No reference data for store_nbr={store_nbr}. This store was not "
                "seen during training."
            ) from exc

        # Economic indicators default to the store's last known value
        # (persistence assumption — documented trade-off, see snapshot.py)
        # unless the caller supplies an actual forecast for them.
        row = {
            "store_nbr": store_nbr,
            "holiday_flag": (
                holiday_flag if holiday_flag is not None else is_major_holiday_week(date)
            ),
            "temperature": temperature if temperature is not None else ref["temperature"],
            "fuel_price": fuel_price if fuel_price is not None else ref["fuel_price"],
            "cpi": cpi if cpi is not None else ref["cpi"],
            "unemployment": unemployment if unemployment is not None else ref["unemployment"],
            "month": date.month,
            "week_of_year": int(date.isocalendar().week),
            "year": date.year,
            "sales_lag_1": ref["sales_lag_1"],
            "sales_lag_4": ref["sales_lag_4"],
            "sales_lag_52": ref["sales_lag_52"],
            "sales_roll_mean_4": ref["sales_roll_mean_4"],
            "sales_roll_mean_8": ref["sales_roll_mean_8"],
            "sales_roll_std_4": ref["sales_roll_std_4"],
        }
        frame = pd.DataFrame([row])
        for col in CATEGORICAL_FEATURES:
            frame[col] = frame[col].astype("category")
        return frame[[c for c in ALL_FEATURES if c in frame.columns]]

    def predict(
        self,
        store_nbr: int,
        date: str | pd.Timestamp,
        holiday_flag: int | None = None,
        temperature: float | None = None,
        fuel_price: float | None = None,
        cpi: float | None = None,
        unemployment: float | None = None,
    ) -> float:
        features = self.build_feature_row(
            store_nbr, date, holiday_flag, temperature, fuel_price, cpi, unemployment
        )
        log_pred = self.model.predict(features)
        pred = float(np.expm1(np.asarray(log_pred).reshape(-1)[0]))
        return max(pred, 0.0)

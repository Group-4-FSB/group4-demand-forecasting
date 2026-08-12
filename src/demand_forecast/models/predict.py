"""Inference: load the registered MLflow model and score single requests using
the reference feature snapshot (see data/snapshot.py)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from demand_forecast.config import settings
from demand_forecast.data.features import ALL_FEATURES, CATEGORICAL_FEATURES
from demand_forecast.data.ingest import build_holiday_flags
from demand_forecast.data.snapshot import load_reference_artifacts


class UnknownStoreFamilyError(ValueError):
    """Raised when a (store_nbr, family) combination has no reference snapshot."""


def load_production_model(model_uri: str | None = None):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = model_uri or f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    return mlflow.pyfunc.load_model(uri)


class PredictionService:
    """Bundles the model + reference data so both API and tests can share one
    instance instead of re-loading MLflow artifacts on every call."""

    def __init__(self, model, snapshot: pd.DataFrame, holidays: pd.DataFrame):
        self.model = model
        self.snapshot = snapshot.set_index(["store_nbr", "family"])
        self.holidays_daily = build_holiday_flags(holidays)

    @classmethod
    def load(
        cls, processed_dir: Path | None = None, model_uri: str | None = None
    ) -> PredictionService:
        processed_dir = processed_dir or settings.data_processed_dir
        snapshot, holidays = load_reference_artifacts(processed_dir)
        model = load_production_model(model_uri)
        return cls(model, snapshot, holidays)

    def _holiday_flags_for(self, date: pd.Timestamp) -> tuple[int, int]:
        row = self.holidays_daily.loc[self.holidays_daily["date"] == date]
        if row.empty:
            return 0, 0
        return int(row["is_holiday"].iloc[0]), int(row["is_event"].iloc[0])

    def build_feature_row(
        self, store_nbr: int, family: str, date: str | pd.Timestamp, onpromotion: int
    ) -> pd.DataFrame:
        date = pd.Timestamp(date)
        try:
            ref = self.snapshot.loc[(store_nbr, family)]
        except KeyError as exc:
            raise UnknownStoreFamilyError(
                f"No reference data for store_nbr={store_nbr}, family={family!r}. "
                "This store/family combination was not seen during training."
            ) from exc

        is_holiday, is_event = self._holiday_flags_for(date)
        row = {
            "store_nbr": store_nbr,
            "family": family,
            "city": ref["city"],
            "state": ref["state"],
            "store_type": ref["store_type"],
            "cluster": ref["cluster"],
            "onpromotion": onpromotion,
            "oil_price": ref["oil_price"],
            "is_holiday": is_holiday,
            "is_event": is_event,
            "day_of_week": date.dayofweek,
            "day_of_month": date.day,
            "month": date.month,
            "week_of_year": int(date.isocalendar().week),
            "is_weekend": int(date.dayofweek >= 5),
            "sales_lag_7": ref["sales_lag_7"],
            "sales_lag_14": ref["sales_lag_14"],
            "sales_lag_28": ref["sales_lag_28"],
            "sales_roll_mean_7": ref["sales_roll_mean_7"],
            "sales_roll_mean_28": ref["sales_roll_mean_28"],
            "sales_roll_std_7": ref["sales_roll_std_7"],
        }
        frame = pd.DataFrame([row])
        for col in CATEGORICAL_FEATURES:
            frame[col] = frame[col].astype("category")
        return frame[[c for c in ALL_FEATURES if c in frame.columns]]

    def predict(
        self, store_nbr: int, family: str, date: str | pd.Timestamp, onpromotion: int
    ) -> float:
        features = self.build_feature_row(store_nbr, family, date, onpromotion)
        log_pred = self.model.predict(features)
        pred = float(np.expm1(np.asarray(log_pred).reshape(-1)[0]))
        return max(pred, 0.0)

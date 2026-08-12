"""Evaluation metrics for the demand forecasting model.

RMSLE is the primary metric: it is the Kaggle competition's own metric, and it
penalizes relative (not absolute) error — appropriate since sales volume varies
hugely across store/family combinations (a few units vs thousands).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmsle(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_all(y_true, y_pred) -> dict[str, float]:
    return {
        "rmsle": rmsle(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }

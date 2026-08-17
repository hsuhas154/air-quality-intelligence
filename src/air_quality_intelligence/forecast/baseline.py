from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def naive_forecast(series: pd.Series, horizon: int = 24) -> np.ndarray:
    if series.empty:
        raise ValueError("Series cannot be empty")
    return np.repeat(float(series.iloc[-1]), horizon)


def sarima_forecast(series: pd.Series, horizon: int = 24) -> np.ndarray:
    clean = pd.Series(series, dtype="float64").dropna()
    if len(clean) < 48:
        return naive_forecast(clean, horizon)
    model = SARIMAX(clean, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24), enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    return np.asarray(fitted.forecast(horizon), dtype=float)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

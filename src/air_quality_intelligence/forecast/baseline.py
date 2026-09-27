"""Persistence baseline and the error metrics.

SARIMA used to live here too. It has moved to forecast/sarima.py, for two
reasons.

The implementation was wrong in a way that would have flattered it. It called
``dropna()`` on the series before fitting, which closes every gap and shifts
each later observation earlier in time, so a seasonal order of 24 was being
estimated against a series whose spacing was no longer hourly. It also fell
back to the naive forecast whenever it had fewer than 48 points, which meant
a benchmark could report persistence's score under SARIMA's name.

The import was also the only reason this module needed statsmodels. Keeping
the persistence baseline and the metrics free of it means they load anywhere,
including in a test environment that has no model libraries installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(series: pd.Series, horizon: int = 24) -> np.ndarray:
    """Persistence: the last observed value, repeated.

    This is the baseline every other model in the project is measured
    against, and at horizons shorter than 24 hours it is very hard to beat,
    because the AQI is a 24-hour rolling mean and two windows less than 24
    hours apart share most of their observations.
    """

    if series.empty:
        raise ValueError("Series cannot be empty")

    return np.repeat(float(series.iloc[-1]), horizon)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)

    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))

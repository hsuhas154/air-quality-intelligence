import pandas as pd

from air_quality_intelligence.forecast.baseline import mae, naive_forecast


def test_naive_forecast_repeats_last_value():
    series = pd.Series([10, 20, 30])
    assert naive_forecast(series, 3).tolist() == [30.0, 30.0, 30.0]


def test_mae():
    assert mae([1, 2, 3], [1, 4, 2]) == 1.0

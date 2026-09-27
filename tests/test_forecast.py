import pandas as pd

from air_quality_intelligence.forecast.baseline import mae, naive_forecast


def test_naive_forecast_repeats_last_value():
    series = pd.Series([10, 20, 30])
    assert naive_forecast(series, 3).tolist() == [30.0, 30.0, 30.0]


def test_mae():
    assert mae([1, 2, 3], [1, 4, 2]) == 1.0


def test_baseline_does_not_require_statsmodels():
    """This module used to import SARIMAX at load time for a function nothing
    called, which made the persistence baseline and the metrics depend on a
    model library."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src/air_quality_intelligence/forecast/baseline.py"
    ).read_text()

    imported = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "statsmodels" not in imported


def test_rmse_penalises_large_errors_more_than_mae():
    from air_quality_intelligence.forecast.baseline import rmse

    assert rmse([0, 0], [1, 1]) == 1.0
    assert rmse([0, 0], [0, 2]) > mae([0, 0], [0, 2])


def test_naive_forecast_rejects_an_empty_series():
    import pytest

    with pytest.raises(ValueError):
        naive_forecast(pd.Series(dtype="float64"), 3)

"""SARIMA inside the evaluation, tested without statsmodels.

The unit tests in test_sarima.py cover the walk-forward itself. These cover
the part that decides whether the benchmark is comparable: that SARIMA is
scored on the same folds and the same rows as the other models, and that a
partial SARIMA is reported as partial rather than averaged against a fuller
set of rows.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from air_quality_intelligence.forecast.sarima import SarimaFitReport

_spec = importlib.util.spec_from_file_location(
    "horizon_forecast",
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_horizon_forecast.py",
)
forecast = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(forecast)

HORIZON = 24


def synthetic(hours=300, cities=("Delhi", "Bengaluru")):
    """A dataset shaped like build_horizon_dataset's output."""
    rows = []

    for offset, city in enumerate(cities):
        index = pd.date_range("2026-07-01", periods=hours, freq="h", tz="UTC")

        for i, ts in enumerate(index):
            aqi = 100.0 + offset * 20 + 10.0 * np.sin(i / 12.0)
            rows.append(
                {
                    "city": city,
                    "hour": ts,
                    "aqi": aqi,
                    f"target_aqi_{HORIZON}h": aqi + 5.0,
                    "temp_c": 30.0 + (i % 10),
                    "humidity": 60.0,
                    "wind_speed": 2.0,
                    "hour_of_day": ts.hour,
                    "day_of_week": ts.dayofweek,
                }
            )

    return pd.DataFrame(rows)


def fast_forest(monkeypatch):
    """A 300-tree forest is the right production setting and the wrong test
    setting: these tests are about bookkeeping, not fit quality."""
    monkeypatch.setitem(forecast.RANDOM_FOREST_PARAMS, "n_estimators", 5)
    monkeypatch.setitem(forecast.RANDOM_FOREST_PARAMS, "max_depth", 3)


def install_stub(monkeypatch, coverage=1.0, record=None):
    """Replace the walk-forward with one that predicts a fixed share of rows."""

    def stub(series, cutoff, targets, horizon, **kwargs):
        if record is not None:
            record.append((kwargs.get("city"), cutoff, tuple(targets)))

        wanted = sorted(targets)
        keep = wanted[: max(1, int(len(wanted) * coverage))]

        predictions = {t: float(series.loc[:t].dropna().iloc[-1]) for t in keep}

        report = SarimaFitReport(
            city=kwargs.get("city", ""),
            order=(1, 1, 1),
            seasonal_order=None,
            aic=123.4,
            training_observations=int(series.loc[:cutoff].notna().sum()),
            predictions=len(predictions),
            failures=len(wanted) - len(predictions),
        )

        return predictions, report

    monkeypatch.setattr(forecast, "walk_forward_forecast", stub)


def test_sarima_is_given_the_same_folds_as_the_other_models(monkeypatch):
    """A separate script with its own splitting is free to drift from this
    one, and then the two sets of numbers are not a comparison."""
    record = []
    fast_forest(monkeypatch)
    install_stub(monkeypatch, record=record)
    forecast.TEST_OBSERVATIONS_PER_CITY = 20

    data = synthetic()
    folds = forecast.create_expanding_folds(
        data,
        data.dropna(subset=["aqi", f"target_aqi_{HORIZON}h"]),
        minimum_training_days=forecast.MINIMUM_TRAINING_DAYS,
        test_observations_per_city=20,
        number_of_folds=forecast.NUMBER_OF_FOLDS,
    )

    forecast.evaluate_horizon(data, HORIZON, sarima=True)

    seen_cutoffs = {cutoff for _, cutoff, _ in record}

    assert seen_cutoffs == {cutoff for cutoff, _ in folds}


def test_sarima_predictions_land_on_the_right_city_and_hour(monkeypatch):
    fast_forest(monkeypatch)
    install_stub(monkeypatch)
    forecast.TEST_OBSERVATIONS_PER_CITY = 20

    data = synthetic()
    _, predictions = forecast.evaluate_horizon(data, HORIZON, sarima=True)

    assert "sarima" in predictions.columns

    lookup = data.set_index(["city", "hour"])["aqi"]

    for row in predictions.dropna(subset=["sarima"]).itertuples():
        assert row.sarima == lookup.loc[(row.city, row.hour)]


def test_a_partial_sarima_is_scored_on_its_own_rows(monkeypatch):
    """Scoring SARIMA on half the rows and persistence on all of them, then
    printing the two side by side, is not a comparison."""
    fast_forest(monkeypatch)
    install_stub(monkeypatch, coverage=0.5)
    forecast.TEST_OBSERVATIONS_PER_CITY = 20

    data = synthetic()
    summary, predictions = forecast.evaluate_horizon(data, HORIZON, sarima=True)

    assert "subset" in summary.columns

    paired = summary[summary["subset"] == "sarima_rows"]
    everything = summary[summary["subset"] == "all_rows"]

    assert set(paired["model"]) == set(everything["model"])
    assert paired["n"].nunique() == 1
    assert int(paired["n"].iloc[0]) == int(predictions["sarima"].notna().sum())
    assert paired["n"].iloc[0] < everything["n"].max()


def test_full_coverage_needs_no_paired_table(monkeypatch):
    fast_forest(monkeypatch)
    install_stub(monkeypatch, coverage=1.0)
    forecast.TEST_OBSERVATIONS_PER_CITY = 20

    data = synthetic()
    summary, _ = forecast.evaluate_horizon(data, HORIZON, sarima=True)

    assert "sarima" in set(summary["model"])
    assert "subset" not in summary.columns or set(summary["subset"]) == {"all_rows"}


def test_without_the_flag_nothing_changes(monkeypatch):
    called = []

    def stub(*args, **kwargs):
        called.append(True)

        return {}, None

    fast_forest(monkeypatch)
    monkeypatch.setattr(forecast, "walk_forward_forecast", stub)
    forecast.TEST_OBSERVATIONS_PER_CITY = 20

    data = synthetic()
    summary, predictions = forecast.evaluate_horizon(data, HORIZON, sarima=False)

    assert not called
    assert "sarima" not in predictions.columns
    assert "sarima" not in set(summary["model"])


def test_every_model_is_scored_against_the_same_persistence_baseline(monkeypatch):
    fast_forest(monkeypatch)
    install_stub(monkeypatch)
    forecast.TEST_OBSERVATIONS_PER_CITY = 20

    data = synthetic()
    summary, _ = forecast.evaluate_horizon(data, HORIZON, sarima=True)

    persistence = summary[summary["model"] == "persistence"]

    assert (persistence["mae_gain_vs_persistence_pct"].abs() < 1e-9).all()

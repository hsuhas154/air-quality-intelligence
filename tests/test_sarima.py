"""The SARIMA walk-forward, exercised without statsmodels.

The fit is injectable precisely so that the part that decides whether the
benchmark is fair, which is the bookkeeping around the fit rather than the
fit itself, can be tested. The stub below records exactly which observations
the filter had seen when each forecast was asked for, which is the property
the comparison against persistence depends on.
"""

import numpy as np
import pandas as pd
import pytest

from air_quality_intelligence.forecast.sarima import (
    MINIMUM_TRAINING_OBSERVATIONS,
    to_regular_hourly,
    walk_forward_forecast,
)

START = pd.Timestamp("2026-07-01T00:00:00Z")


class StubResults:
    """Stands in for SARIMAXResults, recording what it was shown and when."""

    def __init__(self, seen, log, aic):
        self.seen = list(seen)
        self.log = log
        self.aic = aic

    def extend(self, observation):
        # type(self) so that a subclass survives the walk. Returning the base
        # class here made a subclass's behaviour vanish after one step, which
        # is exactly the kind of thing a stub should not quietly do.
        return type(self)(self.seen + list(observation.index), self.log, self.aic)

    def append(self, observation, refit=False):  # noqa: ARG002
        return self.extend(observation)

    def forecast(self, steps):
        # Record the last hour visible at forecast time, then return a ramp so
        # the caller can tell which element of the forecast it picked.
        self.log.append((self.seen[-1] if self.seen else None, steps))

        return np.arange(1, steps + 1, dtype="float64")


def make_fitter(log, aic_by_order=None):
    def fitter(series, order, seasonal_order):  # noqa: ARG001
        aic = (aic_by_order or {}).get(order, 100.0 + sum(order))

        return StubResults(list(series.index), log, aic)

    return fitter


def hourly_series(hours=240, gap=None):
    index = pd.date_range(START, periods=hours, freq="h", tz="UTC")
    values = np.linspace(50.0, 150.0, hours)
    series = pd.Series(values, index=index, name="aqi")

    if gap:
        series.iloc[gap[0] : gap[1]] = np.nan

    return series


# ---------------------------------------------------------------------------
# The series itself
# ---------------------------------------------------------------------------


def test_missing_hours_become_nan_rather_than_disappearing():
    """The bug in the earlier implementation.

    Dropping missing hours closes the holes and shifts every later
    observation earlier in the series, which silently destroys the lag
    structure the model is meant to estimate.
    """
    index = pd.date_range(START, periods=6, freq="h", tz="UTC")
    frame = pd.DataFrame({"hour": index.delete([2, 3]), "aqi": [10, 20, 50, 60]})

    series = to_regular_hourly(frame)

    assert len(series) == 6
    assert series.isna().sum() == 2
    assert series.index.freq is not None or list(series.index) == list(index)
    assert series.iloc[0] == 10
    assert series.iloc[-1] == 60


def test_regular_hourly_sorts_and_deduplicates():
    index = pd.to_datetime(
        ["2026-07-01T02:00:00Z", "2026-07-01T00:00:00Z", "2026-07-01T02:00:00Z"]
    )
    frame = pd.DataFrame({"hour": index, "aqi": [30.0, 10.0, 99.0]})

    series = to_regular_hourly(frame)

    assert list(series.index) == list(
        pd.date_range(START, periods=3, freq="h", tz="UTC")
    )
    assert series.iloc[0] == 10.0
    assert series.iloc[2] == 99.0


def test_an_empty_frame_gives_an_empty_series():
    frame = pd.DataFrame({"hour": pd.to_datetime([]), "aqi": []})

    assert to_regular_hourly(frame).empty


# ---------------------------------------------------------------------------
# What the forecaster was allowed to see
# ---------------------------------------------------------------------------


def test_a_forecast_sees_observations_up_to_its_own_hour_and_no_further():
    """The property the whole comparison rests on.

    Persistence at t uses AQI(t). If SARIMA at t had seen t+1, the benchmark
    would be measuring hindsight rather than forecasting.
    """
    series = hourly_series()
    cutoff = series.index[100]
    targets = [series.index[120], series.index[140]]
    log = []

    predictions, report = walk_forward_forecast(
        series, cutoff, targets, horizon=24, fitter=make_fitter(log)
    )

    assert set(predictions) == set(targets)
    assert report.failures == 0

    for (last_seen, _), target in zip(log, targets, strict=True):
        assert last_seen == target


def test_the_prediction_is_the_h_step_value_not_the_first_step():
    """forecast(h) returns h values; the one wanted is the last."""
    series = hourly_series()
    cutoff = series.index[100]
    target = series.index[120]
    log = []

    predictions, _ = walk_forward_forecast(
        series, cutoff, [target], horizon=24, fitter=make_fitter(log)
    )

    # The stub returns 1..steps, so picking the h-step value gives 24.
    assert predictions[target] == 24.0
    assert log[0][1] == 24


def test_parameters_are_fitted_before_the_walk_begins():
    """The fit sees the training window only, never a test hour."""
    series = hourly_series()
    cutoff = series.index[100]
    fitted_on = []

    def fitter(training, order, seasonal_order):  # noqa: ARG001
        fitted_on.append(training.index.max())

        return StubResults(list(training.index), [], 10.0 + sum(order))

    walk_forward_forecast(
        series, cutoff, [series.index[130]], horizon=24, fitter=fitter
    )

    assert fitted_on
    assert max(fitted_on) == cutoff


# ---------------------------------------------------------------------------
# Order selection
# ---------------------------------------------------------------------------


def test_the_order_with_the_lowest_training_aic_is_chosen():
    series = hourly_series()
    aic = {(1, 1, 1): 500.0, (2, 1, 2): 120.0, (0, 1, 1): 900.0}

    _, report = walk_forward_forecast(
        series,
        series.index[100],
        [series.index[120]],
        horizon=24,
        fitter=make_fitter([], aic),
    )

    assert report.order == (2, 1, 2)
    assert report.aic == 120.0


def test_orders_that_fail_to_fit_are_skipped_and_noted():
    series = hourly_series()

    def fitter(training, order, seasonal_order):  # noqa: ARG001
        if order == (1, 1, 1):
            raise ValueError("did not converge")

        return StubResults(list(training.index), [], 50.0)

    _, report = walk_forward_forecast(
        series, series.index[100], [series.index[120]], horizon=24, fitter=fitter
    )

    assert report.order != (1, 1, 1)
    assert any("did not converge" in note for note in report.notes)


def test_a_non_finite_aic_is_not_selected():
    series = hourly_series()
    aic = {(1, 1, 1): float("nan"), (2, 1, 2): 300.0, (0, 1, 1): 400.0}

    _, report = walk_forward_forecast(
        series,
        series.index[100],
        [series.index[120]],
        horizon=24,
        fitter=make_fitter([], aic),
    )

    assert report.order == (2, 1, 2)


# ---------------------------------------------------------------------------
# Failures are counted, never quietly filled in
# ---------------------------------------------------------------------------


def test_too_little_training_data_produces_no_predictions():
    """Silently falling back to persistence would report persistence's score
    under SARIMA's name, which is how a benchmark lies."""
    series = hourly_series(hours=MINIMUM_TRAINING_OBSERVATIONS + 40)
    cutoff = series.index[MINIMUM_TRAINING_OBSERVATIONS - 30]

    predictions, report = walk_forward_forecast(
        series, cutoff, [series.index[-1]], horizon=24, fitter=make_fitter([])
    )

    assert predictions == {}
    assert report.failures == 1
    assert any("below the" in note for note in report.notes)


def test_every_candidate_failing_is_reported_not_hidden():
    series = hourly_series()

    def fitter(training, order, seasonal_order):  # noqa: ARG001
        raise RuntimeError("singular matrix")

    predictions, report = walk_forward_forecast(
        series,
        series.index[100],
        [series.index[120], series.index[130]],
        horizon=24,
        fitter=fitter,
    )

    assert predictions == {}
    assert report.failures == 2
    assert any("no candidate order fitted" in note for note in report.notes)


def test_a_non_finite_forecast_counts_as_a_failure():
    series = hourly_series()

    class NanResults(StubResults):
        def forecast(self, steps):
            return np.full(steps, np.nan)

    def fitter(training, order, seasonal_order):  # noqa: ARG001
        return NanResults(list(training.index), [], 10.0)

    predictions, report = walk_forward_forecast(
        series, series.index[100], [series.index[120]], horizon=24, fitter=fitter
    )

    assert predictions == {}
    assert report.failures == 1


def test_the_append_fallback_is_used_when_extend_refuses():
    series = hourly_series()
    used = []

    class NoExtend(StubResults):
        def extend(self, observation):
            raise NotImplementedError("extend unavailable")

        def append(self, observation, refit=False):
            used.append(refit)

            return NoExtend(self.seen + list(observation.index), self.log, self.aic)

    def fitter(training, order, seasonal_order):  # noqa: ARG001
        return NoExtend(list(training.index), [], 10.0)

    predictions, report = walk_forward_forecast(
        series, series.index[100], [series.index[110]], horizon=24, fitter=fitter
    )

    assert len(predictions) == 1
    assert report.failures == 0
    assert used and all(refit is False for refit in used)


def test_gaps_in_the_series_do_not_stop_the_walk():
    """The three-day hole in late August has to be survivable."""
    series = hourly_series(gap=(105, 150))
    targets = [series.index[160], series.index[170]]

    predictions, report = walk_forward_forecast(
        series, series.index[100], targets, horizon=24, fitter=make_fitter([])
    )

    assert set(predictions) == set(targets)
    assert report.failures == 0


def test_no_targets_is_not_an_error():
    series = hourly_series()

    predictions, report = walk_forward_forecast(
        series, series.index[100], [], horizon=24, fitter=make_fitter([])
    )

    assert predictions == {}
    assert report.predictions == 0


@pytest.mark.parametrize("horizon", [1, 6, 18, 24])
def test_the_horizon_is_honoured(horizon):
    series = hourly_series()
    log = []

    predictions, _ = walk_forward_forecast(
        series,
        series.index[100],
        [series.index[120]],
        horizon=horizon,
        fitter=make_fitter(log),
    )

    assert log[0][1] == horizon
    assert predictions[series.index[120]] == float(horizon)


# ---------------------------------------------------------------------------
# Divergence: the bug the first benchmark run shipped with
# ---------------------------------------------------------------------------


def test_a_divergent_forecast_is_discarded_not_scored():
    """MAE 2.6e54.

    The first run of this benchmark reported that, because the only guard was
    np.isfinite and 1e54 is finite. A number that large is not a bad
    forecast, it is a diverged filter, and averaging it into an error metric
    destroys the metric.
    """
    series = hourly_series()

    class Explosive(StubResults):
        def forecast(self, steps):
            return np.full(steps, 1e54)

    def fitter(training, order, seasonal_order):  # noqa: ARG001
        return Explosive(list(training.index), [], 10.0)

    predictions, report = walk_forward_forecast(
        series,
        series.index[100],
        [series.index[120], series.index[130]],
        horizon=24,
        fitter=fitter,
    )

    assert predictions == {}
    assert report.diverged == 2
    assert report.failures == 2
    assert any("divergent" in note for note in report.notes)


def test_a_negative_forecast_is_discarded():
    """The AQI cannot be negative, so a negative forecast is the same class
    of failure as an explosive one."""
    series = hourly_series()

    class Negative(StubResults):
        def forecast(self, steps):
            return np.full(steps, -50.0)

    def fitter(training, order, seasonal_order):  # noqa: ARG001
        return Negative(list(training.index), [], 10.0)

    predictions, report = walk_forward_forecast(
        series, series.index[100], [series.index[120]], horizon=24, fitter=fitter
    )

    assert predictions == {}
    assert report.diverged == 1


def test_a_divergent_forecast_is_never_clamped_into_range():
    """Clamping would hide a diverged filter inside a respectable error."""
    series = hourly_series()

    class Explosive(StubResults):
        def forecast(self, steps):
            return np.full(steps, 1e30)

    def fitter(training, order, seasonal_order):  # noqa: ARG001
        return Explosive(list(training.index), [], 10.0)

    predictions, _ = walk_forward_forecast(
        series, series.index[100], [series.index[120]], horizon=24, fitter=fitter
    )

    assert not predictions


def test_the_divergence_bound_comes_from_the_training_window():
    series = hourly_series()
    training_max = float(series.loc[: series.index[100]].max())

    class AtTheEdge(StubResults):
        def __init__(self, seen, log, aic, value=0.0):
            super().__init__(seen, log, aic)
            self.value = value

        def extend(self, observation):
            return AtTheEdge(
                self.seen + list(observation.index), self.log, self.aic, self.value
            )

        def forecast(self, steps):
            return np.full(steps, self.value)

    def make(value):
        def fitter(training, order, seasonal_order):  # noqa: ARG001
            return AtTheEdge(list(training.index), [], 10.0, value)

        return fitter

    inside, _ = walk_forward_forecast(
        series,
        series.index[100],
        [series.index[120]],
        horizon=24,
        fitter=make(training_max * 2.0),
    )
    outside, _ = walk_forward_forecast(
        series,
        series.index[100],
        [series.index[120]],
        horizon=24,
        fitter=make(max(training_max * 50.0, 1e6)),
    )

    assert len(inside) == 1
    assert not outside


def test_ordinary_forecasts_are_unaffected_by_the_tripwire():
    series = hourly_series()
    log = []

    predictions, report = walk_forward_forecast(
        series, series.index[100], [series.index[120]], horizon=24, fitter=make_fitter(log)
    )

    assert len(predictions) == 1
    assert report.diverged == 0


def test_a_convergence_warning_is_recorded_rather_than_printed():
    """statsmodels warns on stderr, which scrolls past. The fold report is
    where it belongs."""
    import warnings as _warnings

    series = hourly_series()

    def fitter(training, order, seasonal_order):  # noqa: ARG001
        _warnings.warn(
            "Maximum Likelihood optimization failed to converge.",
            stacklevel=1,
        )

        return StubResults(list(training.index), [], 10.0)

    _, report = walk_forward_forecast(
        series, series.index[100], [series.index[120]], horizon=24, fitter=fitter
    )

    assert any("did not converge" in note for note in report.notes)


def test_the_fit_constrains_the_model_to_the_stationary_region():
    """The root cause.

    With enforce_stationarity off, an AR root inside the unit circle makes
    the process explosive: the forecast grows like phi to the horizon and the
    filter state like phi to the length of the walk. That is where 1e54 came
    from.
    """
    import sys
    import types

    captured = {}

    class FakeSARIMAX:
        def __init__(self, series, **kwargs):
            captured.update(kwargs)

        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs

            return StubResults([], [], 1.0)

    module = types.ModuleType("statsmodels.tsa.statespace.sarimax")
    module.SARIMAX = FakeSARIMAX

    for name in (
        "statsmodels",
        "statsmodels.tsa",
        "statsmodels.tsa.statespace",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    sys.modules["statsmodels.tsa.statespace.sarimax"] = module

    try:
        from air_quality_intelligence.forecast.sarima import default_fitter

        default_fitter(hourly_series(hours=100), (1, 1, 1), None)
    finally:
        for name in (
            "statsmodels.tsa.statespace.sarimax",
            "statsmodels.tsa.statespace",
            "statsmodels.tsa",
            "statsmodels",
        ):
            sys.modules.pop(name, None)

    assert captured["enforce_stationarity"] is True
    assert captured["enforce_invertibility"] is True
    assert captured["fit_kwargs"]["maxiter"] >= 200

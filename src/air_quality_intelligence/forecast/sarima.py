"""SARIMA forecasting of the hourly CPCB AQI, benchmarked honestly.

The point of this module is comparability. Persistence predicts AQI(t+h) as
AQI(t), and the Random Forest predicts it from features known at t. A SARIMA
benchmark only means something if it answers the same question from the same
information, so everything here is arranged to make that true:

Regular hourly index, gaps left missing
    The AQI series has holes, including a three-day one at the end of August.
    Dropping the missing hours, as the earlier implementation did, closes the
    holes and silently shifts every later observation earlier in the series.
    That destroys the lag structure the model is supposed to estimate, and it
    does so invisibly. Here the series is reindexed to every hour and the
    holes stay as NaN, which the Kalman filter treats as what they are:
    missing information at a known time.

Parameters from training data only
    The order is chosen by AIC over a small candidate set, fitted on the
    training window alone, once per fold. Choosing it on the test period, or
    once over the whole series, would be selecting a model with knowledge of
    what it is about to be scored on.

State updated, parameters frozen
    At each test hour the filter is advanced with the observations that
    existed at that hour, then asked for an h-step forecast. The coefficients
    never see the test period; only the state does, and the state is the
    history a real forecaster would have. Refitting at every step would be
    more thorough and is far too slow to be worth it here.

The statsmodels fit is injectable so that everything above can be tested
without statsmodels installed. See tests/test_sarima.py, which exercises the
whole walk-forward against a stub.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Candidate (p, d, q) orders, scored by AIC on the training window.
#
# The AQI is a 24-hour rolling mean, so it arrives already smoothed and the
# diurnal cycle is largely averaged out of it. That is why the default
# candidates carry no seasonal term: a seasonal period of 24 on a 24-hour mean
# is mostly modelling the smoothing. --seasonal adds one back for comparison.
DEFAULT_ORDERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (2, 1, 2),
    (0, 1, 1),
)

SEASONAL_ORDER: tuple[int, int, int, int] = (1, 0, 1, 24)

# Below this many observed hours a fit is not worth attempting.
MINIMUM_TRAINING_OBSERVATIONS = 72

# The likelihood surface for a 24-hour rolling mean is close to flat, so the
# default iteration budget is often spent without converging.
MAXIMUM_FIT_ITERATIONS = 200

# A forecast further than this multiple of the training window's highest
# observed AQI is divergence, not a prediction.
#
# The exact multiple does not matter, and that is the argument for it. The
# highest AQI anywhere in this dataset is 712. A diverging SARIMA produced
# 1e54. Any threshold in the fifty orders of magnitude between those two
# rejects exactly the same forecasts, so the number is a tripwire rather than
# a tuned parameter.
#
# A tripped forecast is recorded as a failure and dropped. It is never
# clamped: clamping a divergent forecast to a plausible number would hide the
# divergence inside a respectable-looking error metric.
DIVERGENCE_MULTIPLE = 5.0

# Floor for that bound, so an unusually clean training window cannot set a
# tripwire so low that ordinary forecasts trip it.
DIVERGENCE_FLOOR = 1000.0


@dataclass
class SarimaFitReport:
    """What one fold's fit did, so the run can be audited."""

    city: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None
    aic: float | None
    training_observations: int
    predictions: int
    failures: int
    diverged: int = 0
    converged: bool | None = None
    notes: list[str] = field(default_factory=list)


def to_regular_hourly(
    frame: pd.DataFrame,
    *,
    time_column: str = "hour",
    value_column: str = "aqi",
) -> pd.Series:
    """One city's AQI on a complete hourly index, holes left as NaN.

    Reindexing rather than dropping is the whole point: an hour with no
    observation has to stay an hour with no observation, or every lag after it
    refers to the wrong time.
    """

    series = (
        frame[[time_column, value_column]]
        .dropna(subset=[time_column])
        .drop_duplicates(subset=[time_column], keep="last")
        .set_index(time_column)[value_column]
        .sort_index()
        .astype("float64")
    )

    if series.empty:
        return series

    full = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="h",
        tz=series.index.tz,
    )

    reindexed = series.reindex(full)
    reindexed.index.name = time_column

    return reindexed


def default_fitter(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] | None,
):
    """Fit SARIMAX. Imported lazily so the module loads without statsmodels.

    enforce_stationarity and enforce_invertibility are on, and that is not a
    detail. With them off the optimiser is free to settle on an AR root
    inside the unit circle, which makes the process explosive: the forecast
    grows like phi to the power of the horizon, and the filter state grows
    like phi to the power of the walk. The first run of this benchmark had
    them off, copied from the implementation this replaced, and reported a
    mean absolute error of 2.6e54. That is not a bad forecast, it is a
    divergent one.

    The AQI makes this easy to fall into. It is a 24-hour rolling mean, so
    the differenced series is very nearly flat, the likelihood surface is
    correspondingly flat, and an unconstrained optimiser wanders. Constrain
    it to the stationary region and it has nowhere bad to wander to.
    """

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order or (0, 0, 0, 0),
        enforce_stationarity=True,
        enforce_invertibility=True,
    )

    return model.fit(disp=False, maxiter=MAXIMUM_FIT_ITERATIONS)


def select_order(
    training: pd.Series,
    orders: Sequence[tuple[int, int, int]],
    seasonal_order: tuple[int, int, int, int] | None,
    fitter: Callable = default_fitter,
) -> tuple[tuple[int, int, int], object, float | None, list[str]]:
    """Pick the order with the lowest AIC on the training window.

    Returns the order, the fitted results object, its AIC, and any notes.
    Selection touches the training window only.
    """

    notes: list[str] = []
    best: tuple[tuple[int, int, int], object, float] | None = None

    for order in orders:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = fitter(training, order, seasonal_order)

            for warning in caught:
                if "converge" in str(warning.message).lower():
                    notes.append(f"order {order} did not converge")
                    break

            aic = float(getattr(fitted, "aic", np.nan))
        except Exception as exc:  # noqa: BLE001 - any fit failure is a skip
            notes.append(f"order {order} failed to fit: {exc}")
            continue

        if not np.isfinite(aic):
            notes.append(f"order {order} produced a non-finite AIC")
            continue

        if best is None or aic < best[2]:
            best = (order, fitted, aic)

    if best is None:
        return orders[0], None, None, notes

    return best[0], best[1], best[2], notes


def _advance(results, observation: pd.Series):
    """Move the filter forward by the given observations, keeping parameters.

    ``extend`` continues from the existing final state and is the cheap path.
    Some statsmodels versions and model shapes refuse it, so ``append`` with
    ``refit=False`` is the fallback: slower, because it refilters the whole
    series, but it keeps the parameters frozen, which is the property that
    matters.
    """

    try:
        return results.extend(observation)
    except Exception:  # noqa: BLE001 - fall back rather than abandon the fold
        return results.append(observation, refit=False)


def walk_forward_forecast(
    series: pd.Series,
    cutoff: pd.Timestamp,
    targets: Sequence[pd.Timestamp],
    horizon: int,
    *,
    city: str = "",
    orders: Sequence[tuple[int, int, int]] = DEFAULT_ORDERS,
    seasonal_order: tuple[int, int, int, int] | None = None,
    fitter: Callable = default_fitter,
) -> tuple[dict[pd.Timestamp, float], SarimaFitReport]:
    """Predict AQI(t+horizon) at each t in targets, using data up to t.

    series must be on a regular hourly index with gaps as NaN. cutoff is the
    last hour of the training window. targets are the hours at which a
    forecast is wanted, all of which must be after cutoff.
    """

    report = SarimaFitReport(
        city=city,
        order=orders[0],
        seasonal_order=seasonal_order,
        aic=None,
        training_observations=0,
        predictions=0,
        failures=0,
    )

    wanted = sorted({pd.Timestamp(t) for t in targets})

    if not wanted:
        return {}, report

    training = series.loc[series.index <= cutoff]
    observed = int(training.notna().sum())
    report.training_observations = observed

    if observed < MINIMUM_TRAINING_OBSERVATIONS:
        report.notes.append(
            f"only {observed} observed training hours, below the "
            f"{MINIMUM_TRAINING_OBSERVATIONS} needed to fit"
        )
        report.failures = len(wanted)
        return {}, report

    order, results, aic, notes = select_order(
        training, orders, seasonal_order, fitter
    )
    report.order = order
    report.aic = aic
    report.notes.extend(notes)

    if results is None:
        report.notes.append("no candidate order fitted; fold skipped")
        report.failures = len(wanted)
        return {}, report

    # Divergence tripwire, anchored on what the training window actually
    # contained rather than on a constant.
    training_max = float(training.max(skipna=True)) if observed else 0.0
    bound = max(DIVERGENCE_MULTIPLE * training_max, DIVERGENCE_FLOOR)

    predictions: dict[pd.Timestamp, float] = {}
    diverged = 0
    wanted_set = set(wanted)
    last = wanted[-1]

    # Hours strictly after the cutoff, up to the last hour a forecast is
    # wanted for. The filter is advanced one hour at a time so that a forecast
    # made at hour u has seen the observations up to and including u, and no
    # more.
    walk = series.loc[(series.index > cutoff) & (series.index <= last)]

    for timestamp in walk.index:
        try:
            results = _advance(results, walk.loc[[timestamp]])
        except Exception as exc:  # noqa: BLE001 - record and stop this fold
            report.notes.append(f"filter failed at {timestamp}: {exc}")
            break

        if timestamp not in wanted_set:
            continue

        try:
            forecast = np.asarray(results.forecast(horizon), dtype="float64")
        except Exception as exc:  # noqa: BLE001 - record and continue
            report.notes.append(f"forecast failed at {timestamp}: {exc}")
            report.failures += 1
            continue

        if forecast.size < horizon or not np.isfinite(forecast[horizon - 1]):
            continue

        value = float(forecast[horizon - 1])

        # Checking finiteness alone is not enough, and the first run of this
        # benchmark proved it: 1e54 is a perfectly finite number, so it
        # passed the guard and went straight into the mean absolute error.
        if value < 0.0 or value > bound:
            diverged += 1
            continue

        predictions[timestamp] = value

    report.predictions = len(predictions)
    report.failures = len(wanted) - len(predictions)
    report.diverged = diverged

    if diverged:
        report.notes.append(
            f"{diverged} forecasts outside [0, {bound:g}] were discarded as "
            "divergent rather than scored"
        )

    return predictions, report

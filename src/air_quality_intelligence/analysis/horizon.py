"""Multi-horizon CPCB AQI forecasting.

The hourly AQI is built from 24-hour rolling pollutant averages. Two windows
that are less than 24 hours apart therefore share most of their observations,
which makes the persistence baseline extremely strong at short horizons for
arithmetic rather than atmospheric reasons.

Measured autocorrelation of the hardened AQI against window overlap:

    horizon   corr(AQI_t, AQI_t+h)   window overlap
    1h        0.997                  96%
    6h        0.942                  75%
    12h       0.783                  50%
    18h       0.459                  25%
    24h       -0.085                 0%

A forecast only adds value once the windows stop overlapping, so this module
targets horizons at or beyond 18 hours. The 24-hour horizon also matches what
CPCB publishes operationally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from air_quality_intelligence.analysis.aqi import calculate_aqi
from air_quality_intelligence.analysis.temporal import calculate_strict_averages

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "o3", "co"]

# Horizons in hours. Below 18 the persistence baseline wins because the
# CPCB averaging windows still overlap.
DEFAULT_HORIZONS = (18, 24)

AQI_COLUMN = "aqi"
TARGET_TEMPLATE = "target_aqi_{horizon}h"


def add_strict_averages(features: pd.DataFrame) -> pd.DataFrame:
    """Attach the CPCB temporal average of each pollutant as avg_<pollutant>.

    These averages, not the raw hourly concentrations, are what the AQI is
    calculated from.
    """

    long = features.melt(
        id_vars=["city", "hour"],
        value_vars=POLLUTANTS,
        var_name="pollutant",
        value_name="value",
    ).dropna(subset=["value"])

    averages = calculate_strict_averages(
        long.rename(columns={"hour": "ts"}),
        group_cols=("city",),
    )

    wide = averages.pivot_table(
        index=["city", "ts"],
        columns="pollutant",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None

    wide = wide.rename(
        columns={"ts": "hour", **{p: f"avg_{p}" for p in POLLUTANTS}}
    )

    return features.merge(wide, on=["city", "hour"], how="left")


def calculate_row_aqi(row: pd.Series) -> float:
    """Return the CPCB AQI for one row, or NaN if any pollutant is missing."""

    if not all(pd.notna(row.get(f"avg_{p}")) for p in POLLUTANTS):
        return np.nan

    aqi, _ = calculate_aqi({p: row[f"avg_{p}"] for p in POLLUTANTS})

    return float(aqi)


def build_horizon_dataset(
    features: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Build the AQI column and one forecast target per horizon.

    Targets are taken from the real future timestamp rather than a positional
    shift, so a gap in the hourly series produces a missing target instead of
    a silently mislabelled one.
    """

    data = features.copy()
    data["hour"] = pd.to_datetime(data["hour"], utc=True)

    data = add_strict_averages(data)
    data[AQI_COLUMN] = data.apply(calculate_row_aqi, axis=1)

    # Honour the existing PM2.5 station-coverage gate.
    if "pm25_coverage_valid" in data.columns:
        data.loc[data["pm25_coverage_valid"] != True, AQI_COLUMN] = np.nan  # noqa: E712

    parts = []

    for _, group in data.groupby("city"):
        group = group.set_index("hour").sort_index()

        for horizon in horizons:
            group[TARGET_TEMPLATE.format(horizon=horizon)] = (
                group[AQI_COLUMN].shift(-horizon, freq="h").reindex(group.index)
            )

        parts.append(group.reset_index())

    return (
        pd.concat(parts)
        .sort_values(["city", "hour"])
        .reset_index(drop=True)
    )


def model_feature_names(data: pd.DataFrame) -> list[str]:
    """Return the model features, excluding anything derived from the future."""

    candidates = (
        POLLUTANTS
        + [f"avg_{p}" for p in POLLUTANTS]
        + [f"{p}_station_count" for p in POLLUTANTS]
        + [
            "total_station_count",
            "pm25_coverage_valid",
            "temp_c",
            "humidity",
            "wind_speed",
            "wind_dir",
            "blh",
            "hour_of_day",
            "day_of_week",
        ]
        + [f"{p}_lag{lag}" for p in POLLUTANTS for lag in (1, 3, 6)]
    )

    forbidden = {AQI_COLUMN} | {
        column for column in data.columns if column.startswith("target_")
    }

    selected = [
        column
        for column in candidates
        if column in data.columns and column not in forbidden
    ]

    leaked = forbidden.intersection(selected)
    if leaked:
        raise ValueError(f"Future information in model features: {sorted(leaked)}")

    return selected


def build_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-safe numeric matrices using training-only imputation."""

    X_train = train[features].copy()
    X_test = test[features].copy()

    for frame in (X_train, X_test):
        for column in frame.columns:
            if frame[column].dtype == bool:
                frame[column] = frame[column].astype(int)

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    all_missing = [c for c in X_train.columns if X_train[c].isna().all()]

    if all_missing:
        X_train = X_train.drop(columns=all_missing)
        X_test = X_test.drop(columns=all_missing)

    medians = X_train.median(numeric_only=True)

    return X_train.fillna(medians), X_test.fillna(medians)


def create_expanding_folds(
    data: pd.DataFrame,
    evaluable: pd.DataFrame,
    minimum_training_days: int = 7,
    test_observations_per_city: int = 16,
    number_of_folds: int = 10,
) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Strict calendar-time expanding-window folds with a leakage assertion."""

    start = max(group["hour"].min() for _, group in data.groupby("city"))
    cutoff = start + pd.Timedelta(days=minimum_training_days)

    folds = []

    for _ in range(number_of_folds):
        future = evaluable[evaluable["hour"] > cutoff]

        test_parts = [
            group.sort_values("hour").head(test_observations_per_city)
            for _, group in future.groupby("city")
        ]

        if not test_parts:
            break

        test = pd.concat(test_parts)

        if any(
            len(group) < test_observations_per_city
            for _, group in test.groupby("city")
        ):
            break

        if cutoff >= test["hour"].min():
            raise ValueError("Temporal leakage: training cutoff is not before the test window.")

        folds.append((cutoff, test))
        cutoff = test["hour"].max()

    return folds

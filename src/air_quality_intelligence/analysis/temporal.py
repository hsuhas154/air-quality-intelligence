"""Strict temporal averaging for CPCB-style AQI inputs."""

from __future__ import annotations

import pandas as pd


AVERAGING_HOURS = {
    "pm25": 24,
    "pm10": 24,
    "no2": 24,
    "so2": 24,
    "o3": 8,
    "co": 8,
}


def strict_rolling_mean(
    series: pd.Series,
    hours: int,
) -> pd.Series:
    """
    Calculate a rolling hourly mean only when every required hourly
    observation is present.
    """
    if hours <= 0:
        raise ValueError("hours must be positive.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a DatetimeIndex.")

    if series.index.has_duplicates:
        raise ValueError("Duplicate timestamps are not allowed.")

    series = series.sort_index()

    if series.empty:
        return series.astype(float)

    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="h",
        tz=series.index.tz,
    )

    hourly = series.reindex(full_index)

    result = hourly.rolling(
        window=hours,
        min_periods=hours,
    ).mean()

    result.index.name = series.index.name

    return result


def calculate_strict_averages(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "ts",
    pollutant_col: str = "pollutant",
    value_col: str = "value",
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    Calculate strict CPCB-style rolling averages.

    Each pollutant receives its required averaging period from
    AVERAGING_HOURS.

    Optional group_cols allow independent averaging by station,
    city, etc.
    """
    required = {
        timestamp_col,
        pollutant_col,
        value_col,
        *group_cols,
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    work = df.copy()

    work[timestamp_col] = pd.to_datetime(
        work[timestamp_col],
        utc=True,
    )

    work[pollutant_col] = (
        work[pollutant_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unsupported = sorted(
        set(work[pollutant_col]) - set(AVERAGING_HOURS)
    )

    if unsupported:
        raise ValueError(
            f"Unsupported pollutants: {unsupported}"
        )

    work[timestamp_col] = work[timestamp_col].dt.floor("h")

    grouping = [*group_cols, pollutant_col]

    if work.duplicated(
        subset=[*grouping, timestamp_col]
    ).any():
        raise ValueError(
            "Duplicate pollutant/group/hour observations found."
        )

    results: list[pd.DataFrame] = []

    for keys, group in work.groupby(
        grouping,
        sort=True,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        pollutant = keys[-1]
        hours = AVERAGING_HOURS[pollutant]

        series = group.set_index(timestamp_col)[value_col]
        series.index.name = timestamp_col

        rolling = strict_rolling_mean(series, hours)

        valid = rolling.dropna()

        if valid.empty:
            continue

        result = (
            valid.rename("value")
            .reset_index()
        )

        for column, key in zip(group_cols, keys[:-1]):
            result[column] = key

        result[pollutant_col] = pollutant
        result["averaging_hours"] = hours

        result = result[
            [
                *group_cols,
                timestamp_col,
                pollutant_col,
                "value",
                "averaging_hours",
            ]
        ]

        results.append(result)

    if not results:
        return pd.DataFrame(
            columns=[
                *group_cols,
                timestamp_col,
                pollutant_col,
                "value",
                "averaging_hours",
            ]
        )

    return (
        pd.concat(results, ignore_index=True)
        .sort_values(
            [*group_cols, timestamp_col, pollutant_col]
        )
        .reset_index(drop=True)
    )

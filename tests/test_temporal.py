import pandas as pd
import pytest

from air_quality_intelligence.analysis.temporal import (
    AVERAGING_HOURS,
    calculate_strict_averages,
    strict_rolling_mean,
)


def hourly_index(n):
    return pd.date_range(
        "2026-01-01 00:00",
        periods=n,
        freq="h",
        tz="UTC",
    )


def test_complete_24_hour_window():
    index = hourly_index(24)
    series = pd.Series(range(1, 25), index=index)

    result = strict_rolling_mean(series, 24)

    assert result.iloc[-1] == 12.5
    assert result.notna().sum() == 1


def test_missing_hour_invalidates_24_hour_window():
    index = hourly_index(24).delete(12)
    series = pd.Series(range(1, 24), index=index)

    result = strict_rolling_mean(series, 24)

    assert result.dropna().empty


def test_complete_8_hour_window():
    index = hourly_index(8)
    series = pd.Series(range(1, 9), index=index)

    result = strict_rolling_mean(series, 8)

    assert result.iloc[-1] == 4.5
    assert result.notna().sum() == 1


def test_missing_hour_invalidates_8_hour_window():
    index = hourly_index(8).delete(4)
    series = pd.Series(range(1, 8), index=index)

    result = strict_rolling_mean(series, 8)

    assert result.dropna().empty


def test_gap_does_not_create_false_valid_window():
    index = hourly_index(10).delete(5)
    series = pd.Series(range(1, 10), index=index)

    result = strict_rolling_mean(series, 8)

    assert result.dropna().empty


def test_duplicate_timestamps_raise():
    index = hourly_index(8)
    index = index.insert(4, index[4])

    series = pd.Series(range(9), index=index)

    with pytest.raises(ValueError, match="Duplicate timestamps"):
        strict_rolling_mean(series, 8)


def test_timestamps_are_normalized_to_utc():
    df = pd.DataFrame(
        {
            "ts": pd.date_range(
                "2026-01-01 00:15",
                periods=24,
                freq="h",
                tz="Asia/Kolkata",
            ),
            "pollutant": ["pm25"] * 24,
            "value": [10.0] * 24,
        }
    )

    result = calculate_strict_averages(df)

    assert len(result) == 1
    assert result.iloc[0]["value"] == 10.0
    assert result.iloc[0]["averaging_hours"] == 24
    assert str(result.iloc[0]["ts"].tz) == "UTC"


def test_pollutant_specific_averaging_periods():
    rows = []

    for pollutant, hours in AVERAGING_HOURS.items():
        for i in range(hours):
            rows.append(
                {
                    "ts": pd.Timestamp(
                        "2026-01-01 00:00",
                        tz="UTC",
                    ) + pd.Timedelta(hours=i),
                    "pollutant": pollutant,
                    "value": 10.0,
                }
            )

    df = pd.DataFrame(rows)

    result = calculate_strict_averages(df)

    assert set(result["pollutant"]) == set(AVERAGING_HOURS)

    for pollutant, hours in AVERAGING_HOURS.items():
        subset = result[result["pollutant"] == pollutant]

        assert len(subset) == 1
        assert subset.iloc[0]["averaging_hours"] == hours
        assert subset.iloc[0]["value"] == 10.0


def test_group_columns_are_processed_independently():
    rows = []

    for station, value in [("A", 10.0), ("B", 20.0)]:
        for i in range(24):
            rows.append(
                {
                    "station_id": station,
                    "ts": pd.Timestamp(
                        "2026-01-01 00:00",
                        tz="UTC",
                    ) + pd.Timedelta(hours=i),
                    "pollutant": "pm25",
                    "value": value,
                }
            )

    df = pd.DataFrame(rows)

    result = calculate_strict_averages(
        df,
        group_cols=("station_id",),
    )

    assert len(result) == 2

    values = dict(
        zip(result["station_id"], result["value"])
    )

    assert values["A"] == 10.0
    assert values["B"] == 20.0

"""City-level aggregation, the coverage gate, and the unit guardrails.

These are the steps between a raw sensor reading and the AQI the model is
trained on, so a defect here is invisible in the AQI code and shows up only as
a number that is quietly wrong.
"""

import numpy as np
import pandas as pd
import pytest

from air_quality_intelligence.analysis.features import (
    MIN_PM25_STATION_FRACTION,
    MINIMUM_READINGS_TO_JUDGE_A_SENSOR,
    _build_temporal_aqi_table,
    _city_hourly_concentrations,
    _normalize_measurement_units,
    _pm25_station_thresholds,
)

HOUR = pd.Timestamp("2026-07-01T10:00:00Z")
LATER = pd.Timestamp("2026-07-01T11:00:00Z")


def test_each_station_counts_once_however_often_it_reports():
    """A chatty station must not outweigh a quiet one.

    Station 1 reports four times at 100 and station 2 once at 20. Averaging
    the rows gives 84, which is a reporting-frequency artefact. Averaging
    station means gives 60, the mean of the two stations.
    """
    rows = [(HOUR, 1, "pm25", 100.0)] * 4 + [(HOUR, 2, "pm25", 20.0)]
    frame = pd.DataFrame(rows, columns=["hour", "station_id", "pollutant", "value"])
    frame["city"] = "Delhi"

    result = _city_hourly_concentrations(frame)

    assert len(result) == 1
    assert result.loc[0, "value"] == pytest.approx(60.0)
    assert result.loc[0, "station_count"] == 2


def test_sub_hourly_readings_from_one_station_are_averaged_first():
    rows = [
        (HOUR, 1, "pm25", 40.0),
        (HOUR, 1, "pm25", 60.0),
        (HOUR, 2, "pm25", 200.0),
    ]
    frame = pd.DataFrame(rows, columns=["hour", "station_id", "pollutant", "value"])
    frame["city"] = "Delhi"

    result = _city_hourly_concentrations(frame)

    assert result.loc[0, "value"] == pytest.approx(125.0)


def test_hours_and_pollutants_stay_separate():
    rows = [
        (HOUR, 1, "pm25", 40.0),
        (LATER, 1, "pm25", 80.0),
        (HOUR, 1, "pm10", 120.0),
    ]
    frame = pd.DataFrame(rows, columns=["hour", "station_id", "pollutant", "value"])
    frame["city"] = "Delhi"

    result = _city_hourly_concentrations(frame).set_index(["hour", "pollutant"])

    assert result.loc[(HOUR, "pm25"), "value"] == pytest.approx(40.0)
    assert result.loc[(LATER, "pm25"), "value"] == pytest.approx(80.0)
    assert result.loc[(HOUR, "pm10"), "value"] == pytest.approx(120.0)


def test_coverage_threshold_follows_the_median_not_the_maximum():
    """The old gate was the observed maximum, so one outage failed the hour."""
    counts = [31, 30, 29, 28, 30, 29]
    df = pd.DataFrame({"city": "Delhi", "pm25_station_count": counts})

    expected = np.ceil(np.median(counts) * MIN_PM25_STATION_FRACTION)
    thresholds = _pm25_station_thresholds(df)

    assert set(thresholds) == {expected}
    assert expected < max(counts)


def test_coverage_threshold_is_computed_per_city():
    df = pd.DataFrame(
        {
            "city": ["Delhi"] * 3 + ["Bengaluru"] * 3,
            "pm25_station_count": [30, 30, 30, 8, 8, 8],
        }
    )

    thresholds = _pm25_station_thresholds(df)

    assert thresholds.iloc[0] == np.ceil(30 * MIN_PM25_STATION_FRACTION)
    assert thresholds.iloc[-1] == np.ceil(8 * MIN_PM25_STATION_FRACTION)


def test_coverage_threshold_is_never_below_one():
    df = pd.DataFrame({"city": ["Delhi"], "pm25_station_count": [1]})

    assert _pm25_station_thresholds(df).iloc[0] == 1.0


def test_coverage_threshold_defaults_to_one_without_counts():
    df = pd.DataFrame(
        {"city": ["Delhi"], "pm25_station_count": [np.nan]},
    )

    assert _pm25_station_thresholds(df).iloc[0] == 1.0


def measurement_frame(rows):
    return pd.DataFrame(
        rows,
        columns=["station_id", "pollutant", "value", "unit"],
    )


def test_mislabelled_co_is_corrected_inside_the_pipeline():
    """The correction existed but nothing called it.

    A CO reading of 0.48 labelled ppb is 0.55 mg/m3 once the label is
    corrected. Taken literally it is 0.00055 mg/m3, three orders of magnitude
    below the global background.
    """
    frame = measurement_frame([(17, "co", 0.48, "ppb")])

    result = _normalize_measurement_units(frame)

    assert result.loc[0, "value"] == pytest.approx(0.55, abs=0.01)
    assert result.loc[0, "unit"] == "mg/m³"


def test_plausible_readings_pass_through_normalization():
    frame = measurement_frame(
        [
            (1, "pm25", 24.4, "µg/m³"),
            (1, "no2", 13.3, "ppb"),
            (1, "so2", 12.9, "ppb"),
            (1, "o3", 0.02, "ppm"),
        ]
    )

    result = _normalize_measurement_units(frame)

    assert len(result) == 4
    assert (result["value"] > 0).all()


def test_an_isolated_impossible_reading_is_dropped_not_fatal():
    """One bad reading is missing data, not a reason to stop the run.

    A CO analyser sitting at its zero produces readings below the global
    atmospheric background. Those are unusable, but they say nothing about
    the other sensors at the same station, and aborting on them stopped the
    whole pipeline on one station.
    """
    frame = measurement_frame(
        [(1, "pm25", 24.4, "µg/m³"), (1, "pm25", 100000.0, "µg/m³")]
    )

    result = _normalize_measurement_units(frame)

    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(24.4)


def test_a_systematically_failing_feed_is_reported_not_fatal(capsys):
    """A whole dead feed is dropped and named, and the run continues.

    Naming it is the point: a silent drop hides a unit label error, and an
    exception stops a ninety-day run over one station.
    """
    readings = MINIMUM_READINGS_TO_JUDGE_A_SENSOR

    frame = measurement_frame(
        [(6984, "co", 0.03, "ppb")] * readings
        + [(17, "co", 0.48, "ppb")] * readings
    )

    result = _normalize_measurement_units(frame)

    assert set(result["station_id"]) == {17}
    assert len(result) == readings

    report = capsys.readouterr().out

    assert "station_id=6984" in report
    assert "impossible" in report


def test_isolated_bad_readings_do_not_name_a_feed(capsys):
    """Scattered noise is dropped quietly; only a failing feed is named."""
    frame = measurement_frame(
        [(1, "pm25", 24.4, "µg/m³")] * MINIMUM_READINGS_TO_JUDGE_A_SENSOR
        + [(1, "pm25", 99999.0, "µg/m³")]
    )

    result = _normalize_measurement_units(frame)

    assert len(result) == MINIMUM_READINGS_TO_JUDGE_A_SENSOR
    assert "station_id=1" not in capsys.readouterr().out


def test_a_short_feed_is_not_named(capsys):
    """Too few readings to tell a dead sensor from a bad hour."""
    frame = measurement_frame([(1, "co", 0.001, "ppm")] * 3)

    result = _normalize_measurement_units(frame)

    assert result.empty
    assert "station_id=1" not in capsys.readouterr().out


def test_the_documented_pm25_record_is_not_rejected():
    """The highest documented hourly mean PM2.5 over northwest India."""
    frame = measurement_frame([(1, "pm25", 1341.0, "µg/m³")])

    result = _normalize_measurement_units(frame)

    assert result.loc[0, "value"] == pytest.approx(1341.0)


def test_non_finite_measurements_are_dropped_before_conversion():
    frame = measurement_frame(
        [
            (1, "pm25", 24.4, "µg/m³"),
            (1, "pm25", np.nan, "µg/m³"),
            (1, "pm25", np.inf, "µg/m³"),
        ]
    )

    result = _normalize_measurement_units(frame)

    assert len(result) == 1


def test_temporal_aqi_table_accepts_aggregated_input():
    """Guard the contract between the two aggregation steps.

    The temporal AQI table is fed the city-hour frame, which has no
    station_id. A change that made it re-aggregate by station would raise
    here rather than in production.
    """
    hours = pd.date_range("2026-07-01", periods=30, freq="h", tz="UTC")

    rows = []
    for pollutant, value in [
        ("pm25", 40.0),
        ("pm10", 80.0),
        ("no2", 30.0),
        ("so2", 20.0),
        ("o3", 25.0),
        ("co", 0.8),
    ]:
        for hour in hours:
            rows.append(("Delhi", hour, pollutant, value, 30))

    frame = pd.DataFrame(
        rows,
        columns=["city", "hour", "pollutant", "value", "station_count"],
    )

    result = _build_temporal_aqi_table(frame)

    assert not result.empty
    assert result["aqi_temporal_valid"].any()
    assert result.loc[result["aqi_temporal_valid"], "hourly_aqi"].notna().all()


def test_temporal_aqi_table_rejects_unaggregated_input():
    frame = pd.DataFrame(
        {
            "city": ["Delhi"],
            "hour": [HOUR],
            "value": [40.0],
        }
    )

    with pytest.raises(ValueError, match="aggregated"):
        _build_temporal_aqi_table(frame)

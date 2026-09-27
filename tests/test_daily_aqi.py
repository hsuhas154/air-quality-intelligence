"""The published aggregation rules that decide whether a day gets an AQI.

Every test here is a day that the previous implementation published and that
CPCB's own rules do not allow, or a day it silently dropped.
"""

from datetime import date

import pandas as pd
import pytest

from air_quality_intelligence.analysis.daily_aqi import (
    MINIMUM_HOURS_FOR_DAILY_MEAN,
    SHORT_TERM_WINDOW_HOURS,
    build_daily_aqi_rows,
)
from air_quality_intelligence.db.daily_aqi import (
    DailyAqiRecord,
    replace_station_daily_aqi,
)
from air_quality_intelligence.transform.units import CANONICAL_UNITS

DAY = "2026-07-01"
NEXT_DAY = "2026-07-02"


def measurements(rows):
    """Build a measurements frame from (ts, pollutant, value) triples.

    Values are given in each pollutant's canonical AQI unit, so the unit
    column is filled in from CANONICAL_UNITS rather than by hand.
    """
    frame = pd.DataFrame(rows, columns=["ts", "pollutant", "value"])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["unit"] = [CANONICAL_UNITS[p] for p in frame["pollutant"]]
    return frame


def hours(pollutant, value, count, day=DAY, start=0):
    return [
        (f"{day}T{start + hour:02d}:00:00Z", pollutant, value)
        for hour in range(count)
    ]


def full_day(pm25=40.0, no2=30.0, so2=20.0, day=DAY):
    """A day that satisfies every rule: three pollutants, PM present, 24 hours."""
    return (
        hours("pm25", pm25, 24, day=day)
        + hours("no2", no2, 24, day=day)
        + hours("so2", so2, 24, day=day)
    )


def test_a_complete_day_is_published():
    rows = build_daily_aqi_rows(measurements(full_day()))

    assert len(rows) == 1
    assert rows[0].pollutant_count == 3
    assert rows[0].extrapolated is False
    assert rows[0].aqi > 0


def test_a_single_low_pollutant_is_not_published():
    """The AQI of 2 in the daily table.

    One pollutant, averaged alone, used to become the day's AQI.
    """
    rows = build_daily_aqi_rows(measurements(hours("so2", 1.5, 24)))

    assert rows == []


def test_two_pollutants_are_not_enough():
    data = hours("pm25", 40.0, 24) + hours("no2", 30.0, 24)

    assert build_daily_aqi_rows(measurements(data)) == []


def test_three_gases_without_particulate_matter_are_not_enough():
    """At least one of the three must be PM2.5 or PM10."""
    data = hours("no2", 30.0, 24) + hours("so2", 20.0, 24) + hours("o3", 25.0, 24)

    assert build_daily_aqi_rows(measurements(data)) == []


def test_pm10_satisfies_the_particulate_matter_requirement():
    data = hours("pm10", 80.0, 24) + hours("no2", 30.0, 24) + hours("so2", 20.0, 24)

    rows = build_daily_aqi_rows(measurements(data))

    assert len(rows) == 1


def test_a_day_below_the_minimum_hours_is_not_published():
    """A thin day used to pass as a 24 hour mean.

    Three pollutants are present, so only the coverage rule can reject this.
    """
    short = MINIMUM_HOURS_FOR_DAILY_MEAN - 1
    data = (
        hours("pm25", 40.0, short)
        + hours("no2", 30.0, short)
        + hours("so2", 20.0, short)
    )

    assert build_daily_aqi_rows(measurements(data)) == []


def test_exactly_the_minimum_hours_is_published():
    data = (
        hours("pm25", 40.0, MINIMUM_HOURS_FOR_DAILY_MEAN)
        + hours("no2", 30.0, MINIMUM_HOURS_FOR_DAILY_MEAN)
        + hours("so2", 20.0, MINIMUM_HOURS_FOR_DAILY_MEAN)
    )

    rows = build_daily_aqi_rows(measurements(data))

    assert len(rows) == 1
    assert rows[0].pollutant_count == 3


def test_a_thin_pollutant_is_dropped_without_sinking_the_day():
    """Coverage is per pollutant, and dropping one can cost the day its AQI."""
    data = (
        hours("pm25", 40.0, 24)
        + hours("no2", 30.0, 24)
        + hours("so2", 20.0, 3)
    )

    rows = build_daily_aqi_rows(measurements(data))

    assert rows == []


def test_several_sensors_in_one_hour_count_as_one_hour():
    """Two sensors reporting for eight hours are not sixteen hours of data.

    Averaging within the hour first is what keeps the coverage rule honest.
    """
    data = []
    for sensor_value in (38.0, 42.0):
        data += hours("pm25", sensor_value, 8)
        data += hours("no2", sensor_value, 8)
        data += hours("so2", sensor_value, 8)

    assert build_daily_aqi_rows(measurements(data)) == []


def test_a_single_reading_is_not_an_eight_hour_mean():
    """CO and O3 need a complete window, not one observation."""
    data = full_day() + [(f"{DAY}T12:00:00Z", "co", 30.0)]

    rows = build_daily_aqi_rows(measurements(data))

    assert len(rows) == 1
    assert rows[0].pollutant_count == 3
    assert rows[0].dominant_pollutant != "co"


def test_a_complete_eight_hour_window_is_used():
    data = full_day() + hours("co", 30.0, SHORT_TERM_WINDOW_HOURS, start=8)

    rows = build_daily_aqi_rows(measurements(data))

    assert len(rows) == 1
    assert rows[0].pollutant_count == 4
    assert rows[0].dominant_pollutant == "co"


def test_an_eight_hour_window_may_cross_midnight():
    """Rolling per calendar day threw away every window spanning midnight.

    CO sits at a harmless level all through the first day and spikes in the
    four hours either side of midnight. The window ending at 03:00 on the
    second day is the highest, and it only exists if the roll is continuous.
    """
    data = full_day() + full_day(day=NEXT_DAY)
    data += hours("co", 1.0, 20, day=DAY)
    data += hours("co", 30.0, 4, day=DAY, start=20)
    data += hours("co", 30.0, 4, day=NEXT_DAY)
    data += hours("co", 1.0, 20, day=NEXT_DAY, start=4)

    rows = {row.day.isoformat(): row for row in build_daily_aqi_rows(measurements(data))}

    assert rows[NEXT_DAY].dominant_pollutant == "co"
    assert rows[NEXT_DAY].pollutant_count == 4


def test_extrapolation_is_reported_on_the_day_it_happens():
    """A reading past the highest published breakpoint marks the row."""
    quiet = build_daily_aqi_rows(measurements(full_day()))
    extreme = build_daily_aqi_rows(measurements(full_day(pm25=900.0)))

    assert quiet[0].extrapolated is False
    assert extreme[0].extrapolated is True
    assert extreme[0].aqi > 500


def test_high_readings_inside_the_published_bands_are_not_flagged():
    rows = build_daily_aqi_rows(measurements(full_day(pm25=200.0)))

    assert rows[0].extrapolated is False
    assert 300 < rows[0].aqi < 400


def test_empty_input_produces_no_rows():
    empty = pd.DataFrame(columns=["ts", "pollutant", "value"])

    assert build_daily_aqi_rows(empty) == []


def test_missing_values_are_ignored():
    data = full_day() + [(f"{DAY}T05:00:00Z", "pm10", None)]

    rows = build_daily_aqi_rows(measurements(data))

    assert len(rows) == 1
    assert rows[0].pollutant_count == 3


@pytest.mark.parametrize("pollutant", ["pm25", "pm10"])
def test_each_particulate_alone_with_two_gases_is_enough(pollutant):
    data = (
        hours(pollutant, 50.0, 24)
        + hours("no2", 30.0, 24)
        + hours("so2", 20.0, 24)
    )

    assert len(build_daily_aqi_rows(measurements(data))) == 1


class FakeConnection:
    def __init__(self, log):
        self.log = log

    def execute(self, query, params=None):
        self.log.append((str(query).strip().split()[0].upper(), params))


class FakeEngine:
    """Records the statements a write path issues, in order."""

    def __init__(self):
        self.log = []

    def begin(self):
        log = self.log

        class Transaction:
            def __enter__(self):
                return FakeConnection(log)

            def __exit__(self, *exc):
                return False

        return Transaction()


def test_storing_a_station_deletes_its_old_rows_first():
    """The stale-row bug.

    Applying the published rules made some days stop qualifying. An upsert
    cannot remove them, so 1,159 rows survived carrying the very AQI of 2 the
    rules were meant to remove. The delete has to come first, and it has to
    happen even when there is nothing to insert.
    """
    engine = FakeEngine()

    written = replace_station_daily_aqi(
        engine,
        station_id=4663956,
        records=[
            DailyAqiRecord(
                day=date(2026, 8, 7),
                aqi=92,
                dominant_pollutant="pm25",
                pollutant_count=4,
                extrapolated=False,
            )
        ],
    )

    verbs = [verb for verb, _ in engine.log]

    assert written == 1
    assert verbs == ["DELETE", "INSERT"]


def test_a_station_with_no_qualifying_days_is_emptied():
    """The exact shape of the bug.

    GK1 (Oberoi Terrace) reports PM2.5 alone, so under the published rules it
    has no publishable days at all. The old path wrote nothing and therefore
    removed nothing, leaving every one of its pre-rules rows in place.
    """
    engine = FakeEngine()

    written = replace_station_daily_aqi(engine, station_id=1, records=[])

    assert written == 0
    assert [verb for verb, _ in engine.log] == ["DELETE"]


def test_a_single_pollutant_station_produces_no_rows_to_store():
    """PM2.5 alone is two pollutants short, whatever its coverage."""
    data = hours("pm25", 1.5, 24)

    assert build_daily_aqi_rows(measurements(data)) == []

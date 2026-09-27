"""Daily station AQI, following the published CPCB aggregation rules.

Three rules govern whether a day gets an AQI at all, and they are the reason
this module is stricter than a plain groupby-mean:

1.  A 24-hour sub-index needs at least 16 hourly observations in the day.
2.  An 8-hour sub-index needs a complete 8-hour window, and that window may
    cross midnight.
3.  An overall AQI needs at least three pollutants, one of which must be
    PM2.5 or PM10.

Without rule 1 a single clean hourly reading becomes a "daily mean". Without
rule 3 a day carrying only a low SO2 reading is published as AQI 2. Both
produced the implausible low end that this module previously reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from air_quality_intelligence.analysis.aqi import (
    MINIMUM_HOURS_FOR_SUB_INDEX,
    InsufficientPollutantsError,
    calculate_aqi_detail,
)
from air_quality_intelligence.db.daily_aqi import (
    DailyAqiRecord,
    replace_station_daily_aqi,
)
from air_quality_intelligence.transform.units import (
    CANONICAL_UNITS,
    UnitConversionError,
    normalize_concentration,
)
from air_quality_intelligence.transform.validation import check_plausible

LONG_TERM_POLLUTANTS = ("pm25", "pm10", "no2", "so2")
SHORT_TERM_POLLUTANTS = ("co", "o3")

# Hours in the averaging window for the 8-hourly pollutants.
SHORT_TERM_WINDOW_HOURS = 8

# Hours that must be present in a calendar day before its 24-hour mean is
# usable. This is the published minimum.
MINIMUM_HOURS_FOR_DAILY_MEAN = MINIMUM_HOURS_FOR_SUB_INDEX


@dataclass(frozen=True)
class DailyAqiRow:
    """One station-day AQI and the evidence behind it."""

    day: date
    aqi: int
    dominant_pollutant: str
    pollutant_count: int
    extrapolated: bool


def _normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every reading to its canonical AQI unit.

    Stations expose the same pollutant through more than one sensor, and those
    sensors do not agree on units: one reports ug/m3 and another ppb. Averaging
    the raw values, as this module used to, mixes the two scales and produces a
    concentration that belongs to neither. Rows whose unit cannot be converted
    are dropped with the reason attached rather than silently averaged in.
    """

    work = df.copy()

    if "unit" not in work.columns:
        raise ValueError(
            "Measurements must carry a unit column; averaging raw values "
            "across sensors with different units is not meaningful."
        )

    values: list[float] = []
    keep: list[bool] = []

    for row in work.itertuples(index=False):
        try:
            converted, _ = normalize_concentration(
                pollutant=row.pollutant,
                value=row.value,
                unit=row.unit,
            )
            check_plausible(str(row.pollutant).strip().lower(), converted)
        except (UnitConversionError, ValueError):
            values.append(float("nan"))
            keep.append(False)
            continue

        values.append(converted)
        keep.append(True)

    work["value"] = values
    work["unit"] = [
        CANONICAL_UNITS.get(str(pollutant).strip().lower())
        for pollutant in work["pollutant"]
    ]

    return work[pd.Series(keep, index=work.index)]


def _to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse normalized measurements onto whole hours.

    Several sensors at one station can report the same pollutant within an
    hour. Averaging them first means the hour counts once, so a busy hour
    cannot stand in for the 16 hours the daily mean requires.
    """

    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True).dt.floor("h")
    work["pollutant"] = work["pollutant"].astype(str).str.strip().str.lower()
    work = work.dropna(subset=["value"])

    return (
        work.groupby(["pollutant", "ts"], as_index=False)["value"]
        .mean()
        .sort_values(["pollutant", "ts"])
        .reset_index(drop=True)
    )


def _daily_means_24h(hourly: pd.DataFrame) -> pd.DataFrame:
    """Calendar-day means for the 24-hourly pollutants, coverage gated."""

    subset = hourly[hourly["pollutant"].isin(LONG_TERM_POLLUTANTS)]

    if subset.empty:
        return pd.DataFrame(columns=["date", "pollutant", "value"])

    grouped = (
        subset.assign(date=subset["ts"].dt.date)
        .groupby(["date", "pollutant"])["value"]
        .agg(["mean", "count"])
        .reset_index()
    )

    kept = grouped[grouped["count"] >= MINIMUM_HOURS_FOR_DAILY_MEAN]

    return kept.rename(columns={"mean": "value"})[["date", "pollutant", "value"]]


def _daily_maxima_8h(hourly: pd.DataFrame) -> pd.DataFrame:
    """Highest complete 8-hour mean per day for the 8-hourly pollutants.

    The rolling window runs over the whole continuous hourly series, so a
    window that starts late one evening and ends after midnight is counted.
    Grouping by calendar day before rolling, as an earlier version did,
    silently discarded every window that crossed midnight and let a single
    reading pass as an 8-hour mean.

    Each window is attributed to the day of its ending hour.
    """

    subset = hourly[hourly["pollutant"].isin(SHORT_TERM_POLLUTANTS)]

    if subset.empty:
        return pd.DataFrame(columns=["date", "pollutant", "value"])

    frames: list[pd.DataFrame] = []

    for pollutant, group in subset.groupby("pollutant"):
        series = group.set_index("ts")["value"].sort_index()

        full_index = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq="h",
            tz=series.index.tz,
        )

        rolling = (
            series.reindex(full_index)
            .rolling(
                window=SHORT_TERM_WINDOW_HOURS,
                min_periods=SHORT_TERM_WINDOW_HOURS,
            )
            .mean()
            .dropna()
        )

        if rolling.empty:
            continue

        windows = pd.DataFrame(
            {
                "ts": rolling.index,
                "value": rolling.to_numpy(),
            }
        )

        daily = (
            windows.assign(date=windows["ts"].dt.date)
            .groupby("date", as_index=False)["value"]
            .max()
        )

        daily["pollutant"] = pollutant
        frames.append(daily[["date", "pollutant", "value"]])

    if not frames:
        return pd.DataFrame(columns=["date", "pollutant", "value"])

    return pd.concat(frames, ignore_index=True)


def build_daily_concentrations(df: pd.DataFrame) -> pd.DataFrame:
    """One station's daily pollutant concentrations, after every rule.

    Returns columns date, pollutant and value, with the 24-hourly pollutants
    coverage-gated and the 8-hourly ones taken as the day's highest complete
    window. Exposed separately so the AQI behind any station-day can be
    inspected without reimplementing the aggregation.
    """

    if df.empty:
        return pd.DataFrame(columns=["date", "pollutant", "value"])

    hourly = _to_hourly(_normalize_units(df))

    if hourly.empty:
        return pd.DataFrame(columns=["date", "pollutant", "value"])

    # Concatenating an empty frame makes pandas guess at dtypes and warn, so
    # the empty halves are dropped first. A station with no 8-hourly
    # pollutants is ordinary, not an error.
    parts = [
        part
        for part in (_daily_means_24h(hourly), _daily_maxima_8h(hourly))
        if not part.empty
    ]

    if not parts:
        return pd.DataFrame(columns=["date", "pollutant", "value"])

    return pd.concat(parts, ignore_index=True)


def build_daily_aqi_rows(df: pd.DataFrame) -> list[DailyAqiRow]:
    """Turn one station's raw measurements into publishable daily AQI rows.

    Days that fail any published rule are left out rather than published
    with a value derived from too little data.
    """

    daily = build_daily_concentrations(df)

    if daily.empty:
        return []

    rows: list[DailyAqiRow] = []

    for day, group in daily.groupby("date", sort=True):
        concentrations = {
            row.pollutant: float(row.value)
            for row in group.itertuples()
            if pd.notna(row.value)
        }

        if not concentrations:
            continue

        try:
            dominant, details = calculate_aqi_detail(concentrations)
        except InsufficientPollutantsError:
            continue

        rows.append(
            DailyAqiRow(
                day=day,
                aqi=dominant.value,
                dominant_pollutant=dominant.pollutant,
                pollutant_count=len(details),
                extrapolated=any(
                    detail.extrapolated for detail in details.values()
                ),
            )
        )

    return rows


def calculate_daily_aqi_for_station(
    engine: Engine,
    station_id: int,
) -> int:
    """Calculate and store the CPCB daily AQI for one monitoring station."""

    query = text(
        """
        SELECT ts, pollutant, value, unit
        FROM measurements
        WHERE station_id = :station_id
          AND pollutant IN (
              'pm25', 'pm10', 'no2', 'so2', 'co', 'o3'
          )
        ORDER BY ts
        """
    )

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"station_id": station_id},
        )

    rows = build_daily_aqi_rows(df)

    # Replace rather than upsert. A day that stops qualifying under the
    # published rules has to disappear from the table, and an upsert can only
    # add or overwrite.
    return replace_station_daily_aqi(
        engine=engine,
        station_id=station_id,
        records=[
            DailyAqiRecord(
                day=row.day,
                aqi=row.aqi,
                dominant_pollutant=row.dominant_pollutant,
                pollutant_count=row.pollutant_count,
                extrapolated=row.extrapolated,
            )
            for row in rows
        ],
    )

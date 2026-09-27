"""Show what is actually behind the highest and lowest daily AQI values.

The published rules stopped the daily table reporting an AQI from one
pollutant or from a handful of hours, and the minimum-pollutant count now
reads 3 everywhere. The extremes did not move: Delhi still reports a daily
minimum of 2 and Bengaluru a maximum of 607.

So the remaining question is not about the rules. It is whether the
concentrations feeding them are real. This script prints the pollutant
concentrations behind each extreme station-day, with the hours behind each
one, so the answer comes from the data instead of from a guess.

    python scripts/diagnose_aqi_extremes.py
    python scripts/diagnose_aqi_extremes.py --rows 15
"""

from __future__ import annotations

import argparse

import pandas as pd
from sqlalchemy import text

from air_quality_intelligence.analysis.aqi import calculate_sub_index_detail
from air_quality_intelligence.analysis.daily_aqi import (
    build_daily_concentrations,
)
from air_quality_intelligence.db.engine import get_engine

MEASUREMENTS = text(
    """
    SELECT ts, pollutant, value, unit
    FROM measurements
    WHERE station_id = :station_id
      AND pollutant IN ('pm25', 'pm10', 'no2', 'so2', 'co', 'o3')
    ORDER BY ts
    """
)

EXTREMES = text(
    """
    (SELECT d.station_id, s.name, s.city, d.date, d.aqi,
            d.dominant_pollutant, d.pollutant_count, d.extrapolated
     FROM daily_aqi d JOIN stations s ON s.station_id = d.station_id
     ORDER BY d.aqi ASC LIMIT :rows)
    UNION ALL
    (SELECT d.station_id, s.name, s.city, d.date, d.aqi,
            d.dominant_pollutant, d.pollutant_count, d.extrapolated
     FROM daily_aqi d JOIN stations s ON s.station_id = d.station_id
     ORDER BY d.aqi DESC LIMIT :rows)
    """
)

NEAR_ZERO = text(
    """
    SELECT s.city, m.station_id, s.name, m.pollutant, m.unit,
           COUNT(*) AS readings,
           COUNT(*) FILTER (WHERE m.value <= 0) AS zero_readings,
           ROUND(MIN(m.value)::numeric, 4) AS min_value,
           ROUND((percentile_cont(0.5) WITHIN GROUP (ORDER BY m.value))::numeric, 4)
               AS median_value,
           ROUND(MAX(m.value)::numeric, 2) AS max_value
    FROM measurements m
    JOIN stations s ON s.station_id = m.station_id
    WHERE m.pollutant IN ('pm25', 'pm10', 'no2', 'so2', 'co', 'o3')
    GROUP BY s.city, m.station_id, s.name, m.pollutant, m.unit
    HAVING COUNT(*) FILTER (WHERE m.value <= 0) > 0
    ORDER BY
        COUNT(*) FILTER (WHERE m.value <= 0)::float / COUNT(*) DESC
    LIMIT 25
    """
)

PEER_OUTLIERS = text(
    """
    WITH daily AS (
        SELECT s.city, m.station_id, s.name, m.pollutant,
               DATE(m.ts) AS day,
               AVG(m.value) AS value
        FROM measurements m
        JOIN stations s ON s.station_id = m.station_id
        WHERE m.pollutant IN ('pm25', 'pm10', 'no2', 'so2', 'o3')
          AND m.value > 0
          -- PostgreSQL orders NaN above every real number and propagates it
          -- through AVG, so without this every row here was a NaN reading
          -- rather than an outlier.
          AND m.value <> 'NaN'::double precision
          AND m.value < 'Infinity'::double precision
        GROUP BY s.city, m.station_id, s.name, m.pollutant, DATE(m.ts)
    ),
    peers AS (
        SELECT city, pollutant, day,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS city_median,
               COUNT(*) AS stations
        FROM daily
        GROUP BY city, pollutant, day
    )
    SELECT d.city, d.station_id, d.name, d.pollutant, d.day,
           ROUND(d.value::numeric, 1) AS station_value,
           ROUND(p.city_median::numeric, 1) AS city_median,
           p.stations AS peer_stations,
           ROUND((d.value / NULLIF(p.city_median, 0))::numeric, 1) AS times_median
    FROM daily d
    JOIN peers p
      ON p.city = d.city AND p.pollutant = d.pollutant AND p.day = d.day
    WHERE p.stations >= 5
      AND p.city_median > 0
      AND d.value / p.city_median >= 5
    ORDER BY d.value / p.city_median DESC
    LIMIT 20
    """
)

NON_FINITE = text(
    """
    SELECT s.city, m.pollutant,
           COUNT(*) FILTER (WHERE m.value = 'NaN'::double precision) AS nan_rows,
           COUNT(*) FILTER (
               WHERE m.value = 'Infinity'::double precision
                  OR m.value = '-Infinity'::double precision
           ) AS infinite_rows,
           COUNT(*) AS total_rows
    FROM measurements m
    JOIN stations s ON s.station_id = m.station_id
    GROUP BY s.city, m.pollutant
    HAVING COUNT(*) FILTER (
        WHERE m.value = 'NaN'::double precision
           OR m.value = 'Infinity'::double precision
           OR m.value = '-Infinity'::double precision
    ) > 0
    ORDER BY s.city, m.pollutant
    """
)

DAY_COVERAGE = text(
    """
    WITH measured AS (
        SELECT s.city, DATE(m.ts) AS day, COUNT(*) AS readings
        FROM measurements m
        JOIN stations s ON s.station_id = m.station_id
        GROUP BY s.city, DATE(m.ts)
    ),
    published AS (
        SELECT s.city, d.date AS day, COUNT(*) AS stations
        FROM daily_aqi d
        JOIN stations s ON s.station_id = d.station_id
        GROUP BY s.city, d.date
    )
    SELECT m.city, m.day, m.readings,
           COALESCE(p.stations, 0) AS stations_published
    FROM measured m
    LEFT JOIN published p ON p.city = m.city AND p.day = m.day
    WHERE COALESCE(p.stations, 0) = 0
    ORDER BY m.city, m.day
    """
)

CO_FEED = text(
    """
    SELECT s.city, m.unit, COUNT(DISTINCT m.station_id) AS stations,
           COUNT(*) AS readings,
           ROUND(MIN(m.value)::numeric, 4) AS min_value,
           ROUND((percentile_cont(0.5) WITHIN GROUP (ORDER BY m.value))::numeric, 4)
               AS median_value,
           ROUND(MAX(m.value)::numeric, 3) AS max_value
    FROM measurements m
    JOIN stations s ON s.station_id = m.station_id
    WHERE m.pollutant = 'co'
    GROUP BY s.city, m.unit
    ORDER BY s.city, m.unit
    """
)


def show(title: str, frame: pd.DataFrame) -> None:
    print()
    print(f"=== {title} ===")

    if frame.empty:
        print("  (nothing)")
        return

    print(frame.to_string(index=False))


def explain_station_day(engine, row) -> pd.DataFrame:
    """The daily concentrations and sub-indices behind one station-day."""

    with engine.connect() as connection:
        raw = pd.read_sql(
            MEASUREMENTS, connection, params={"station_id": int(row.station_id)}
        )

    daily = build_daily_concentrations(raw)

    if daily.empty:
        return pd.DataFrame()

    day = daily[daily["date"] == row.date].copy()

    if day.empty:
        return pd.DataFrame()

    details = [calculate_sub_index_detail(p, v) for p, v in
               zip(day["pollutant"], day["value"], strict=True)]

    day["sub_index"] = [d.value if d else None for d in details]
    day["extrapolated"] = [d.extrapolated if d else None for d in details]
    day["value"] = day["value"].round(3)

    return day[["pollutant", "value", "sub_index", "extrapolated"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8,
                        help="How many station-days from each end.")
    args = parser.parse_args()

    engine = get_engine()

    with engine.connect() as connection:
        extremes = pd.read_sql(EXTREMES, connection, params={"rows": args.rows})
        near_zero = pd.read_sql(NEAR_ZERO, connection)
        co_feed = pd.read_sql(CO_FEED, connection)
        peer_outliers = pd.read_sql(PEER_OUTLIERS, connection)
        non_finite = pd.read_sql(NON_FINITE, connection)
        day_coverage = pd.read_sql(DAY_COVERAGE, connection)

    show("Readings stored as NaN or Infinity", non_finite)
    print()
    print("  DOUBLE PRECISION accepts these and NOT NULL does not stop them.")
    print("  Nothing downstream uses them, but they poison SQL aggregates.")
    print("  To clear them out, see the note in sql/schema.sql.")

    show("Days with measurements but no published AQI", day_coverage)
    print()
    print("  A day drops out when no station in that city reached three")
    print("  pollutants with sixteen hours each. Edge days of the ingestion")
    print("  window are expected here; a day in the middle is not.")

    show("CO feed, by city and reported unit", co_feed)
    print()
    print("  Readings labelled ppb are corrected to ppm before conversion.")
    print("  As ppm, a median near 0.5 is ordinary urban air and a median")
    print("  near 0.03 is below the global atmospheric background.")

    show(
        "Station-days at least five times their city's median, same pollutant",
        peer_outliers,
    )
    print()
    print("  A genuine episode moves a whole city. One station many times its")
    print("  peers on the same day and pollutant is the station, not the air.")
    print("  This is a report, not a filter: nothing here has been excluded")
    print("  from any result.")

    show("Sensor feeds reporting non-positive values", near_zero)
    print()
    print("  CPCB treats a reading of zero or below as missing, not as clean")
    print("  air. A feed that is mostly zero is an instrument that stopped.")

    print()
    print("=== Station-days behind the extremes ===")

    for row in extremes.itertuples():
        print()
        print(
            f"  {row.city} / {row.name} / {row.date}  "
            f"AQI {row.aqi} on {row.dominant_pollutant}, "
            f"{row.pollutant_count} pollutants, extrapolated={row.extrapolated}"
        )

        detail = explain_station_day(engine, row)

        if detail.empty:
            print("    (could not reconstruct this day)")
            continue

        for line in detail.to_string(index=False).splitlines():
            print(f"    {line}")


if __name__ == "__main__":
    main()

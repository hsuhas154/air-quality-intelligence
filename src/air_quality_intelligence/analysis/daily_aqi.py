from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from air_quality_intelligence.analysis.aqi import calculate_aqi
from air_quality_intelligence.db.daily_aqi import insert_daily_aqi


LONG_TERM_POLLUTANTS = {"pm25", "pm10", "no2", "so2"}
SHORT_TERM_POLLUTANTS = {"co", "o3"}


def calculate_daily_aqi_for_station(
    engine: Engine,
    station_id: int,
) -> int:
    """Calculate CPCB-style daily AQI for one monitoring station."""

    query = text(
        """
        SELECT ts, pollutant, value
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

    if df.empty:
        return 0

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.date

    daily_results = []

    # PM10, PM2.5, NO2 and SO2:
    # use daily 24-hour mean concentrations.
    long_term = df[df["pollutant"].isin(LONG_TERM_POLLUTANTS)]

    if not long_term.empty:
        daily_long_term = (
            long_term
            .groupby(["date", "pollutant"])["value"]
            .mean()
            .reset_index()
        )
        daily_results.append(daily_long_term)

    # CO and O3:
    # calculate rolling 8-hour means and use the maximum
    # 8-hour mean within each day.
    short_term = df[df["pollutant"].isin(SHORT_TERM_POLLUTANTS)].copy()

    if not short_term.empty:
        short_term = short_term.sort_values("ts")

        rolling_frames = []

        for (day, pollutant), group in short_term.groupby(
            ["date", "pollutant"]
        ):
            group = group.sort_values("ts").set_index("ts")

            rolling = (
                group["value"]
                .rolling("8h", min_periods=1)
                .mean()
                .max()
            )

            rolling_frames.append(
                {
                    "date": day,
                    "pollutant": pollutant,
                    "value": rolling,
                }
            )

        if rolling_frames:
            daily_results.append(pd.DataFrame(rolling_frames))

    if not daily_results:
        return 0

    daily = pd.concat(daily_results, ignore_index=True)

    inserted = 0

    for day, group in daily.groupby("date"):
        concentrations = {
            row["pollutant"]: float(row["value"])
            for _, row in group.iterrows()
            if pd.notna(row["value"])
        }

        if not concentrations:
            continue

        aqi, dominant_pollutant = calculate_aqi(concentrations)

        insert_daily_aqi(
            engine=engine,
            station_id=station_id,
            day=day,
            aqi=aqi,
            dominant_pollutant=dominant_pollutant,
        )

        inserted += 1

    return inserted

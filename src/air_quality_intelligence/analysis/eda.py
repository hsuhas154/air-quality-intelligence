from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def load_daily_aqi(engine: Engine) -> pd.DataFrame:
    """Load daily AQI with station and city metadata."""

    query = text(
        """
        SELECT
            s.station_id,
            s.name AS station_name,
            s.city,
            s.latitude,
            s.longitude,
            d.date,
            d.aqi,
            d.dominant_pollutant
        FROM daily_aqi d
        JOIN stations s
            ON s.station_id = d.station_id
        ORDER BY d.date, s.city, s.station_id
        """
    )

    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def city_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return city-level AQI summary statistics."""

    return (
        df.groupby("city")
        .agg(
            stations=("station_id", "nunique"),
            days=("date", "nunique"),
            mean_aqi=("aqi", "mean"),
            median_aqi=("aqi", "median"),
            min_aqi=("aqi", "min"),
            max_aqi=("aqi", "max"),
        )
        .round(1)
        .reset_index()
    )


def pollutant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return dominant-pollutant frequencies by city."""

    return (
        pd.crosstab(
            df["city"],
            df["dominant_pollutant"],
            normalize="index",
        )
        .mul(100)
        .round(1)
    )

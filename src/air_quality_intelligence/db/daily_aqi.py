from __future__ import annotations

from datetime import date

from sqlalchemy import text
from sqlalchemy.engine import Engine


def insert_daily_aqi(
    engine: Engine,
    station_id: int,
    day: date,
    aqi: int,
    dominant_pollutant: str,
) -> None:
    """Insert or update a station's daily AQI."""

    query = text(
        """
        INSERT INTO daily_aqi (
            station_id,
            date,
            aqi,
            dominant_pollutant
        )
        VALUES (
            :station_id,
            :date,
            :aqi,
            :dominant_pollutant
        )
        ON CONFLICT (station_id, date)
        DO UPDATE SET
            aqi = EXCLUDED.aqi,
            dominant_pollutant = EXCLUDED.dominant_pollutant
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "station_id": station_id,
                "date": day,
                "aqi": aqi,
                "dominant_pollutant": dominant_pollutant,
            },
        )

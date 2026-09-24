from sqlalchemy import text
from sqlalchemy.engine import Engine
import pandas as pd


def insert_weather(
    engine: Engine,
    city: str,
    weather: pd.DataFrame,
) -> int:
    """Insert hourly weather observations for a city."""
    query = text(
        """
        INSERT INTO weather (
            city,
            ts,
            temp_c,
            humidity,
            wind_speed,
            wind_dir,
            blh
        )
        VALUES (
            :city,
            :ts,
            :temp_c,
            :humidity,
            :wind_speed,
            :wind_dir,
            :blh
        )
        ON CONFLICT (city, ts)
        DO UPDATE SET
            temp_c = EXCLUDED.temp_c,
            humidity = EXCLUDED.humidity,
            wind_speed = EXCLUDED.wind_speed,
            wind_dir = EXCLUDED.wind_dir,
            blh = EXCLUDED.blh
        """
    )

    records = weather.to_dict(orient="records")

    with engine.begin() as connection:
        for record in records:
            connection.execute(
                query,
                {
                    "city": city,
                    "ts": record["ts"].to_pydatetime(),
                    "temp_c": record["temp_c"],
                    "humidity": record["humidity"],
                    "wind_speed": record["wind_speed"],
                    "wind_dir": record["wind_dir"],
                    "blh": record["blh"],
                },
            )

    return len(records)

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def insert_measurements(engine: Engine, station_id: int, df: pd.DataFrame) -> int:
    """Insert OpenAQ measurements for a station, ignoring duplicates."""

    if df.empty:
        return 0

    rows = df.to_dict(orient="records")

    query = text(
        """
        INSERT INTO measurements (
            station_id,
            ts,
            pollutant,
            value,
            unit,
            source
        )
        VALUES (
            :station_id,
            :ts,
            :pollutant,
            :value,
            :unit,
            :source
        )
        ON CONFLICT (station_id, ts, pollutant, source)
        DO NOTHING
        """
    )

    with engine.begin() as connection:
        inserted = 0

        for row in rows:
            result = connection.execute(
                query,
                {
                    "station_id": station_id,
                    "ts": row["ts"],
                    "pollutant": row["pollutant"],
                    "value": row["value"],
                    "unit": row["unit"],
                    "source": row["source"],
                },
            )
            inserted += result.rowcount

    return inserted

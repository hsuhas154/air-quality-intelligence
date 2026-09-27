from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def drop_non_finite(df: pd.DataFrame) -> pd.DataFrame:
    """Remove readings that are NULL, NaN or infinite.

    DOUBLE PRECISION accepts NaN, so NOT NULL does not keep it out, and the
    provider does send them. Every consumer already discards them, which is
    why they never reached an AQI. They do reach SQL: an AVG over a group
    holding one NaN is NaN for the whole group, and PostgreSQL orders NaN
    above every real number, so a NaN reading takes the top of any query
    ranked by value. The station-outlier report was filled with them.

    A reading that is not a number is not a measurement, so it is dropped at
    the boundary rather than guarded against everywhere downstream.
    """

    if df.empty or "value" not in df.columns:
        return df

    values = pd.to_numeric(df["value"], errors="coerce")

    return df[np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))]


def insert_measurements(engine: Engine, station_id: int, df: pd.DataFrame) -> int:
    """Insert OpenAQ measurements for a station, ignoring duplicates."""

    if df.empty:
        return 0

    df = drop_non_finite(df)

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

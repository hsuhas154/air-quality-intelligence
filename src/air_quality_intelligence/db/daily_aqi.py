from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import NamedTuple

from sqlalchemy import text
from sqlalchemy.engine import Engine


class DailyAqiRecord(NamedTuple):
    """One station-day, ready to store."""

    day: date
    aqi: int
    dominant_pollutant: str
    pollutant_count: int
    extrapolated: bool


def replace_station_daily_aqi(
    engine: Engine,
    station_id: int,
    records: Sequence[DailyAqiRecord],
) -> int:
    """Replace a station's daily AQI rows with exactly these ones.

    Every row for the station is deleted first, so the table is a pure
    function of the measurements it was computed from.

    An upsert alone is not enough, and the difference was not academic. When
    the published CPCB rules were applied, days that had previously been
    stored from a single pollutant stopped qualifying. The upsert simply did
    not write them, and the old rows stayed: 1,159 of them, still carrying
    the AQI of 2 the rules were meant to remove, with NULL in the columns
    added alongside those rules. The minimum-pollutant summary read 3
    because SQL MIN skips NULLs, so the stale rows were invisible in the one
    place they would have been noticed.

    The delete and the inserts share one transaction. A station ends up
    either with the rows its measurements support or with the rows it had
    before, never a mix of the two.

    pollutant_count and extrapolated are stored so that a published value
    can be audited from SQL alone: how many pollutants stood behind it, and
    whether any sub-index came from continuing the top band's line past the
    highest published breakpoint.
    """

    delete_query = text(
        """
        DELETE FROM daily_aqi
        WHERE station_id = :station_id
        """
    )

    insert_query = text(
        """
        INSERT INTO daily_aqi (
            station_id,
            date,
            aqi,
            dominant_pollutant,
            pollutant_count,
            extrapolated
        )
        VALUES (
            :station_id,
            :date,
            :aqi,
            :dominant_pollutant,
            :pollutant_count,
            :extrapolated
        )
        ON CONFLICT (station_id, date)
        DO UPDATE SET
            aqi = EXCLUDED.aqi,
            dominant_pollutant = EXCLUDED.dominant_pollutant,
            pollutant_count = EXCLUDED.pollutant_count,
            extrapolated = EXCLUDED.extrapolated
        """
    )

    with engine.begin() as connection:
        connection.execute(delete_query, {"station_id": station_id})

        for record in records:
            connection.execute(
                insert_query,
                {
                    "station_id": station_id,
                    "date": record.day,
                    "aqi": record.aqi,
                    "dominant_pollutant": record.dominant_pollutant,
                    "pollutant_count": record.pollutant_count,
                    "extrapolated": record.extrapolated,
                },
            )

    return len(records)


def insert_daily_aqi(
    engine: Engine,
    station_id: int,
    day: date,
    aqi: int,
    dominant_pollutant: str,
    pollutant_count: int,
    extrapolated: bool,
) -> None:
    """Insert or update a single station-day.

    Kept for callers that write one row at a time. Anything recomputing a
    station's whole history must use replace_station_daily_aqi instead: this
    function cannot remove a row that no longer qualifies.
    """

    replace_query = text(
        """
        INSERT INTO daily_aqi (
            station_id,
            date,
            aqi,
            dominant_pollutant,
            pollutant_count,
            extrapolated
        )
        VALUES (
            :station_id,
            :date,
            :aqi,
            :dominant_pollutant,
            :pollutant_count,
            :extrapolated
        )
        ON CONFLICT (station_id, date)
        DO UPDATE SET
            aqi = EXCLUDED.aqi,
            dominant_pollutant = EXCLUDED.dominant_pollutant,
            pollutant_count = EXCLUDED.pollutant_count,
            extrapolated = EXCLUDED.extrapolated
        """
    )

    with engine.begin() as connection:
        connection.execute(
            replace_query,
            {
                "station_id": station_id,
                "date": day,
                "aqi": aqi,
                "dominant_pollutant": dominant_pollutant,
                "pollutant_count": pollutant_count,
                "extrapolated": extrapolated,
            },
        )

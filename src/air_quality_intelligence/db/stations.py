from sqlalchemy import text
from sqlalchemy.engine import Engine


def insert_station(
    engine: Engine,
    station_id: int,
    name: str,
    city: str,
    state: str | None,
    latitude: float,
    longitude: float,
) -> None:
    """Insert a monitoring station into the stations table."""
    query = text(
        """
        INSERT INTO stations (
            station_id,
            name,
            city,
            state,
            latitude,
            longitude,
            geom
        )
        VALUES (
            :station_id,
            :name,
            :city,
            :state,
            :latitude,
            :longitude,
            ST_SetSRID(
                ST_MakePoint(:longitude, :latitude),
                4326
            )
        )
        ON CONFLICT (station_id) DO NOTHING
        """
    )

    with engine.begin() as connection:
        connection.execute(
            query,
            {
                "station_id": station_id,
                "name": name,
                "city": city,
                "state": state,
                "latitude": latitude,
                "longitude": longitude,
            },
        )

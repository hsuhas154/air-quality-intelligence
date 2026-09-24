from air_quality_intelligence.db.engine import get_engine
from air_quality_intelligence.db.stations import insert_station
from air_quality_intelligence.ingestion.openaq import fetch_location


def register_location(location_id: int, city: str, state: str | None = None) -> None:
    """Fetch an OpenAQ location and register it as a database station."""

    location = fetch_location(location_id)

    coordinates = location.get("coordinates") or {}

    latitude = coordinates.get("latitude")
    longitude = coordinates.get("longitude")

    if latitude is None or longitude is None:
        raise ValueError(f"Location {location_id} has no coordinates.")

    insert_station(
        engine=get_engine(),
        station_id=location["id"],
        name=location["name"],
        city=city,
        state=state,
        latitude=latitude,
        longitude=longitude,
    )

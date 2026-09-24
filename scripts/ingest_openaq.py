import argparse
import os
from datetime import UTC, datetime, timedelta

import httpx

from air_quality_intelligence.config.settings import settings
from air_quality_intelligence.db.engine import get_engine
from air_quality_intelligence.db.measurements import insert_measurements
from air_quality_intelligence.db.stations import insert_station
from air_quality_intelligence.ingestion.openaq import fetch_sensor_measurements

TARGET_CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
}


def fetch_city_locations(city: str, latitude: float, longitude: float) -> list[dict]:
    """Find OpenAQ locations near a target city."""

    headers = {"X-API-Key": settings.openaq_api_key}

    with httpx.Client(timeout=30, headers=headers) as client:
        response = client.get(
            f"{settings.openaq_base_url}/locations",
            params={
                "coordinates": f"{latitude},{longitude}",
                "radius": 25000,
                "limit": 100,
            },
        )
        response.raise_for_status()

    return response.json().get("results", [])


DEFAULT_HISTORY_DAYS = int(os.environ.get("AQI_HISTORY_DAYS", "90"))


def main(history_days: int = DEFAULT_HISTORY_DAYS) -> None:
    if not settings.openaq_api_key:
        raise RuntimeError("OPENAQ_API_KEY is required.")

    engine = get_engine()

    end = datetime.now(UTC)
    start = end - timedelta(days=history_days)

    print(f"Ingestion window: {history_days} days ({start.date()} -> {end.date()})")

    total_stations = 0
    total_measurements = 0

    for city, (latitude, longitude) in TARGET_CITIES.items():
        print(f"\n=== {city} ===")

        locations = fetch_city_locations(city, latitude, longitude)
        print(f"Locations found: {len(locations)}")

        for location in locations:
            location_id = location["id"]
            location_name = location["name"]
            location_last = location.get("datetimeLast")

            if not location_last:
                continue

            coordinates = location.get("coordinates") or {}
            station_lat = coordinates.get("latitude")
            station_lon = coordinates.get("longitude")

            if station_lat is None or station_lon is None:
                continue

            insert_station(
                engine=engine,
                station_id=location_id,
                name=location_name,
                city=city,
                state=None,
                latitude=station_lat,
                longitude=station_lon,
            )

            total_stations += 1

            for sensor in location.get("sensors", []):
                sensor_id = sensor["id"]
                pollutant = sensor.get("parameter", {}).get("name")

                if pollutant not in {"pm25", "pm10", "no2", "so2", "o3", "co"}:
                    continue

                try:
                    measurements = fetch_sensor_measurements(
                        sensor_id=sensor_id,
                        start=start,
                        end=end,
                    )
                except Exception:
                    continue

                if measurements.empty:
                    continue

                inserted = insert_measurements(
                    engine=engine,
                    station_id=location_id,
                    df=measurements,
                )

                if inserted:
                    print(
                        f"  {location_name} | {pollutant} | "
                        f"{inserted} measurements"
                    )

                total_measurements += inserted

    print("\n=== INGESTION COMPLETE ===")
    print(f"Stations processed: {total_stations}")
    print(f"Measurements inserted: {total_measurements}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest OpenAQ measurements.")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="Days of history to request per sensor.",
    )
    main(parser.parse_args().days)

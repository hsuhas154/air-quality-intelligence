import argparse
import os
from datetime import date, timedelta

from air_quality_intelligence.config.cities import CITIES
from air_quality_intelligence.db.engine import get_engine
from air_quality_intelligence.db.weather import insert_weather
from air_quality_intelligence.ingestion.open_meteo import fetch_hourly_weather


DEFAULT_HISTORY_DAYS = int(os.environ.get("AQI_HISTORY_DAYS", "90"))

# The Open-Meteo forecast endpoint serves roughly 92 days of past data.
# Anything longer needs the archive endpoint, which this script does not use.
MAX_FORECAST_ENDPOINT_DAYS = 92


def main(history_days: int = DEFAULT_HISTORY_DAYS) -> None:
    if history_days > MAX_FORECAST_ENDPOINT_DAYS:
        raise ValueError(
            f"The Open-Meteo forecast endpoint serves about "
            f"{MAX_FORECAST_ENDPOINT_DAYS} days of history; "
            f"{history_days} was requested. Use the archive endpoint instead."
        )

    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=history_days - 1)

    print(f"Weather window: {start_date} -> {end_date} ({history_days} days)")

    engine = get_engine()

    for city, (latitude, longitude) in CITIES.items():
        print(f"Ingesting weather for {city}...")

        weather = fetch_hourly_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        inserted = insert_weather(engine, city, weather)
        print(f"  {inserted} rows processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Open-Meteo weather.")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="Days of hourly weather history to request per city.",
    )
    main(parser.parse_args().days)

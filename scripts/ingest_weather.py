from datetime import date, timedelta

from air_quality_intelligence.config.cities import CITIES
from air_quality_intelligence.db.engine import get_engine
from air_quality_intelligence.db.weather import insert_weather
from air_quality_intelligence.ingestion.open_meteo import fetch_hourly_weather


def main() -> None:
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)

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
    main()

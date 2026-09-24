from sqlalchemy import text

from air_quality_intelligence.analysis.daily_aqi import (
    calculate_daily_aqi_for_station,
)
from air_quality_intelligence.db.engine import get_engine


def main() -> None:
    engine = get_engine()

    with engine.connect() as connection:
        station_ids = connection.execute(
            text("SELECT DISTINCT station_id FROM measurements ORDER BY station_id")
        ).scalars().all()

    print(f"Stations with measurements: {len(station_ids)}")

    total_rows = 0

    for station_id in station_ids:
        rows = calculate_daily_aqi_for_station(engine, station_id)
        total_rows += rows

    print(f"Daily AQI rows created/updated: {total_rows}")


if __name__ == "__main__":
    main()

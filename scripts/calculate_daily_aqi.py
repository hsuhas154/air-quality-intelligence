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

    summary_query = text(
        """
        SELECT
            s.city,
            COUNT(*) AS rows,
            MIN(d.aqi) AS min_aqi,
            MAX(d.aqi) AS max_aqi,
            MIN(d.pollutant_count) AS min_pollutants,
            COUNT(*) FILTER (WHERE d.pollutant_count IS NULL) AS unaudited_rows,
            SUM(CASE WHEN d.extrapolated THEN 1 ELSE 0 END) AS extrapolated_rows
        FROM daily_aqi d
        JOIN stations s ON s.station_id = d.station_id
        GROUP BY s.city
        ORDER BY s.city
        """
    )

    with engine.connect() as connection:
        summary = connection.execute(summary_query).all()

    print()
    print("Published daily AQI, by city:")
    print(
        f"{'city':<12}{'rows':>8}{'min':>6}{'max':>6}"
        f"{'min pollutants':>16}{'unaudited':>11}{'extrapolated':>13}"
    )

    stored = 0

    for row in summary:
        stored += row.rows
        print(
            f"{row.city:<12}{row.rows:>8}{row.min_aqi:>6}{row.max_aqi:>6}"
            f"{row.min_pollutants:>16}{row.unaudited_rows:>11}"
            f"{row.extrapolated_rows:>13}"
        )

    print()
    print(
        "min pollutants below 3 would mean the minimum-pollutant rule is not "
        "being applied; extrapolated rows sit above the highest published "
        "breakpoint, so their AQI can exceed 500."
    )

    # The stale-row check.
    #
    # MIN(pollutant_count) skips NULLs, so rows written before the rules
    # existed read as compliant no matter what they contain. Counting them
    # explicitly, and comparing the table against what this run wrote, is what
    # makes a leftover row impossible to miss.
    if stored != total_rows:
        print()
        print(
            f"  WARNING: the table holds {stored} rows but this run wrote "
            f"{total_rows}. The difference is rows for stations that no "
            "longer have measurements, or rows left by an older run."
        )

    unaudited = sum(row.unaudited_rows for row in summary)

    if unaudited:
        print()
        print(
            f"  WARNING: {unaudited} rows have no pollutant_count. They were "
            "written before the published rules were applied and were never "
            "checked against them."
        )


if __name__ == "__main__":
    main()

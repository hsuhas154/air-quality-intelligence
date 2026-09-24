from air_quality_intelligence.analysis.features import load_hourly_features
from air_quality_intelligence.db.engine import get_engine


def longest_missing_run(series):
    missing = series.isna()

    groups = (
        missing
        .ne(missing.shift())
        .cumsum()
    )

    runs = missing.groupby(groups).sum()

    return int(runs.max())


def main():
    engine = get_engine()

    df = load_hourly_features(engine)

    pollutants = [
        "pm25",
        "pm10",
        "no2",
        "so2",
        "o3",
        "co",
    ]

    print("=== Missing observations ===")

    print(
        df.groupby("city")[pollutants]
        .apply(lambda x: x.isna().sum())
        .to_string()
    )

    print("\n=== Longest consecutive missing runs ===")

    for city, group in df.groupby("city"):
        print(f"\n{city}")

        group = group.sort_values("hour")

        for pollutant in pollutants:
            longest = longest_missing_run(
                group[pollutant]
            )

            print(
                f"  {pollutant}: "
                f"{longest} hours"
            )


if __name__ == "__main__":
    main()

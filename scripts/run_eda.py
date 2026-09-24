from pathlib import Path

import matplotlib.pyplot as plt

from air_quality_intelligence.analysis.eda import (
    city_summary,
    load_daily_aqi,
)
from air_quality_intelligence.db.engine import get_engine

OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    engine = get_engine()
    df = load_daily_aqi(engine)
    summary = city_summary(df)

    print(summary.to_string(index=False))

    # City-level mean AQI
    plt.figure(figsize=(8, 5))
    plt.bar(summary["city"], summary["mean_aqi"])
    plt.ylabel("Mean AQI")
    plt.title("Mean AQI by City")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_aqi_by_city.png", dpi=150)
    plt.close()

    # Station-level AQI distribution
    plt.figure(figsize=(9, 5))
    for city in df["city"].unique():
        subset = df[df["city"] == city]
        plt.scatter(
            subset["station_id"],
            subset["aqi"],
            label=city,
            alpha=0.7,
        )

    plt.xlabel("Station ID")
    plt.ylabel("AQI")
    plt.title("Station-Level AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "station_aqi.png", dpi=150)
    plt.close()

    print(f"\nEDA plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

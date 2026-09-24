from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from air_quality_intelligence.analysis.features import load_hourly_features
from air_quality_intelligence.db.engine import get_engine


def evaluate_delta_baseline(df: pd.DataFrame) -> None:
    """Evaluate a historical mean-change baseline for next-hour AQI."""

    data = df.dropna(
        subset=["hourly_aqi", "target_aqi_next_hour"]
    ).copy()

    # AQI change from the current hour to the next hour.
    data["aqi_change"] = (
        data["target_aqi_next_hour"] - data["hourly_aqi"]
    )

    # Chronological split: last 24 hours of each city are test data.
    train_parts = []
    test_parts = []

    for city, group in data.groupby("city"):
        group = group.sort_values("hour").copy()

        if len(group) <= 24:
            raise ValueError(
                f"Not enough observations for {city}: {len(group)}"
            )

        train_parts.append(group.iloc[:-24])
        test_parts.append(group.iloc[-24:])

    train = pd.concat(train_parts).sort_values(
        ["city", "hour"]
    )

    test = pd.concat(test_parts).sort_values(
        ["city", "hour"]
    )

    # Calculate the historical mean AQI change using TRAINING data only.
    mean_change_by_city = (
        train.groupby("city")["aqi_change"]
        .mean()
        .to_dict()
    )

    # Predict the next-hour AQI.
    test["predicted_change"] = (
        test["city"].map(mean_change_by_city)
    )

    test["prediction"] = (
        test["hourly_aqi"] + test["predicted_change"]
    )

    overall_mae = mean_absolute_error(
        test["target_aqi_next_hour"],
        test["prediction"],
    )

    overall_rmse = root_mean_squared_error(
        test["target_aqi_next_hour"],
        test["prediction"],
    )

    print("=== Mean-Change Baseline ===")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    print("\nMean AQI change learned from training data:")

    for city, change in mean_change_by_city.items():
        print(f"{city}: {change:.3f}")

    print("\nOverall:")
    print(f"MAE: {overall_mae:.3f}")
    print(f"RMSE: {overall_rmse:.3f}")

    print("\nBy city:")

    results = []

    for city, group in test.groupby("city"):
        mae = mean_absolute_error(
            group["target_aqi_next_hour"],
            group["prediction"],
        )

        rmse = root_mean_squared_error(
            group["target_aqi_next_hour"],
            group["prediction"],
        )

        results.append(
            {
                "city": city,
                "test_rows": len(group),
                "mae": mae,
                "rmse": rmse,
            }
        )

    results_df = pd.DataFrame(results)

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    dataframe = load_hourly_features(get_engine())
    evaluate_delta_baseline(dataframe)

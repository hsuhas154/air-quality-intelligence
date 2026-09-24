from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from air_quality_intelligence.analysis.features import load_hourly_features
from air_quality_intelligence.analysis.forecast import build_model_features
from air_quality_intelligence.db.engine import get_engine

TARGET = "target_aqi_next_hour"
DELTA_TARGET = "aqi_change"

# The final holdout is constructed chronologically from the latest
# 20% of usable observations in each city. This replaces the old fixed
# timestamp, which was tied to the pre-hardening dataset size.
HOLDOUT_FRACTION = 0.20

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


SELECTED_FEATURES = [
    "hourly_aqi",
    "hour_of_day",
    "day_of_week",

    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co",

    "pm25_station_count",
    "pm10_station_count",
    "no2_station_count",
    "so2_station_count",
    "o3_station_count",
    "co_station_count",
    "total_station_count",
    "pm25_coverage_valid",

    "temp_c",
    "humidity",
    "wind_speed",
    "wind_dir",
    "blh",

    "pm25_lag1",
    "pm25_lag3",
    "pm25_lag6",

    "pm10_lag1",
    "pm10_lag3",
    "pm10_lag6",

    "no2_lag1",
    "no2_lag3",
    "no2_lag6",

    "so2_lag1",
    "so2_lag3",
    "so2_lag6",

    "o3_lag1",
    "o3_lag3",
    "o3_lag6",

    "co_lag1",
    "co_lag3",
    "co_lag6",
]


def evaluate(actual: pd.Series, prediction: pd.Series):
    mae = mean_absolute_error(actual, prediction)
    rmse = root_mean_squared_error(actual, prediction)
    return mae, rmse


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["hour"] = pd.to_datetime(
        data["hour"],
        utc=True,
    )

    data = data.dropna(
        subset=[
            "hourly_aqi",
            TARGET,
        ]
    ).copy()

    data[DELTA_TARGET] = (
        data[TARGET]
        - data["hourly_aqi"]
    )

    data = (
        data
        .sort_values(["city", "hour"])
        .reset_index(drop=True)
    )

    return data


def select_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    missing = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in X_train.columns
    ]

    if missing:
        raise ValueError(
            "Missing required model features:\n"
            + "\n".join(missing)
        )

    X_train = X_train[SELECTED_FEATURES].copy()
    X_test = X_test[SELECTED_FEATURES].copy()

    for frame in (X_train, X_test):
        for column in frame.columns:
            if frame[column].dtype == bool:
                frame[column] = frame[column].astype(int)

    all_missing = [
        column
        for column in X_train.columns
        if X_train[column].isna().all()
    ]

    if all_missing:
        print("\nDropping all-missing training features:")
        for column in all_missing:
            print(f"  {column}")

        X_train = X_train.drop(columns=all_missing)
        X_test = X_test.drop(columns=all_missing)

    medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    non_numeric = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        raise TypeError(
            "Non-numeric model features:\n"
            + "\n".join(non_numeric)
        )

    return X_train, X_test


def main():
    print("=" * 78)
    print("FINAL UNTOUCHED HOLDOUT EVALUATION")
    print("=" * 78)

    print("\nHoldout policy:")
    print(
        f"  Latest {HOLDOUT_FRACTION:.0%} of usable observations "
        "per city"
    )

    engine = get_engine()

    print("\nLoading hourly features...")
    features = load_hourly_features(engine)

    data = prepare_data(features)

    print(f"\nTotal usable observations: {len(data)}")

    print("\nUsable observations by city:")
    print(data.groupby("city").size().to_string())

    print(
        "\nFull usable range:",
        data["hour"].min(),
        "->",
        data["hour"].max(),
    )

    # --------------------------------------------------------------
    # Chronological holdout construction
    # --------------------------------------------------------------
    # The previous implementation used a fixed timestamp and hard-coded
    # row counts from the pre-hardening dataset. After strict temporal AQI
    # validation, those counts are no longer applicable.
    #
    # Preserve the intended ~20% untouched holdout while ensuring that
    # every city contributes the same proportion of its latest usable
    # observations. The split remains strictly chronological within each
    # city.
    development_parts = []
    holdout_parts = []

    for city, group in data.groupby("city"):
        group = (
            group
            .sort_values("hour")
            .reset_index(drop=True)
        )

        holdout_size = max(
            1,
            int(round(len(group) * HOLDOUT_FRACTION)),
        )

        development_parts.append(
            group.iloc[:-holdout_size].copy()
        )
        holdout_parts.append(
            group.iloc[-holdout_size:].copy()
        )

    development = (
        pd.concat(development_parts)
        .sort_values(["city", "hour"])
        .reset_index(drop=True)
    )

    holdout = (
        pd.concat(holdout_parts)
        .sort_values(["city", "hour"])
        .reset_index(drop=True)
    )

    print("\n" + "=" * 78)
    print("DEVELOPMENT / HOLDOUT SPLIT")
    print("=" * 78)

    print(f"\nDevelopment observations: {len(development)}")
    print(development.groupby("city").size().to_string())

    print(
        "\nDevelopment range:",
        development["hour"].min(),
        "->",
        development["hour"].max(),
    )

    print(f"\nHoldout observations: {len(holdout)}")
    print(holdout.groupby("city").size().to_string())

    print(
        "\nHoldout range:",
        holdout["hour"].min(),
        "->",
        holdout["hour"].max(),
    )

    for city in sorted(data["city"].unique()):
        city_development = development[
            development["city"] == city
        ]
        city_holdout = holdout[
            holdout["city"] == city
        ]

        if city_development.empty or city_holdout.empty:
            raise ValueError(
                f"Invalid development/holdout split for {city}."
            )

        if (
            city_development["hour"].max()
            >= city_holdout["hour"].min()
        ):
            raise ValueError(
                f"Temporal overlap detected for {city}: "
                f"development ends at "
                f"{city_development['hour'].max()}, "
                f"holdout begins at "
                f"{city_holdout['hour'].min()}."
            )

    print("\nSplit integrity: PASS")

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    holdout_predictions = holdout.copy()

    holdout_predictions[
        "persistence_prediction"
    ] = holdout_predictions["hourly_aqi"]

    mean_changes = (
        development
        .groupby("city")[DELTA_TARGET]
        .mean()
    )

    holdout_predictions[
        "mean_change_prediction"
    ] = holdout_predictions.apply(
        lambda row: (
            row["hourly_aqi"]
            + mean_changes.get(row["city"], 0.0)
        ),
        axis=1,
    )

    # ------------------------------------------------------------------
    # Frozen RF/C model
    # ------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("TRAINING FROZEN RANDOM FOREST / C FEATURE SET")
    print("=" * 78)

    X_train_raw = development.copy()
    X_test_raw = holdout.copy()

    # Target-validity metadata is not a predictive feature.
    # It describes whether the temporal AQI target was successfully
    # constructed and must therefore not enter the model.
    X_train_raw = X_train_raw.drop(
        columns=["aqi_temporal_valid"],
        errors="ignore",
    )

    X_test_raw = X_test_raw.drop(
        columns=["aqi_temporal_valid"],
        errors="ignore",
    )

    X_train, X_test, _ = build_model_features(
        X_train_raw,
        X_test_raw,
    )

    X_train, X_test = select_features(
        X_train,
        X_test,
    )

    print(f"\nModel features used: {X_train.shape[1]}")
    print(f"Training matrix: {X_train.shape}")
    print(f"Holdout matrix: {X_test.shape}")

    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        **RANDOM_FOREST_PARAMS
    )

    model.fit(
        X_train,
        development[DELTA_TARGET],
    )

    predicted_delta = model.predict(X_test)

    holdout_predictions[
        "random_forest_prediction"
    ] = (
        holdout_predictions["hourly_aqi"]
        + predicted_delta
    )

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("FINAL HOLDOUT RESULTS")
    print("=" * 78)

    target = holdout_predictions[TARGET]

    results = []

    for model_name, column in {
        "Persistence": "persistence_prediction",
        "Mean Change": "mean_change_prediction",
        "Random Forest": "random_forest_prediction",
    }.items():

        mae, rmse = evaluate(
            target,
            holdout_predictions[column],
        )

        results.append(
            {
                "model": model_name,
                "mae": mae,
                "rmse": rmse,
            }
        )

    results_df = pd.DataFrame(results)

    persistence_mae = results_df.loc[
        results_df["model"] == "Persistence",
        "mae",
    ].iloc[0]

    persistence_rmse = results_df.loc[
        results_df["model"] == "Persistence",
        "rmse",
    ].iloc[0]

    results_df[
        "mae_improvement_vs_persistence_pct"
    ] = (
        (
            persistence_mae
            - results_df["mae"]
        )
        / persistence_mae
        * 100
    )

    results_df[
        "rmse_improvement_vs_persistence_pct"
    ] = (
        (
            persistence_rmse
            - results_df["rmse"]
        )
        / persistence_rmse
        * 100
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # City-level evaluation
    # ------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("CITY-LEVEL FINAL HOLDOUT RESULTS")
    print("=" * 78)

    city_rows = []

    for city, group in holdout_predictions.groupby("city"):

        persistence_mae_city, persistence_rmse_city = evaluate(
            group[TARGET],
            group["persistence_prediction"],
        )

        mean_mae_city, mean_rmse_city = evaluate(
            group[TARGET],
            group["mean_change_prediction"],
        )

        rf_mae_city, rf_rmse_city = evaluate(
            group[TARGET],
            group["random_forest_prediction"],
        )

        city_rows.append(
            {
                "city": city,
                "test_rows": len(group),

                "persistence_mae": persistence_mae_city,
                "persistence_rmse": persistence_rmse_city,

                "mean_change_mae": mean_mae_city,
                "mean_change_rmse": mean_rmse_city,

                "random_forest_mae": rf_mae_city,
                "random_forest_rmse": rf_rmse_city,

                "rf_mae_improvement_pct": (
                    (
                        persistence_mae_city
                        - rf_mae_city
                    )
                    / persistence_mae_city
                    * 100
                ),

                "rf_rmse_improvement_pct": (
                    (
                        persistence_rmse_city
                        - rf_rmse_city
                    )
                    / persistence_rmse_city
                    * 100
                ),
            }
        )

    city_df = pd.DataFrame(city_rows)

    print(
        city_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # Holdout diagnostics
    # ------------------------------------------------------------------

    holdout_predictions[
        "rf_error"
    ] = (
        holdout_predictions["random_forest_prediction"]
        - holdout_predictions[TARGET]
    )

    holdout_predictions[
        "persistence_error"
    ] = (
        holdout_predictions["persistence_prediction"]
        - holdout_predictions[TARGET]
    )

    holdout_predictions[
        "rf_abs_error"
    ] = holdout_predictions["rf_error"].abs()

    holdout_predictions[
        "persistence_abs_error"
    ] = holdout_predictions["persistence_error"].abs()

    print("\n" + "=" * 78)
    print("HOLDOUT ERROR DIAGNOSTICS")
    print("=" * 78)

    print("\nPrediction error summary:")
    print(
        holdout_predictions[
            [
                "rf_error",
                "persistence_error",
                "rf_abs_error",
                "persistence_abs_error",
            ]
        ].describe().to_string()
    )

    rf_wins = (
        holdout_predictions["rf_abs_error"]
        < holdout_predictions["persistence_abs_error"]
    ).sum()

    persistence_wins = (
        holdout_predictions["persistence_abs_error"]
        < holdout_predictions["rf_abs_error"]
    ).sum()

    ties = len(holdout_predictions) - rf_wins - persistence_wins

    print("\nObservation-level wins:")
    print(f"  Random Forest: {rf_wins}")
    print(f"  Persistence:   {persistence_wins}")
    print(f"  Ties:           {ties}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results_path = (
        output_dir
        / "final_holdout_model_results.csv"
    )

    city_path = (
        output_dir
        / "final_holdout_model_results_by_city.csv"
    )

    predictions_path = (
        output_dir
        / "final_holdout_predictions.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    city_df.to_csv(
        city_path,
        index=False,
    )

    holdout_predictions.to_csv(
        predictions_path,
        index=False,
    )

    print("\n" + "=" * 78)
    print("FILES SAVED")
    print("=" * 78)

    print(f"\n  {results_path}")
    print(f"  {city_path}")
    print(f"  {predictions_path}")

    print("\nFINAL HOLDOUT EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()

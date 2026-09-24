from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)

from air_quality_intelligence.analysis.features import (
    load_hourly_features,
)
from air_quality_intelligence.analysis.forecast import (
    build_model_features,
)
from air_quality_intelligence.db.engine import get_engine

from xgboost import XGBRegressor


TARGET = "target_aqi_next_hour"
DELTA_TARGET = "aqi_change"

TEST_OBSERVATIONS_PER_CITY = 16
NUMBER_OF_FOLDS = 5
MIN_TRAINING_DAYS = 7

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

HIST_GRADIENT_BOOSTING_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "random_state": 42,
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate(
    actual: pd.Series,
    prediction: pd.Series,
) -> tuple[float, float]:
    """Return MAE and RMSE."""

    mae = mean_absolute_error(
        actual,
        prediction,
    )

    rmse = root_mean_squared_error(
        actual,
        prediction,
    )

    return mae, rmse


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------


def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the chronological forecasting dataset.

    Only observations with a valid current AQI and valid next-hour
    target are retained.

    The supervised target is:

        AQI(t+1) - AQI(t)
    """

    data = df.copy()

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


# ---------------------------------------------------------------------
# Rolling folds
# ---------------------------------------------------------------------


def create_rolling_folds(df):
    """
    Create strict chronological expanding-window folds.

    Design:
    - Each city contributes exactly TEST_OBSERVATIONS_PER_CITY
      observations to every test fold.
    - The initial training period contains at least
      MIN_TRAINING_DAYS of calendar time.
    - Training always occurs strictly before testing.
    - The training window expands after every fold.
    - No future observation can enter the training set of a fold.
    """

    city_data = {}

    for city, city_df in df.groupby("city"):
        city_df = city_df.sort_values("hour").reset_index(drop=True)

        unique_hours = (
            city_df["hour"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        city_data[city] = {
            "df": city_df,
            "hours": unique_hours,
        }

    cities = sorted(city_data)

    if not cities:
        raise ValueError("No cities available for temporal cross-validation.")

    # ------------------------------------------------------------------
    # Establish a common chronological starting point.
    #
    # The first usable timestamp for every city is the maximum of the
    # individual city starting timestamps. This guarantees that the
    # initial training period is defined on a common calendar.
    # ------------------------------------------------------------------

    common_start = max(
        city_data[city]["hours"].min()
        for city in cities
    )

    initial_cutoff = (
        common_start
        + pd.Timedelta(days=MIN_TRAINING_DAYS)
    )

    folds = []

    current_cutoff = initial_cutoff

    for fold_number in range(1, NUMBER_OF_FOLDS + 1):

        train_parts = []
        test_parts = []

        # --------------------------------------------------------------
        # Training:
        # Everything up to and including the current chronological
        # cutoff.
        # --------------------------------------------------------------

        for city in cities:
            city_df = city_data[city]["df"]

            train_city = city_df[
                city_df["hour"] <= current_cutoff
            ].copy()

            if train_city.empty:
                raise ValueError(
                    f"Fold {fold_number}: city '{city}' has no "
                    f"training observations."
                )

            train_parts.append(train_city)

        train = pd.concat(
            train_parts,
            ignore_index=True,
        )

        # --------------------------------------------------------------
        # Testing:
        # Take the first N observations strictly after the training
        # cutoff for EACH city.
        # --------------------------------------------------------------

        for city in cities:
            city_df = city_data[city]["df"]

            future_city = city_df[
                city_df["hour"] > current_cutoff
            ].copy()

            test_city = future_city.head(
                TEST_OBSERVATIONS_PER_CITY
            )

            if len(test_city) < TEST_OBSERVATIONS_PER_CITY:
                raise ValueError(
                    f"Fold {fold_number}: city '{city}' has only "
                    f"{len(test_city)} test observations available; "
                    f"{TEST_OBSERVATIONS_PER_CITY} required."
                )

            test_parts.append(test_city)

        test = pd.concat(
            test_parts,
            ignore_index=True,
        )

        # --------------------------------------------------------------
        # Global chronological safety check.
        # --------------------------------------------------------------

        latest_train = train["hour"].max()
        earliest_test = test["hour"].min()

        if latest_train >= earliest_test:
            raise AssertionError(
                f"Fold {fold_number}: temporal leakage detected. "
                f"latest_train={latest_train}, "
                f"earliest_test={earliest_test}"
            )

        # --------------------------------------------------------------
        # City-level chronological safety check.
        # --------------------------------------------------------------

        for city in cities:
            train_city = train[train["city"] == city]
            test_city = test[test["city"] == city]

            if train_city["hour"].max() >= test_city["hour"].min():
                raise AssertionError(
                    f"Fold {fold_number}: temporal leakage detected "
                    f"for city '{city}'."
                )

        folds.append((train, test))

        # --------------------------------------------------------------
        # Expand the training window.
        #
        # The next fold may train on all observations up to the latest
        # test timestamp from the current fold.
        # --------------------------------------------------------------

        current_cutoff = test["hour"].max()

    return folds


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------


def select_model_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Restrict the model to the empirically selected C feature set.

    Included:
        - current AQI
        - temporal features
        - current pollutants
        - station coverage
        - weather
        - pollutant lags 1, 3 and 6

    Excluded:
        - lag 24
        - rolling features
        - future features
    """

    selected_features = [
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

    missing = [
        feature
        for feature in selected_features
        if feature not in X_train.columns
    ]

    if missing:
        raise ValueError(
            "Selected model features are missing: "
            + ", ".join(missing)
        )

    X_train = X_train[
        selected_features
    ].copy()

    X_test = X_test[
        selected_features
    ].copy()

    # Convert boolean coverage indicator.
    for frame in (X_train, X_test):
        for column in frame.columns:
            if frame[column].dtype == bool:
                frame[column] = (
                    frame[column]
                    .astype(int)
                )

    # Drop features that are completely missing in training.
    all_missing = [
        column
        for column in X_train.columns
        if X_train[column].isna().all()
    ]

    if all_missing:
        X_train = X_train.drop(
            columns=all_missing
        )

        X_test = X_test.drop(
            columns=all_missing
        )

    # Training-only median imputation.
    medians = X_train.median(
        numeric_only=True
    )

    X_train = X_train.fillna(
        medians
    )

    X_test = X_test.fillna(
        medians
    )

    # Any remaining non-numeric columns are invalid.
    non_numeric = X_train.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        raise TypeError(
            "Non-numeric model features: "
            + ", ".join(non_numeric)
        )

    return (
        X_train,
        X_test,
        X_train.columns.tolist(),
    )


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------


def add_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Add persistence and training-only mean-change predictions."""

    result = test.copy()

    result[
        "persistence_prediction"
    ] = result["hourly_aqi"]

    mean_changes = (
        train
        .groupby("city")[DELTA_TARGET]
        .mean()
    )

    result[
        "mean_change_prediction"
    ] = result.apply(
        lambda row: (
            row["hourly_aqi"]
            + mean_changes.get(
                row["city"],
                0.0,
            )
        ),
        axis=1,
    )

    return result


# ---------------------------------------------------------------------
# Supervised model training
# ---------------------------------------------------------------------


def train_model(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Train a delta model and reconstruct next-hour AQI."""

    X_train_raw = train.copy()
    X_test_raw = test.copy()

    # aqi_temporal_valid is target-validity metadata, not a
    # predictive feature. Remove it before model feature construction.
    X_train_raw = X_train_raw.drop(
        columns=["aqi_temporal_valid"],
        errors="ignore",
    )

    X_test_raw = X_test_raw.drop(
        columns=["aqi_temporal_valid"],
        errors="ignore",
    )

    # build_model_features performs the established leakage checks
    # and removes target/future information.
    X_train, X_test, _ = (
        build_model_features(
            X_train_raw,
            X_test_raw,
        )
    )

    # Re-apply the selected C feature set.
    X_train, X_test, _ = (
        select_model_features(
            X_train,
            X_test,
        )
    )

    model.fit(
        X_train,
        train[DELTA_TARGET],
    )

    predicted_delta = model.predict(
        X_test
    )

    result = test.copy()

    result[
        "prediction"
    ] = (
        result["hourly_aqi"]
        + predicted_delta
    )

    return result


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------


def make_models():
    """Return the fixed, untuned ML model configurations."""

    return {
        "Random Forest": RandomForestRegressor(
            **RANDOM_FOREST_PARAMS
        ),

        "HistGradientBoosting": (
            HistGradientBoostingRegressor(
                **HIST_GRADIENT_BOOSTING_PARAMS
            )
        ),

        "XGBoost": XGBRegressor(
            **XGBOOST_PARAMS
        ),
    }


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def calculate_metrics(
    predictions: pd.DataFrame,
) -> dict:
    """Calculate aggregate metrics for every model."""

    metrics = {}

    target = predictions[TARGET]

    model_columns = {
        "Persistence": "persistence_prediction",
        "Mean Change": "mean_change_prediction",
        "Random Forest": "rf_prediction",
        "HistGradientBoosting": "hgb_prediction",
        "XGBoost": "xgb_prediction",
    }

    for model_name, column in model_columns.items():

        mae, rmse = evaluate(
            target,
            predictions[column],
        )

        metrics[
            model_name
        ] = {
            "mae": mae,
            "rmse": rmse,
        }

    return metrics


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:

    print("=" * 78)
    print("FORECAST MODEL COMPARISON")
    print("=" * 78)

    engine = get_engine()

    features = load_hourly_features(
        engine
    )

    data = prepare_data(
        features
    )

    print()
    print(
        "Usable observations:",
        len(data),
    )

    print(
        data.groupby("city")
        .size()
        .to_string()
    )

    print()
    print(
        "Date range:",
        data["hour"].min(),
        "->",
        data["hour"].max(),
    )

    folds = create_rolling_folds(
        data
    )

    fold_results = []
    all_predictions = []

    for fold_number, (
        train,
        test,
    ) in enumerate(
        folds,
        start=1,
    ):

        print()
        print("=" * 78)
        print(
            f"FOLD {fold_number}/{NUMBER_OF_FOLDS}"
        )
        print("=" * 78)

        print(
            "Training observations:",
            len(train),
        )

        print(
            "Test observations:",
            len(test),
        )

        print(
            "Training period:",
            train["hour"].min(),
            "->",
            train["hour"].max(),
        )

        print(
            "Test period:",
            test["hour"].min(),
            "->",
            test["hour"].max(),
        )

        predictions = add_baselines(
            train,
            test,
        )

        models = make_models()

        # -------------------------------------------------------------
        # Random Forest
        # -------------------------------------------------------------

        rf_result = train_model(
            models["Random Forest"],
            train,
            test,
        )

        predictions[
            "rf_prediction"
        ] = rf_result["prediction"].values

        # -------------------------------------------------------------
        # HistGradientBoosting
        # -------------------------------------------------------------

        hgb_result = train_model(
            models["HistGradientBoosting"],
            train,
            test,
        )

        predictions[
            "hgb_prediction"
        ] = hgb_result["prediction"].values

        # -------------------------------------------------------------
        # XGBoost
        # -------------------------------------------------------------

        xgb_result = train_model(
            models["XGBoost"],
            train,
            test,
        )

        predictions[
            "xgb_prediction"
        ] = xgb_result["prediction"].values

        metrics = calculate_metrics(
            predictions
        )

        persistence_mae = metrics[
            "Persistence"
        ]["mae"]

        fold_record = {
            "fold": fold_number,
            "train_rows": len(train),
            "test_rows": len(test),
        }

        for model_name, values in metrics.items():

            fold_record[
                f"{model_name.lower().replace(' ', '_')}_mae"
            ] = values["mae"]

            fold_record[
                f"{model_name.lower().replace(' ', '_')}_rmse"
            ] = values["rmse"]

        fold_results.append(
            fold_record
        )

        predictions[
            "fold"
        ] = fold_number

        all_predictions.append(
            predictions
        )

        print()
        print("Model performance:")

        for model_name, values in metrics.items():

            improvement = (
                (
                    persistence_mae
                    - values["mae"]
                )
                / persistence_mae
                * 100
            )

            print(
                f"  {model_name:<24}"
                f"MAE={values['mae']:.3f}  "
                f"RMSE={values['rmse']:.3f}  "
                f"vs persistence={improvement:+.2f}%"
            )

    fold_results = pd.DataFrame(
        fold_results
    )

    all_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Final aggregate
    # -----------------------------------------------------------------

    print()
    print()
    print("#" * 78)
    print("FINAL MODEL COMPARISON")
    print("#" * 78)

    aggregate_rows = []

    target = all_predictions[TARGET]

    prediction_columns = {
        "Persistence": "persistence_prediction",
        "Mean Change": "mean_change_prediction",
        "Random Forest": "rf_prediction",
        "HistGradientBoosting": "hgb_prediction",
        "XGBoost": "xgb_prediction",
    }

    persistence_mae, persistence_rmse = evaluate(
        target,
        all_predictions[
            "persistence_prediction"
        ],
    )

    for model_name, column in prediction_columns.items():

        mae, rmse = evaluate(
            target,
            all_predictions[column],
        )

        mae_improvement = (
            (
                persistence_mae
                - mae
            )
            / persistence_mae
            * 100
        )

        rmse_improvement = (
            (
                persistence_rmse
                - rmse
            )
            / persistence_rmse
            * 100
        )

        fold_mae_column = (
            f"{model_name.lower().replace(' ', '_')}_mae"
        )

        fold_rmse_column = (
            f"{model_name.lower().replace(' ', '_')}_rmse"
        )

        wins_mae = int(
            (
                fold_results[
                    fold_mae_column
                ]
                < fold_results[
                    "persistence_mae"
                ]
            ).sum()
        )

        wins_rmse = int(
            (
                fold_results[
                    fold_rmse_column
                ]
                < fold_results[
                    "persistence_rmse"
                ]
            ).sum()
        )

        aggregate_rows.append(
            {
                "model": model_name,
                "mae": mae,
                "rmse": rmse,
                "mae_improvement_vs_persistence_pct":
                    mae_improvement,
                "rmse_improvement_vs_persistence_pct":
                    rmse_improvement,
                "mae_fold_wins":
                    wins_mae,
                "rmse_fold_wins":
                    wins_rmse,
            }
        )

    summary = (
        pd.DataFrame(
            aggregate_rows
        )
        .sort_values(
            "mae"
        )
        .reset_index(drop=True)
    )

    print()
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    # -----------------------------------------------------------------
    # City-level aggregate
    # -----------------------------------------------------------------

    print()
    print("CITY-LEVEL AGGREGATE RESULTS")
    print("-" * 78)

    city_rows = []

    for city, group in all_predictions.groupby(
        "city"
    ):

        row = {
            "city": city,
            "test_rows": len(group),
        }

        for model_name, column in prediction_columns.items():

            mae, rmse = evaluate(
                group[TARGET],
                group[column],
            )

            row[
                f"{model_name.lower().replace(' ', '_')}_mae"
            ] = mae

            row[
                f"{model_name.lower().replace(' ', '_')}_rmse"
            ] = rmse

        city_rows.append(row)

    city_summary = pd.DataFrame(
        city_rows
    )

    print(
        city_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    # -----------------------------------------------------------------
    # Save reproducible outputs
    # -----------------------------------------------------------------

    output_dir = Path(
        "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_dir
        / "forecast_model_comparison.csv",
        index=False,
    )

    city_summary.to_csv(
        output_dir
        / "forecast_model_comparison_by_city.csv",
        index=False,
    )

    fold_results.to_csv(
        output_dir
        / "forecast_model_comparison_by_fold.csv",
        index=False,
    )

    print()
    print(
        "Saved results to:"
    )

    print(
        "  outputs/forecast_model_comparison.csv"
    )

    print(
        "  outputs/forecast_model_comparison_by_city.csv"
    )

    print(
        "  outputs/forecast_model_comparison_by_fold.csv"
    )


if __name__ == "__main__":
    main()
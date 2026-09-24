from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from air_quality_intelligence.analysis.features import load_hourly_features
from air_quality_intelligence.db.engine import get_engine

TARGET = "target_aqi_next_hour"
DELTA_TARGET = "aqi_change"

# Columns representing values from the future.
# These must NEVER become model inputs.
FUTURE_FEATURES = {
    "target_pm25_next_hour",
}


def chronological_split(
    data: pd.DataFrame,
    test_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each city chronologically.

    The final `test_hours` observations from each city are used
    as the test set. All earlier observations are used for training.
    """

    train_parts = []
    test_parts = []

    for city, group in data.groupby("city"):
        group = group.sort_values("hour").copy()

        if len(group) <= test_hours:
            raise ValueError(
                f"Not enough observations for {city}: "
                f"{len(group)} rows."
            )

        train_parts.append(
            group.iloc[:-test_hours]
        )

        test_parts.append(
            group.iloc[-test_hours:]
        )

    train = (
        pd.concat(train_parts)
        .sort_values(["city", "hour"])
        .reset_index(drop=True)
    )

    test = (
        pd.concat(test_parts)
        .sort_values(["city", "hour"])
        .reset_index(drop=True)
    )

    return train, test


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


def print_city_results(
    test: pd.DataFrame,
    prediction_column: str,
) -> None:
    """Print forecasting metrics for each city."""

    results = []

    for city, group in test.groupby("city"):
        mae, rmse = evaluate(
            group[TARGET],
            group[prediction_column],
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

    print("\nBy city:")
    print(
        results_df.to_string(index=False)
    )


def build_model_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build leakage-free Random Forest features.

    Processing performed here:

    1. Remove identifiers, targets and future values.
    2. Encode city as a numeric feature.
    3. Convert Boolean features to 0/1.
    4. Verify that every feature is numeric.
    5. Fit median imputation ONLY on training data.
    6. Add missingness indicators.
    """

    # ---------------------------------------------------------
    # 1. Columns that must never be model inputs
    # ---------------------------------------------------------

    model_drop_columns = {
        "city",
        "hour",
        TARGET,
        DELTA_TARGET,
        "persistence_prediction",
        "mean_change_prediction",
        "mean_change_aqi_prediction",
        *FUTURE_FEATURES,
    }

    train_model = train.copy()
    test_model = test.copy()

    # ---------------------------------------------------------
    # 2. Encode city
    # ---------------------------------------------------------

    train_model["city_delhi"] = (
        train_model["city"] == "Delhi"
    ).astype(int)

    test_model["city_delhi"] = (
        test_model["city"] == "Delhi"
    ).astype(int)

    # ---------------------------------------------------------
    # 3. Select model features
    # ---------------------------------------------------------

    X_train_raw = train_model.drop(
        columns=list(model_drop_columns),
        errors="ignore",
    )

    X_test_raw = test_model.drop(
        columns=list(model_drop_columns),
        errors="ignore",
    )

    # ---------------------------------------------------------
    # 4. Leakage checks
    # ---------------------------------------------------------

    leakage_columns = {
        TARGET,
        DELTA_TARGET,
        *FUTURE_FEATURES,
    }

    if not leakage_columns.isdisjoint(
        X_train_raw.columns
    ):
        leaked = sorted(
            leakage_columns.intersection(
                X_train_raw.columns
            )
        )

        raise ValueError(
            f"Target/future leakage detected in training features: "
            f"{leaked}"
        )

    if not leakage_columns.isdisjoint(
        X_test_raw.columns
    ):
        leaked = sorted(
            leakage_columns.intersection(
                X_test_raw.columns
            )
        )

        raise ValueError(
            f"Target/future leakage detected in test features: "
            f"{leaked}"
        )

    if list(X_train_raw.columns) != list(
        X_test_raw.columns
    ):
        raise ValueError(
            "Training and test feature columns do not match."
        )

    # ---------------------------------------------------------
    # 5. Convert Boolean features to numeric 0/1
    # ---------------------------------------------------------
    #
    # Example:
    #
    #     pm25_coverage_valid
    #
    # is a Boolean feature:
    #
    #     True / False
    #
    # Random Forest expects numeric model inputs here, so convert:
    #
    #     True  -> 1
    #     False -> 0
    #
    # Do this generically for every Boolean feature rather than
    # hard-coding pm25_coverage_valid.
    # ---------------------------------------------------------

    boolean_columns = [
        column
        for column in X_train_raw.columns
        if pd.api.types.is_bool_dtype(
            X_train_raw[column]
        )
    ]

    for column in boolean_columns:
        X_train_raw[column] = (
            X_train_raw[column]
            .astype(int)
        )

        X_test_raw[column] = (
            X_test_raw[column]
            .astype(int)
        )

    if boolean_columns:
        print(
            "\nBoolean features converted to 0/1:"
        )

        for column in boolean_columns:
            print(f"  {column}")

    # ---------------------------------------------------------
    # 6. Verify all features are numeric
    # ---------------------------------------------------------

    non_numeric = X_train_raw.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        raise ValueError(
            "Non-numeric model features found: "
            f"{non_numeric}"
        )

    # ---------------------------------------------------------
    # 7. Count missing values before imputation
    # ---------------------------------------------------------

    train_missing = int(
        X_train_raw.isna().sum().sum()
    )

    test_missing = int(
        X_test_raw.isna().sum().sum()
    )

    print("\n=== Missing Feature Values ===")

    print(
        f"Training missing values: "
        f"{train_missing}"
    )

    print(
        f"Test missing values: "
        f"{test_missing}"
    )

    # ---------------------------------------------------------
    # 8. Median imputation
    # ---------------------------------------------------------
    #
    # CRITICAL:
    #
    # The imputer is fitted ONLY on X_train_raw.
    #
    # Test data is transformed using statistics learned from
    # training data.
    #
    # This prevents test-set information from leaking into the
    # model.
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 8a. Remove features that are completely missing in the
    #     training fold.
    #
    #     Such features have no training information from which
    #     a median can be calculated.
    # ---------------------------------------------------------

    all_missing_columns = [
        column
        for column in X_train_raw.columns
        if X_train_raw[column].isna().all()
    ]

    if all_missing_columns:
        print(
            "\nFeatures completely missing in training fold "
            "(dropped):"
        )

        for column in all_missing_columns:
            print(f"  {column}")

        X_train_raw = X_train_raw.drop(
            columns=all_missing_columns
        )

        X_test_raw = X_test_raw.drop(
            columns=all_missing_columns
        )

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    X_train = imputer.fit_transform(
        X_train_raw
    )

    X_test = imputer.transform(
        X_test_raw
    )

    # ---------------------------------------------------------
    # 9. Recover readable feature names
    # ---------------------------------------------------------

    feature_names = (
        imputer
        .get_feature_names_out(
            X_train_raw.columns
        )
        .tolist()
    )

    X_train = pd.DataFrame(
        X_train,
        columns=feature_names,
        index=train.index,
    )

    X_test = pd.DataFrame(
        X_test,
        columns=feature_names,
        index=test.index,
    )

    return (
        X_train,
        X_test,
        feature_names,
    )


def run_forecast(
    df: pd.DataFrame,
    test_hours: int = 24,
) -> None:
    """
    Evaluate next-hour AQI forecasting.

    Models:

    1. Persistence baseline
    2. Historical mean-change baseline
    3. Random Forest delta model

    The Random Forest predicts:

        ΔAQI = next-hour AQI - current-hour AQI

    Then:

        predicted next-hour AQI
            = current AQI + predicted ΔAQI
    """

    # ---------------------------------------------------------
    # 1. Prepare forecasting dataset
    # ---------------------------------------------------------

    # A valid forecasting observation requires:
    #
    #     current-hour AQI
    #
    # and
    #
    #     next-hour AQI
    #
    # Secondary pollutant/weather features are allowed to be
    # missing because the model handles them through imputation.

    data = df.dropna(
        subset=[
            "hourly_aqi",
            TARGET,
        ]
    ).copy()

    # ---------------------------------------------------------
    # Create delta target
    # ---------------------------------------------------------

    data[DELTA_TARGET] = (
        data[TARGET]
        - data["hourly_aqi"]
    )

    # ---------------------------------------------------------
    # Chronological train/test split
    # ---------------------------------------------------------

    train, test = chronological_split(
        data,
        test_hours=test_hours,
    )

    print("=== Dataset ===")

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Test rows: {len(test)}"
    )

    print(
        f"Cities: "
        f"{', '.join(sorted(data['city'].unique()))}"
    )

    # ---------------------------------------------------------
    # 2. Persistence baseline
    # ---------------------------------------------------------
    #
    # Assumption:
    #
    #     next-hour AQI = current-hour AQI
    #
    # Therefore:
    #
    #     predicted ΔAQI = 0
    # ---------------------------------------------------------

    test["persistence_prediction"] = (
        test["hourly_aqi"]
    )

    persistence_mae, persistence_rmse = evaluate(
        test[TARGET],
        test["persistence_prediction"],
    )

    print(
        "\n=== Persistence Baseline ==="
    )

    print(
        f"Overall MAE: "
        f"{persistence_mae:.3f}"
    )

    print(
        f"Overall RMSE: "
        f"{persistence_rmse:.3f}"
    )

    print_city_results(
        test,
        "persistence_prediction",
    )

    # ---------------------------------------------------------
    # 3. Historical mean-change baseline
    # ---------------------------------------------------------
    #
    # Mean change is calculated from TRAINING data only.
    # ---------------------------------------------------------

    mean_change_by_city = (
        train
        .groupby("city")[DELTA_TARGET]
        .mean()
        .to_dict()
    )

    test["mean_change_prediction"] = (
        test["city"].map(
            mean_change_by_city
        )
    )

    test["mean_change_aqi_prediction"] = (
        test["hourly_aqi"]
        + test["mean_change_prediction"]
    )

    # Keep AQI in its valid range.

    test["mean_change_aqi_prediction"] = (
        test["mean_change_aqi_prediction"]
        .clip(0, 500)
    )

    mean_change_mae, mean_change_rmse = evaluate(
        test[TARGET],
        test["mean_change_aqi_prediction"],
    )

    print(
        "\n=== Mean-Change Baseline ==="
    )

    print(
        "\nMean AQI change learned "
        "from training data:"
    )

    for city, change in (
        mean_change_by_city.items()
    ):
        print(
            f"{city}: {change:.3f}"
        )

    print(
        f"\nOverall MAE: "
        f"{mean_change_mae:.3f}"
    )

    print(
        f"Overall RMSE: "
        f"{mean_change_rmse:.3f}"
    )

    print_city_results(
        test,
        "mean_change_aqi_prediction",
    )

    # ---------------------------------------------------------
    # 4. Build Random Forest features
    # ---------------------------------------------------------

    X_train, X_test, feature_names = (
        build_model_features(
            train,
            test,
        )
    )

    y_train = train[DELTA_TARGET]

    # ---------------------------------------------------------
    # 5. Random Forest delta model
    # ---------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Predict AQI change
    # ---------------------------------------------------------

    test["rf_change_prediction"] = (
        model.predict(X_test)
    )

    # ---------------------------------------------------------
    # Convert predicted change into next-hour AQI
    # ---------------------------------------------------------

    test["rf_prediction"] = (
        test["hourly_aqi"]
        + test["rf_change_prediction"]
    )

    # AQI is bounded between 0 and 500.

    test["rf_prediction"] = (
        test["rf_prediction"]
        .clip(0, 500)
    )

    # ---------------------------------------------------------
    # Evaluate against actual next-hour AQI
    # ---------------------------------------------------------

    rf_mae, rf_rmse = evaluate(
        test[TARGET],
        test["rf_prediction"],
    )

    print(
        "\n=== Random Forest Delta Model ==="
    )

    print(
        f"Overall MAE: "
        f"{rf_mae:.3f}"
    )

    print(
        f"Overall RMSE: "
        f"{rf_rmse:.3f}"
    )

    print_city_results(
        test,
        "rf_prediction",
    )

    # ---------------------------------------------------------
    # 6. Model comparison
    # ---------------------------------------------------------

    mae_improvement = (
        (
            persistence_mae
            - rf_mae
        )
        / persistence_mae
        * 100
    )

    rmse_improvement = (
        (
            persistence_rmse
            - rf_rmse
        )
        / persistence_rmse
        * 100
    )

    print(
        "\n=== Model Comparison ==="
    )

    print(
        f"Persistence MAE: "
        f"{persistence_mae:.3f}"
    )

    print(
        f"Mean-Change MAE: "
        f"{mean_change_mae:.3f}"
    )

    print(
        f"Random Forest MAE: "
        f"{rf_mae:.3f}"
    )

    print(
        f"RF MAE improvement vs Persistence: "
        f"{mae_improvement:.1f}%"
    )

    print(
        f"Persistence RMSE: "
        f"{persistence_rmse:.3f}"
    )

    print(
        f"Mean-Change RMSE: "
        f"{mean_change_rmse:.3f}"
    )

    print(
        f"Random Forest RMSE: "
        f"{rf_rmse:.3f}"
    )

    print(
        f"RF RMSE improvement vs Persistence: "
        f"{rmse_improvement:.1f}%"
    )

    # ---------------------------------------------------------
    # 7. Feature importance
    # ---------------------------------------------------------

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": (
                    model.feature_importances_
                ),
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    print(
        "\n=== Top 15 Features ==="
    )

    print(
        importance
        .head(15)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 8. Prediction diagnostics
    # ---------------------------------------------------------

    diagnostics = test[
        [
            "city",
            "hour",
            "hourly_aqi",
            TARGET,
            "persistence_prediction",
            "rf_prediction",
        ]
    ].copy()

    diagnostics["persistence_error"] = (
        diagnostics[TARGET]
        - diagnostics["persistence_prediction"]
    ).abs()

    diagnostics["rf_error"] = (
        diagnostics[TARGET]
        - diagnostics["rf_prediction"]
    ).abs()

    print(
        "\n=== Largest Random Forest Errors ==="
    )

    print(
        diagnostics
        .sort_values(
            "rf_error",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    engine = get_engine()

    dataframe = load_hourly_features(
        engine
    )

    run_forecast(
        dataframe
    )
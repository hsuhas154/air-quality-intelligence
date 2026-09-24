from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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


TARGET = "target_aqi_next_hour"
DELTA_TARGET = "aqi_change"

TEST_OBSERVATIONS_PER_CITY = 16
NUMBER_OF_FOLDS = 10

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------
# Utility functions
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


def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the chronological forecasting dataset.

    Only rows with a valid current-hour AQI and a valid next-hour
    target are retained.

    The delta target is:

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

    data = data.sort_values(
        ["city", "hour"]
    ).reset_index(drop=True)

    return data


# ---------------------------------------------------------------------
# Rolling split
# ---------------------------------------------------------------------


def create_rolling_folds(
    data: pd.DataFrame,
    test_size: int = TEST_OBSERVATIONS_PER_CITY,
    n_folds: int = NUMBER_OF_FOLDS,
) -> list[
    tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]
]:
    """
    Create expanding-window chronological folds.

    IMPORTANT:

    Folds are based on usable observations rather than calendar
    duration.

    Each test fold contains exactly `test_size` observations for
    every city. The current hardened dataset uses 16 observations
    per city per fold so that 10 folds remain feasible.

    Example for one city:

        observations  1 ... 240
        fold 1:
            train = earlier observations
            test  = next 24

        fold 2:
            train = everything before fold 2 test
            test  = next 24

    The training data always precedes the test data chronologically.

    This prevents sparse/missing AQI hours from producing test sets
    containing only a handful of observations.
    """

    cities = sorted(
        data["city"].dropna().unique()
    )

    if not cities:
        raise ValueError(
            "No cities found in dataset."
        )

    city_data: dict[
        str,
        pd.DataFrame,
    ] = {}

    for city in cities:
        group = (
            data[data["city"] == city]
            .sort_values("hour")
            .reset_index(drop=True)
        )

        city_data[city] = group

    minimum_required = (
        n_folds + 1
    ) * test_size

    for city, group in city_data.items():
        if len(group) < minimum_required:
            raise ValueError(
                f"{city} has only {len(group)} usable observations. "
                f"At least {minimum_required} are required for "
                f"{n_folds} folds of {test_size} observations."
            )

    folds = []

    # We construct folds using observation ranks.
    #
    # For fold 1:
    #
    #   train = observations before test block 1
    #   test  = test block 1
    #
    # For fold 2:
    #
    #   train = observations before test block 2
    #   test  = test block 2
    #
    # The training window therefore expands with every fold.

    for fold_number in range(n_folds):

        train_parts = []
        test_parts = []

        test_start = (
            (fold_number + 1)
            * test_size
        )

        test_end = (
            test_start
            + test_size
        )

        for city in cities:

            group = city_data[city]

            train = group.iloc[
                :test_start
            ].copy()

            test = group.iloc[
                test_start:test_end
            ].copy()

            if len(test) != test_size:
                raise ValueError(
                    f"Fold {fold_number + 1}: "
                    f"{city} has {len(test)} test observations; "
                    f"expected {test_size}."
                )

            train_parts.append(train)
            test_parts.append(test)

        train = (
            pd.concat(train_parts)
            .sort_values(
                ["city", "hour"]
            )
            .reset_index(drop=True)
        )

        test = (
            pd.concat(test_parts)
            .sort_values(
                ["city", "hour"]
            )
            .reset_index(drop=True)
        )

        # -------------------------------------------------------------
        # Final chronological safety check.
        # -------------------------------------------------------------

        for city in cities:

            city_train = train[
                train["city"] == city
            ]

            city_test = test[
                test["city"] == city
            ]

            latest_train_time = (
                city_train["hour"].max()
            )

            earliest_test_time = (
                city_test["hour"].min()
            )

            if latest_train_time >= earliest_test_time:
                raise ValueError(
                    f"Temporal leakage detected in fold "
                    f"{fold_number + 1} for {city}: "
                    f"latest training observation "
                    f"{latest_train_time} is not before "
                    f"earliest test observation "
                    f"{earliest_test_time}."
                )

        folds.append(
            (
                train,
                test,
            )
        )

    return folds


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------


def add_baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add persistence and city-level mean-change predictions.

    Mean change is learned strictly from training data.
    """

    result = test.copy()

    # -------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------

    result[
        "persistence_prediction"
    ] = result["hourly_aqi"]

    # -------------------------------------------------------------
    # Mean change
    # -------------------------------------------------------------

    mean_changes = (
        train.groupby("city")[DELTA_TARGET]
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
# Random Forest
# ---------------------------------------------------------------------


def train_random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    RandomForestRegressor,
]:
    """
    Train a Random Forest delta model.

    The model predicts:

        AQI(t+1) - AQI(t)

    rather than the absolute next-hour AQI.

    Feature preprocessing is performed by the existing
    leakage-safe build_model_features() function.
    """

    # aqi_temporal_valid is target-validity metadata, not a
    # predictive feature. Remove it before model feature construction.
    train_model = train.drop(
        columns=["aqi_temporal_valid"],
        errors="ignore",
    ).copy()

    test_model = test.drop(
        columns=["aqi_temporal_valid"],
        errors="ignore",
    ).copy()

    X_train, X_test, feature_names = (
        build_model_features(
            train_model,
            test_model,
        )
    )

    model = RandomForestRegressor(
        **RANDOM_FOREST_PARAMS
    )

    model.fit(
        X_train,
        train[DELTA_TARGET],
    )

    predicted_delta = model.predict(
        X_test
    )

    predictions = test.copy()

    predictions[
        "rf_change_prediction"
    ] = predicted_delta

    predictions[
        "rf_prediction"
    ] = (
        predictions["hourly_aqi"]
        + predictions[
            "rf_change_prediction"
        ]
    )

    return (
        predictions,
        X_train,
        model,
    )


# ---------------------------------------------------------------------
# City-level metrics
# ---------------------------------------------------------------------


def calculate_city_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics separately for every city."""

    results = []

    for city, group in predictions.groupby(
        "city"
    ):

        persistence_mae, persistence_rmse = (
            evaluate(
                group[TARGET],
                group[
                    "persistence_prediction"
                ],
            )
        )

        mean_mae, mean_rmse = evaluate(
            group[TARGET],
            group[
                "mean_change_prediction"
            ],
        )

        rf_mae, rf_rmse = evaluate(
            group[TARGET],
            group["rf_prediction"],
        )

        results.append(
            {
                "city": city,
                "test_rows": len(group),
                "persistence_mae": persistence_mae,
                "mean_change_mae": mean_mae,
                "rf_mae": rf_mae,
                "persistence_rmse": persistence_rmse,
                "mean_change_rmse": mean_rmse,
                "rf_rmse": rf_rmse,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------


def print_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    top_n: int = 15,
) -> None:
    """Print the most important Random Forest features."""

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        "\nTop Random Forest Features:"
    )

    print(
        importance
        .head(top_n)
        .to_string(index=False)
    )


# ---------------------------------------------------------------------
# Fold reporting
# ---------------------------------------------------------------------


def print_fold_results(
    fold_number: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict:
    """Print and return metrics for one fold."""

    persistence_mae, persistence_rmse = (
        evaluate(
            predictions[TARGET],
            predictions[
                "persistence_prediction"
            ],
        )
    )

    mean_mae, mean_rmse = evaluate(
        predictions[TARGET],
        predictions[
            "mean_change_prediction"
        ],
    )

    rf_mae, rf_rmse = evaluate(
        predictions[TARGET],
        predictions["rf_prediction"],
    )

    rf_mae_improvement = (
        (
            persistence_mae
            - rf_mae
        )
        / persistence_mae
        * 100
    )

    rf_rmse_improvement = (
        (
            persistence_rmse
            - rf_rmse
        )
        / persistence_rmse
        * 100
    )

    print()
    print("=" * 78)
    print(
        f"Fold {fold_number}"
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

    print()
    print("Overall:")

    print(
        f"  Persistence  "
        f"MAE={persistence_mae:.3f}  "
        f"RMSE={persistence_rmse:.3f}"
    )

    print(
        f"  Mean-change  "
        f"MAE={mean_mae:.3f}  "
        f"RMSE={mean_rmse:.3f}"
    )

    print(
        f"  Random Forest "
        f"MAE={rf_mae:.3f}  "
        f"RMSE={rf_rmse:.3f}"
    )

    print(
        f"  RF improvement "
        f"MAE={rf_mae_improvement:+.1f}%  "
        f"RMSE={rf_rmse_improvement:+.1f}%"
    )

    city_metrics = calculate_city_metrics(
        predictions
    )

    print()
    print("By city:")

    print(
        city_metrics.to_string(
            index=False
        )
    )

    return {
        "fold": fold_number,
        "train_rows": len(train),
        "test_rows": len(test),
        "persistence_mae": persistence_mae,
        "persistence_rmse": persistence_rmse,
        "mean_change_mae": mean_mae,
        "mean_change_rmse": mean_rmse,
        "rf_mae": rf_mae,
        "rf_rmse": rf_rmse,
        "rf_mae_improvement": rf_mae_improvement,
        "rf_rmse_improvement": rf_rmse_improvement,
    }


# ---------------------------------------------------------------------
# Final aggregate evaluation
# ---------------------------------------------------------------------


def print_final_summary(
    all_predictions: pd.DataFrame,
    fold_results: pd.DataFrame,
) -> None:
    """Print aggregate rolling-forecast results."""

    print()
    print()
    print("#" * 78)
    print("FINAL ROLLING-FORECAST SUMMARY")
    print("#" * 78)

    persistence_mae, persistence_rmse = (
        evaluate(
            all_predictions[TARGET],
            all_predictions[
                "persistence_prediction"
            ],
        )
    )

    mean_mae, mean_rmse = evaluate(
        all_predictions[TARGET],
        all_predictions[
            "mean_change_prediction"
        ],
    )

    rf_mae, rf_rmse = evaluate(
        all_predictions[TARGET],
        all_predictions[
            "rf_prediction"
        ],
    )

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

    print()
    print(
        f"Total test observations: "
        f"{len(all_predictions)}"
    )

    print(
        f"Number of folds: "
        f"{len(fold_results)}"
    )

    print()
    print("Aggregate performance:")

    print(
        f"  Persistence  "
        f"MAE={persistence_mae:.3f}  "
        f"RMSE={persistence_rmse:.3f}"
    )

    print(
        f"  Mean-change  "
        f"MAE={mean_mae:.3f}  "
        f"RMSE={mean_rmse:.3f}"
    )

    print(
        f"  Random Forest "
        f"MAE={rf_mae:.3f}  "
        f"RMSE={rf_rmse:.3f}"
    )

    print()
    print(
        "RF improvement vs persistence:"
    )

    print(
        f"  MAE:  {mae_improvement:+.2f}%"
    )

    print(
        f"  RMSE: {rmse_improvement:+.2f}%"
    )

    # -------------------------------------------------------------
    # Fold consistency
    # -------------------------------------------------------------

    rf_mae_wins = int(
        (
            fold_results["rf_mae"]
            < fold_results[
                "persistence_mae"
            ]
        ).sum()
    )

    rf_rmse_wins = int(
        (
            fold_results["rf_rmse"]
            < fold_results[
                "persistence_rmse"
            ]
        ).sum()
    )

    total_folds = len(
        fold_results
    )

    print()
    print(
        "Fold consistency:"
    )

    print(
        f"  RF lower MAE: "
        f"{rf_mae_wins}/{total_folds} folds"
    )

    print(
        f"  RF lower RMSE: "
        f"{rf_rmse_wins}/{total_folds} folds"
    )

    # -------------------------------------------------------------
    # Distribution of fold improvements
    # -------------------------------------------------------------

    print()
    print(
        "Fold-level MAE improvement:"
    )

    print(
        f"  Mean:   "
        f"{fold_results['rf_mae_improvement'].mean():+.2f}%"
    )

    print(
        f"  Median: "
        f"{fold_results['rf_mae_improvement'].median():+.2f}%"
    )

    print(
        f"  Best:   "
        f"{fold_results['rf_mae_improvement'].max():+.2f}%"
    )

    print(
        f"  Worst:  "
        f"{fold_results['rf_mae_improvement'].min():+.2f}%"
    )

    print()
    print(
        "Fold-level RMSE improvement:"
    )

    print(
        f"  Mean:   "
        f"{fold_results['rf_rmse_improvement'].mean():+.2f}%"
    )

    print(
        f"  Median: "
        f"{fold_results['rf_rmse_improvement'].median():+.2f}%"
    )

    print(
        f"  Best:   "
        f"{fold_results['rf_rmse_improvement'].max():+.2f}%"
    )

    print(
        f"  Worst:  "
        f"{fold_results['rf_rmse_improvement'].min():+.2f}%"
    )

    # -------------------------------------------------------------
    # City-level aggregate performance
    # -------------------------------------------------------------

    print()
    print(
        "Aggregate performance by city:"
    )

    city_results = []

    for city, group in (
        all_predictions.groupby("city")
    ):

        persistence_mae, persistence_rmse = (
            evaluate(
                group[TARGET],
                group[
                    "persistence_prediction"
                ],
            )
        )

        rf_mae, rf_rmse = evaluate(
            group[TARGET],
            group["rf_prediction"],
        )

        mae_imp = (
            (
                persistence_mae
                - rf_mae
            )
            / persistence_mae
            * 100
        )

        rmse_imp = (
            (
                persistence_rmse
                - rf_rmse
            )
            / persistence_rmse
            * 100
        )

        city_results.append(
            {
                "city": city,
                "test_rows": len(group),
                "persistence_mae": persistence_mae,
                "rf_mae": rf_mae,
                "mae_improvement": mae_imp,
                "persistence_rmse": persistence_rmse,
                "rf_rmse": rf_rmse,
                "rmse_improvement": rmse_imp,
            }
        )

    city_df = pd.DataFrame(
        city_results
    )

    print(
        city_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Full fold table
    # -------------------------------------------------------------

    print()
    print(
        "All folds:"
    )

    display_columns = [
        "fold",
        "train_rows",
        "test_rows",
        "persistence_mae",
        "rf_mae",
        "rf_mae_improvement",
        "persistence_rmse",
        "rf_rmse",
        "rf_rmse_improvement",
    ]

    print(
        fold_results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print(
        "Loading hourly features..."
    )

    engine = get_engine()

    df = load_hourly_features(
        engine
    )

    if df.empty:
        raise ValueError(
            "Hourly feature dataset is empty."
        )

    data = prepare_data(df)

    print()
    print(
        "=== Rolling Forecast Dataset ==="
    )

    print(
        f"Total usable rows: "
        f"{len(data)}"
    )

    print()
    print("Rows by city:")

    print(
        data.groupby("city")
        .size()
        .to_string()
    )

    print()
    print(
        "Date range:"
    )

    print(
        data["hour"].min(),
        "->",
        data["hour"].max(),
    )

    print()
    print(
        "Test observations per city per fold:",
        TEST_OBSERVATIONS_PER_CITY,
    )

    print(
        "Number of folds:",
        NUMBER_OF_FOLDS,
    )

    # -------------------------------------------------------------
    # Create folds.
    # -------------------------------------------------------------

    folds = create_rolling_folds(
        data,
        test_size=(
            TEST_OBSERVATIONS_PER_CITY
        ),
        n_folds=NUMBER_OF_FOLDS,
    )

    print()
    print(
        f"Created {len(folds)} rolling folds."
    )

    # -------------------------------------------------------------
    # Evaluate every fold.
    # -------------------------------------------------------------

    fold_results = []

    all_predictions = []

    for fold_number, (
        train,
        test,
    ) in enumerate(
        folds,
        start=1,
    ):

        # ---------------------------------------------------------
        # Baselines
        # ---------------------------------------------------------

        predictions = (
            add_baseline_predictions(
                train,
                test,
            )
        )

        # ---------------------------------------------------------
        # Random Forest
        # ---------------------------------------------------------

        (
            rf_predictions,
            X_train,
            model,
        ) = train_random_forest(
            train,
            predictions,
        )

        # Copy RF predictions back.
        predictions[
            "rf_change_prediction"
        ] = rf_predictions[
            "rf_change_prediction"
        ]

        predictions[
            "rf_prediction"
        ] = rf_predictions[
            "rf_prediction"
        ]

        # ---------------------------------------------------------
        # Fold report
        # ---------------------------------------------------------

        metrics = print_fold_results(
            fold_number,
            train,
            test,
            predictions,
        )

        fold_results.append(
            metrics
        )

        # Add fold identifier.
        predictions[
            "fold"
        ] = fold_number

        all_predictions.append(
            predictions
        )

    # -------------------------------------------------------------
    # Combine results.
    # -------------------------------------------------------------

    all_predictions_df = (
        pd.concat(
            all_predictions,
            ignore_index=True,
        )
    )

    fold_results_df = (
        pd.DataFrame(
            fold_results
        )
    )

    # -------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------

    print_final_summary(
        all_predictions_df,
        fold_results_df,
    )

    print()
    print(
        "Rolling forecast evaluation completed successfully."
    )


if __name__ == "__main__":
    main()
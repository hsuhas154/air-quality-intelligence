from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)

from air_quality_intelligence.analysis.features import (
    load_hourly_features,
)
from air_quality_intelligence.db.engine import get_engine


TARGET = "target_aqi_next_hour"
DELTA_TARGET = "aqi_change"

TEST_OBSERVATIONS_PER_CITY = 24
NUMBER_OF_FOLDS = 10
MIN_TRAINING_DAYS = 7

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================================
# Feature definitions
# ============================================================================

AQI_FEATURES = {
    "hourly_aqi",
}

TEMPORAL_FEATURES = {
    "hour_of_day",
    "day_of_week",
}

CURRENT_POLLUTANT_FEATURES = {
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co",
}

COVERAGE_FEATURES = {
    "co_station_count",
    "no2_station_count",
    "o3_station_count",
    "pm10_station_count",
    "pm25_station_count",
    "so2_station_count",
    "total_station_count",
    "pm25_coverage_valid",
}

WEATHER_FEATURES = {
    "temp_c",
    "humidity",
    "wind_speed",
    "wind_dir",
    "blh",
}

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co",
]


EXPERIMENTS = {
    "A": "Current State",
    "B": "Short History (lag1)",
    "C": "Medium History (lag1/3/6)",
    "D": "Full History (lag1/3/6/24 + rolling)",
}


# ============================================================================
# Metrics
# ============================================================================

def evaluate(
    actual: pd.Series,
    prediction: pd.Series,
) -> tuple[float, float]:

    mae = mean_absolute_error(
        actual,
        prediction,
    )

    rmse = root_mean_squared_error(
        actual,
        prediction,
    )

    return mae, rmse


# ============================================================================
# Data preparation
# ============================================================================

def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = df.dropna(
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
        data.sort_values(
            ["city", "hour"]
        )
        .reset_index(drop=True)
    )

    return data


# ============================================================================
# Rolling folds
# ============================================================================

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


# ============================================================================
# Feature selection
# ============================================================================

def is_pollutant_history(
    column: str,
) -> bool:

    for pollutant in POLLUTANTS:

        if column.startswith(
            f"{pollutant}_lag"
        ):
            return True

        if column.startswith(
            f"{pollutant}_roll"
        ):
            return True

    return False


def history_type(
    column: str,
) -> str:

    """
    Classify pollutant historical features.

    Returns:

        none
        lag1
        lag36
        lag24
        rolling
    """

    if not is_pollutant_history(column):
        return "none"

    if "_lag1" in column:
        return "lag1"

    if "_lag3" in column or "_lag6" in column:
        return "lag36"

    if "_lag24" in column:
        return "lag24"

    if "_roll3" in column or "_roll6" in column:
        return "rolling"

    return "none"


def select_features(
    columns: list[str],
    experiment: str,
) -> list[str]:

    selected = []

    for column in columns:

        # Core features present in every experiment.
        if column in AQI_FEATURES:
            selected.append(column)
            continue

        if column in TEMPORAL_FEATURES:
            selected.append(column)
            continue

        if column in CURRENT_POLLUTANT_FEATURES:
            selected.append(column)
            continue

        if column in COVERAGE_FEATURES:
            selected.append(column)
            continue

        if column in WEATHER_FEATURES:
            selected.append(column)
            continue

        # Historical pollutant features.
        history = history_type(column)

        if experiment == "A":
            continue

        if experiment == "B":
            if history == "lag1":
                selected.append(column)
            continue

        if experiment == "C":
            if history in {
                "lag1",
                "lag36",
            }:
                selected.append(column)
            continue

        if experiment == "D":
            if history in {
                "lag1",
                "lag36",
                "lag24",
                "rolling",
            }:
                selected.append(column)
            continue

    return selected


# ============================================================================
# Model matrix
# ============================================================================

def build_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    experiment: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:

    forbidden = {
        "city",
        "hour",
        TARGET,
        DELTA_TARGET,
        "persistence_prediction",
        "mean_change_prediction",
        "rf_prediction",
        "rf_change_prediction",
        "target_pm25_next_hour",
    }

    candidate_columns = [
        column
        for column in train.columns
        if column not in forbidden
    ]

    selected = select_features(
        candidate_columns,
        experiment,
    )

    if not selected:
        raise ValueError(
            f"No features selected for "
            f"experiment {experiment}."
        )

    X_train = train[selected].copy()
    X_test = test[selected].copy()

    # Convert boolean coverage indicator.
    boolean_columns = X_train.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    for column in boolean_columns:

        X_train[column] = (
            X_train[column]
            .astype(int)
        )

        X_test[column] = (
            X_test[column]
            .astype(int)
        )

    non_numeric = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:

        raise TypeError(
            f"Non-numeric features remain: "
            f"{non_numeric}"
        )

    # Drop features that are completely absent
    # from the training fold.
    usable_columns = [
        column
        for column in X_train.columns
        if not X_train[column].isna().all()
    ]

    dropped = [
        column
        for column in X_train.columns
        if column not in usable_columns
    ]

    if dropped:

        print(
            "      Dropped all-missing training "
            f"features: {', '.join(dropped)}"
        )

    X_train = X_train[
        usable_columns
    ]

    X_test = X_test[
        usable_columns
    ]

    # Training-only imputation.
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    X_train_array = (
        imputer.fit_transform(X_train)
    )

    X_test_array = (
        imputer.transform(X_test)
    )

    feature_names = list(
        imputer.get_feature_names_out(
            usable_columns
        )
    )

    return (
        X_train_array,
        X_test_array,
        feature_names,
    )


# ============================================================================
# Baselines
# ============================================================================

def add_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:

    result = test.copy()

    result[
        "persistence_prediction"
    ] = result["hourly_aqi"]

    mean_changes = (
        train.groupby("city")[
            DELTA_TARGET
        ]
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


# ============================================================================
# Random Forest
# ============================================================================

def train_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    experiment: str,
) -> pd.DataFrame:

    (
        X_train,
        X_test,
        _feature_names,
    ) = build_matrices(
        train,
        test,
        experiment,
    )

    model = RandomForestRegressor(
        **RANDOM_FOREST_PARAMS
    )

    model.fit(
        X_train,
        train[DELTA_TARGET],
    )

    predicted_delta = (
        model.predict(X_test)
    )

    result = test.copy()

    result[
        "rf_prediction"
    ] = (
        result["hourly_aqi"]
        + predicted_delta
    )

    return result


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    print("=" * 78)
    print("TEMPORAL FEATURE-REDUCTION STUDY")
    print("=" * 78)

    engine = get_engine()

    print("\nLoading hourly features...")

    df = load_hourly_features(engine)

    data = prepare_data(df)

    print(
        f"Usable observations: {len(data)}"
    )

    for city in sorted(
        data["city"].unique()
    ):

        print(
            f"  {city}: "
            f"{len(data[data['city'] == city])}"
        )

    print(
        f"Date range: "
        f"{data['hour'].min()} -> "
        f"{data['hour'].max()}"
    )

    folds = create_rolling_folds(data)

    print(
        f"\nRolling evaluation: "
        f"{len(folds)} folds"
    )

    print(
        "Test observations per city per fold: "
        f"{TEST_OBSERVATIONS_PER_CITY}"
    )

    print("\nExperiments:")

    for key, description in EXPERIMENTS.items():

        print(
            f"  {key}: {description}"
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
        print("#" * 78)
        print(
            f"FOLD {fold_number}"
        )
        print("#" * 78)

        print(
            f"Train: {len(train)}"
        )

        print(
            f"Test:  {len(test)}"
        )

        baseline = add_baselines(
            train,
            test,
        )

        persistence_mae, persistence_rmse = (
            evaluate(
                baseline[TARGET],
                baseline[
                    "persistence_prediction"
                ],
            )
        )

        mean_mae, mean_rmse = evaluate(
            baseline[TARGET],
            baseline[
                "mean_change_prediction"
            ],
        )

        print(
            f"\nPersistence: "
            f"MAE={persistence_mae:.3f}, "
            f"RMSE={persistence_rmse:.3f}"
        )

        print(
            f"Mean-change: "
            f"MAE={mean_mae:.3f}, "
            f"RMSE={mean_rmse:.3f}"
        )

        record = {
            "fold": fold_number,
            "train_rows": len(train),
            "test_rows": len(test),
            "persistence_mae":
                persistence_mae,
            "persistence_rmse":
                persistence_rmse,
        }

        for experiment, description in (
            EXPERIMENTS.items()
        ):

            print(
                f"\n  Experiment {experiment}: "
                f"{description}"
            )

            predictions = train_model(
                train,
                test,
                experiment,
            )

            rf_mae, rf_rmse = evaluate(
                predictions[TARGET],
                predictions[
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

            print(
                f"    RF: "
                f"MAE={rf_mae:.3f}, "
                f"RMSE={rf_rmse:.3f}"
            )

            print(
                f"    vs persistence: "
                f"MAE={mae_improvement:+.1f}%, "
                f"RMSE={rmse_improvement:+.1f}%"
            )

            record[
                f"{experiment}_mae"
            ] = rf_mae

            record[
                f"{experiment}_rmse"
            ] = rf_rmse

            record[
                f"{experiment}_mae_improvement"
            ] = mae_improvement

            record[
                f"{experiment}_rmse_improvement"
            ] = rmse_improvement

            predictions[
                "experiment"
            ] = experiment

            predictions[
                "fold"
            ] = fold_number

            all_predictions.append(
                predictions
            )

        fold_results.append(record)

    fold_results_df = pd.DataFrame(
        fold_results
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # ========================================================================
    # Aggregate
    # ========================================================================

    print()
    print()
    print("#" * 78)
    print(
        "FINAL TEMPORAL-REDUCTION SUMMARY"
    )
    print("#" * 78)

    baseline_parts = []

    for fold_number, (
        train,
        test,
    ) in enumerate(
        folds,
        start=1,
    ):

        baseline = add_baselines(
            train,
            test,
        )

        baseline[
            "fold"
        ] = fold_number

        baseline_parts.append(
            baseline
        )

    baseline_df = pd.concat(
        baseline_parts,
        ignore_index=True,
    )

    persistence_mae, persistence_rmse = (
        evaluate(
            baseline_df[TARGET],
            baseline_df[
                "persistence_prediction"
            ],
        )
    )

    mean_mae, mean_rmse = evaluate(
        baseline_df[TARGET],
        baseline_df[
            "mean_change_prediction"
        ],
    )

    print(
        f"\nTotal test observations: "
        f"{len(baseline_df)}"
    )

    print(
        f"Persistence: "
        f"MAE={persistence_mae:.3f}, "
        f"RMSE={persistence_rmse:.3f}"
    )

    print(
        f"Mean-change: "
        f"MAE={mean_mae:.3f}, "
        f"RMSE={mean_rmse:.3f}"
    )

    summary_rows = []

    for experiment, description in (
        EXPERIMENTS.items()
    ):

        group = predictions_df[
            predictions_df[
                "experiment"
            ] == experiment
        ]

        mae, rmse = evaluate(
            group[TARGET],
            group["rf_prediction"],
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

        fold_mae = fold_results_df[
            f"{experiment}_mae"
        ]

        fold_rmse = fold_results_df[
            f"{experiment}_rmse"
        ]

        persistence_fold_mae = (
            fold_results_df[
                "persistence_mae"
            ]
        )

        persistence_fold_rmse = (
            fold_results_df[
                "persistence_rmse"
            ]
        )

        summary_rows.append(
            {
                "experiment":
                    experiment,
                "description":
                    description,
                "mae":
                    mae,
                "rmse":
                    rmse,
                "mae_improvement_pct":
                    mae_improvement,
                "rmse_improvement_pct":
                    rmse_improvement,
                "mae_fold_wins":
                    int(
                        (
                            fold_mae
                            < persistence_fold_mae
                        ).sum()
                    ),
                "rmse_fold_wins":
                    int(
                        (
                            fold_rmse
                            < persistence_fold_rmse
                        ).sum()
                    ),
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            "mae",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    print("\nAggregate results:\n")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    # ========================================================================
    # City-level results
    # ========================================================================

    print()
    print("#" * 78)
    print("CITY-LEVEL RESULTS")
    print("#" * 78)

    city_rows = []

    for experiment, description in (
        EXPERIMENTS.items()
    ):

        model_group = predictions_df[
            predictions_df[
                "experiment"
            ] == experiment
        ]

        for city, city_group in (
            model_group.groupby("city")
        ):

            baseline_city = baseline_df[
                baseline_df[
                    "city"
                ] == city
            ]

            persistence_mae_city, persistence_rmse_city = (
                evaluate(
                    baseline_city[TARGET],
                    baseline_city[
                        "persistence_prediction"
                    ],
                )
            )

            rf_mae_city, rf_rmse_city = (
                evaluate(
                    city_group[TARGET],
                    city_group[
                        "rf_prediction"
                    ],
                )
            )

            city_rows.append(
                {
                    "experiment":
                        experiment,
                    "city":
                        city,
                    "test_rows":
                        len(city_group),
                    "persistence_mae":
                        persistence_mae_city,
                    "rf_mae":
                        rf_mae_city,
                    "mae_improvement_pct":
                        (
                            (
                                persistence_mae_city
                                - rf_mae_city
                            )
                            / persistence_mae_city
                            * 100
                        ),
                    "persistence_rmse":
                        persistence_rmse_city,
                    "rf_rmse":
                        rf_rmse_city,
                    "rmse_improvement_pct":
                        (
                            (
                                persistence_rmse_city
                                - rf_rmse_city
                            )
                            / persistence_rmse_city
                            * 100
                        ),
                }
            )

    city_summary = pd.DataFrame(
        city_rows
    )

    print()

    print(
        city_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    # ========================================================================
    # Incremental improvement
    # ========================================================================

    print()
    print("#" * 78)
    print(
        "INCREMENTAL HISTORY VALUE"
    )
    print("#" * 78)

    ordered = (
        summary.set_index(
            "experiment"
        )
    )

    for previous, current in [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
    ]:

        mae_change = (
            ordered.loc[
                current,
                "mae"
            ]
            - ordered.loc[
                previous,
                "mae"
            ]
        )

        rmse_change = (
            ordered.loc[
                current,
                "rmse"
            ]
            - ordered.loc[
                previous,
                "rmse"
            ]
        )

        print(
            f"\n{previous} -> {current}:"
        )

        print(
            f"  MAE change: "
            f"{mae_change:+.3f}"
        )

        print(
            f"  RMSE change: "
            f"{rmse_change:+.3f}"
        )

    # ========================================================================
    # Final ranking
    # ========================================================================

    print()
    print("#" * 78)
    print(
        "TEMPORAL FEATURE RANKING"
    )
    print("#" * 78)

    ranking = summary[
        [
            "experiment",
            "description",
            "mae",
            "rmse",
            "mae_improvement_pct",
            "rmse_improvement_pct",
            "mae_fold_wins",
            "rmse_fold_wins",
        ]
    ].copy()

    ranking.insert(
        0,
        "rank",
        range(
            1,
            len(ranking) + 1,
        ),
    )

    print()

    print(
        ranking.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print()
    print("=" * 78)
    print(
        "TEMPORAL REDUCTION COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
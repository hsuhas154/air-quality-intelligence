from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from air_quality_intelligence.analysis.features import load_hourly_features
from air_quality_intelligence.db.engine import get_engine


TARGET = "target_aqi_next_hour"
DELTA_TARGET = "aqi_change"

TEST_OBSERVATIONS_PER_CITY = 24
NUMBER_OF_FOLDS = 10

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================================
# Feature groups
# ============================================================================

TEMPORAL_FEATURES = {
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
}

AQI_FEATURES = {
    "hourly_aqi",
}

POLLUTANT_NAMES = {
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co",
}

WEATHER_FEATURES = {
    "temp_c",
    "humidity",
    "wind_speed",
    "wind_dir",
    "blh",
}


def feature_group(feature: str) -> str:
    """
    Classify an engineered feature.

    The classification is deliberately based on the existing feature naming
    convention rather than changing the feature-generation pipeline.
    """

    if feature in AQI_FEATURES:
        return "aqi"

    if feature in TEMPORAL_FEATURES:
        return "temporal"

    if feature in WEATHER_FEATURES:
        return "weather"

    # Pollutant current values.
    if feature in POLLUTANT_NAMES:
        return "pollutant"

    # Pollutant lag / rolling features.
    for pollutant in POLLUTANT_NAMES:
        if feature.startswith(f"{pollutant}_lag"):
            return "pollutant"

        if feature.startswith(f"{pollutant}_roll"):
            return "pollutant"

    # Coverage features belong with pollutant information because they
    # describe pollutant observation availability.
    if feature.endswith("_station_count"):
        return "pollutant"

    if feature == "total_station_count":
        return "pollutant"

    if feature == "pm25_coverage_valid":
        return "pollutant"

    return "other"


def allowed_features(
    columns: list[str],
    experiment: str,
) -> list[str]:
    """
    Return the features permitted for a particular ablation experiment.

    Experiments:

    A = AQI + temporal
    B = AQI + temporal + pollutants
    C = AQI + temporal + pollutants + weather
    D = full engineered feature set
    """

    selected_groups = {
        "A": {"aqi", "temporal"},
        "B": {"aqi", "temporal", "pollutant"},
        "C": {"aqi", "temporal", "pollutant", "weather"},
    }

    if experiment == "D":
        return list(columns)

    groups = selected_groups[experiment]

    return [
        column
        for column in columns
        if feature_group(column) in groups
    ]


# ============================================================================
# Data preparation
# ============================================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only rows with valid current and next-hour AQI.

    The target for the model is AQI(t+1) - AQI(t).
    """

    data = df.dropna(
        subset=[
            "hourly_aqi",
            TARGET,
        ]
    ).copy()

    data[DELTA_TARGET] = (
        data[TARGET] - data["hourly_aqi"]
    )

    data = (
        data.sort_values(["city", "hour"])
        .reset_index(drop=True)
    )

    return data


# ============================================================================
# Rolling folds
# ============================================================================

def create_rolling_folds(
    data: pd.DataFrame,
    test_size: int = TEST_OBSERVATIONS_PER_CITY,
    n_folds: int = NUMBER_OF_FOLDS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create the same balanced expanding-window folds used by the main
    rolling evaluator.

    Each fold contains exactly `test_size` observations per city.
    """

    cities = sorted(
        data["city"].dropna().unique()
    )

    if not cities:
        raise ValueError("No cities found in dataset.")

    city_data = {}

    for city in cities:
        group = (
            data[data["city"] == city]
            .sort_values("hour")
            .reset_index(drop=True)
        )

        city_data[city] = group

    minimum_required = (n_folds + 1) * test_size

    for city, group in city_data.items():
        if len(group) < minimum_required:
            raise ValueError(
                f"{city} has only {len(group)} usable observations. "
                f"At least {minimum_required} are required."
            )

    folds = []

    for fold_number in range(n_folds):

        test_start = (fold_number + 1) * test_size
        test_end = test_start + test_size

        train_parts = []
        test_parts = []

        for city in cities:

            group = city_data[city]

            train = group.iloc[:test_start].copy()

            test = group.iloc[test_start:test_end].copy()

            if len(test) != test_size:
                raise ValueError(
                    f"Fold {fold_number + 1}: {city} has "
                    f"{len(test)} test observations; "
                    f"expected {test_size}."
                )

            train_parts.append(train)
            test_parts.append(test)

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

        # Explicit chronological safety check.
        for city in cities:

            city_train = train[train["city"] == city]
            city_test = test[test["city"] == city]

            latest_train_time = city_train["hour"].max()
            earliest_test_time = city_test["hour"].min()

            if latest_train_time >= earliest_test_time:
                raise ValueError(
                    f"Temporal leakage detected in fold "
                    f"{fold_number + 1} for {city}."
                )

        folds.append((train, test))

    return folds


# ============================================================================
# Feature preprocessing
# ============================================================================

def prepare_model_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    experiment: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Construct leakage-safe train/test matrices for one experiment.

    Imputation statistics are fitted ONLY on the training fold.
    """

    forbidden = {
        "city",
        "hour",
        TARGET,
        DELTA_TARGET,

        # Predictions / derived evaluation columns.
        "persistence_prediction",
        "mean_change_prediction",
        "rf_prediction",
        "rf_change_prediction",

        # Explicit future information.
        "target_pm25_next_hour",
    }

    candidate_columns = [
        column
        for column in train.columns
        if column not in forbidden
    ]

    selected = allowed_features(
        candidate_columns,
        experiment,
    )

    if not selected:
        raise ValueError(
            f"No features selected for experiment {experiment}."
        )

    X_train = train[selected].copy()
    X_test = test[selected].copy()

    # Convert booleans exactly as in the main forecasting pipeline.
    boolean_columns = X_train.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    for column in boolean_columns:
        X_train[column] = X_train[column].astype(int)
        X_test[column] = X_test[column].astype(int)

    # Keep only numeric columns.
    non_numeric = X_train.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if non_numeric:
        raise TypeError(
            f"Non-numeric features remain in experiment "
            f"{experiment}: {non_numeric}"
        )

    # A feature that is entirely missing in the training fold cannot be
    # imputed meaningfully. Drop it before fitting the imputer.
    valid_columns = [
        column
        for column in X_train.columns
        if not X_train[column].isna().all()
    ]

    dropped = [
        column
        for column in X_train.columns
        if column not in valid_columns
    ]

    if dropped:
        print(
            f"    Dropped all-missing training features: "
            f"{', '.join(dropped)}"
        )

    X_train = X_train[valid_columns]
    X_test = X_test[valid_columns]

    # Median imputation is fitted strictly on training data.
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    X_train_array = imputer.fit_transform(X_train)
    X_test_array = imputer.transform(X_test)

    feature_names = list(
        imputer.get_feature_names_out(valid_columns)
    )

    return (
        X_train_array,
        X_test_array,
        feature_names,
        selected,
    )


# ============================================================================
# Model
# ============================================================================

def train_random_forest(
    train: pd.DataFrame,
    test: pd.DataFrame,
    experiment: str,
) -> pd.DataFrame:

    (
        X_train,
        X_test,
        _feature_names,
        _selected_features,
    ) = prepare_model_matrices(
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

    predicted_delta = model.predict(X_test)

    predictions = test.copy()

    predictions["rf_prediction"] = (
        predictions["hourly_aqi"]
        + predicted_delta
    )

    return predictions


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


def add_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:

    result = test.copy()

    # Persistence.
    result["persistence_prediction"] = (
        result["hourly_aqi"]
    )

    # Mean change learned only from training.
    mean_changes = (
        train.groupby("city")[DELTA_TARGET]
        .mean()
    )

    result["mean_change_prediction"] = result.apply(
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
# Main experiment
# ============================================================================

def main() -> None:

    print("=" * 78)
    print("FEATURE ABLATION STUDY")
    print("=" * 78)

    engine = get_engine()

    print("\nLoading hourly features...")

    df = load_hourly_features(engine)

    data = prepare_data(df)

    cities = sorted(data["city"].unique())

    print(f"Usable observations: {len(data)}")

    for city in cities:
        print(
            f"  {city}: "
            f"{len(data[data['city'] == city])}"
        )

    print(
        f"Date range: "
        f"{data['hour'].min()} -> {data['hour'].max()}"
    )

    folds = create_rolling_folds(data)

    print(
        f"\nRolling evaluation: "
        f"{len(folds)} folds"
    )

    print(
        f"Test observations per city per fold: "
        f"{TEST_OBSERVATIONS_PER_CITY}"
    )

    experiments = {
        "A": "AQI + Temporal",
        "B": "AQI + Pollutants",
        "C": "AQI + Pollutants + Weather",
        "D": "Full Engineered Features",
    }

    all_predictions = []
    fold_results = []

    for fold_number, (train, test) in enumerate(
        folds,
        start=1,
    ):

        print()
        print("#" * 78)
        print(f"FOLD {fold_number}")
        print("#" * 78)

        print(
            f"Train: {len(train)} observations"
        )

        print(
            f"Test:  {len(test)} observations"
        )

        baseline_predictions = add_baselines(
            train,
            test,
        )

        persistence_mae, persistence_rmse = evaluate(
            baseline_predictions[TARGET],
            baseline_predictions[
                "persistence_prediction"
            ],
        )

        mean_mae, mean_rmse = evaluate(
            baseline_predictions[TARGET],
            baseline_predictions[
                "mean_change_prediction"
            ],
        )

        print()
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

        fold_record = {
            "fold": fold_number,
            "train_rows": len(train),
            "test_rows": len(test),
            "persistence_mae": persistence_mae,
            "persistence_rmse": persistence_rmse,
            "mean_change_mae": mean_mae,
            "mean_change_rmse": mean_rmse,
        }

        for experiment, description in experiments.items():

            print()
            print(
                f"  Experiment {experiment}: "
                f"{description}"
            )

            predictions = train_random_forest(
                train,
                test,
                experiment,
            )

            rf_mae, rf_rmse = evaluate(
                predictions[TARGET],
                predictions["rf_prediction"],
            )

            mae_improvement = (
                (persistence_mae - rf_mae)
                / persistence_mae
                * 100
            )

            rmse_improvement = (
                (persistence_rmse - rf_rmse)
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

            fold_record[
                f"{experiment}_mae"
            ] = rf_mae

            fold_record[
                f"{experiment}_rmse"
            ] = rf_rmse

            fold_record[
                f"{experiment}_mae_improvement"
            ] = mae_improvement

            fold_record[
                f"{experiment}_rmse_improvement"
            ] = rmse_improvement

            predictions["experiment"] = experiment
            predictions["fold"] = fold_number

            all_predictions.append(
                predictions
            )

        fold_results.append(fold_record)

    fold_results_df = pd.DataFrame(
        fold_results
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # ========================================================================
    # Aggregate results
    # ========================================================================

    print()
    print()
    print("#" * 78)
    print("FINAL FEATURE-ABLATION SUMMARY")
    print("#" * 78)

    baseline_all = []

    for fold_number, (train, test) in enumerate(
        folds,
        start=1,
    ):

        baseline = add_baselines(
            train,
            test,
        )

        baseline["fold"] = fold_number

        baseline_all.append(baseline)

    baseline_df = pd.concat(
        baseline_all,
        ignore_index=True,
    )

    persistence_mae, persistence_rmse = evaluate(
        baseline_df[TARGET],
        baseline_df[
            "persistence_prediction"
        ],
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

    for experiment, description in experiments.items():

        group = predictions_df[
            predictions_df["experiment"]
            == experiment
        ]

        mae, rmse = evaluate(
            group[TARGET],
            group["rf_prediction"],
        )

        mae_improvement = (
            (persistence_mae - mae)
            / persistence_mae
            * 100
        )

        rmse_improvement = (
            (persistence_rmse - rmse)
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

        mae_wins = int(
            (fold_mae < persistence_fold_mae).sum()
        )

        rmse_wins = int(
            (fold_rmse < persistence_fold_rmse).sum()
        )

        summary_rows.append(
            {
                "experiment": experiment,
                "description": description,
                "mae": mae,
                "rmse": rmse,
                "mae_improvement_pct":
                    mae_improvement,
                "rmse_improvement_pct":
                    rmse_improvement,
                "mae_fold_wins":
                    mae_wins,
                "rmse_fold_wins":
                    rmse_wins,
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

    print()
    print("Aggregate results:")
    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # ========================================================================
    # City-level aggregate results
    # ========================================================================

    print()
    print("#" * 78)
    print("CITY-LEVEL RESULTS")
    print("#" * 78)

    city_rows = []

    for experiment, description in experiments.items():

        group = predictions_df[
            predictions_df["experiment"]
            == experiment
        ]

        for city, city_group in group.groupby(
            "city"
        ):

            persistence_group = baseline_df[
                baseline_df["city"] == city
            ]

            persistence_mae_city, persistence_rmse_city = (
                evaluate(
                    persistence_group[TARGET],
                    persistence_group[
                        "persistence_prediction"
                    ],
                )
            )

            rf_mae_city, rf_rmse_city = evaluate(
                city_group[TARGET],
                city_group["rf_prediction"],
            )

            city_rows.append(
                {
                    "experiment": experiment,
                    "city": city,
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

    city_summary = pd.DataFrame(city_rows)

    print()

    print(
        city_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # ========================================================================
    # Ranking
    # ========================================================================

    print()
    print("#" * 78)
    print("FEATURE-GROUP RANKING")
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
        range(1, len(ranking) + 1),
    )

    print()

    print(
        ranking.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("=" * 78)
    print("ABLATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
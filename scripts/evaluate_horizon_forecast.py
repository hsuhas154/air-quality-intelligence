"""Evaluate multi-horizon CPCB AQI forecasting against persistence.

Run from the project root:

    python scripts/evaluate_horizon_forecast.py
    python scripts/evaluate_horizon_forecast.py --horizons 18 24
    python scripts/evaluate_horizon_forecast.py --write-csv outputs/feature_table.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from air_quality_intelligence.analysis.horizon import (
    AQI_COLUMN,
    TARGET_TEMPLATE,
    build_horizon_dataset,
    build_matrices,
    create_expanding_folds,
    model_feature_names,
)

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

MINIMUM_TRAINING_DAYS = 7
NUMBER_OF_FOLDS = 10

# Folds testing less than this share of the evaluable record produce a metric
# that describes a slice rather than the period, and the run says so loudly.
MINIMUM_FOLD_COVERAGE = 0.40

# Share of the evaluable record the folds aim to test, used to size each test
# block from the data rather than from a constant.
#
# A fixed block size cannot work here. create_expanding_folds advances its
# cutoff to the last test timestamp, so a block of 16 observations per city
# moves the window about sixteen hours per fold: ten folds then span a week
# whatever the record length. On thirty days that was tolerable. On ninety it
# tested 13% of the rows, and the coverage guard fired on every run.
#
# Sizing the block from the row count instead keeps coverage roughly constant
# as the dataset grows. The target is below 1.0 because the earliest rows go
# to the first fold's training window and cannot be tested.
TARGET_FOLD_COVERAGE = 0.60

# Set from the data in main(). Kept as a module global because
# evaluate_horizon reads it.
TEST_OBSERVATIONS_PER_CITY = 16


def size_test_block(evaluable_rows: int, cities: int) -> int:
    """Test observations per city per fold, sized to hit the coverage target."""

    if evaluable_rows <= 0 or cities <= 0:
        return 1

    per_city = int(
        TARGET_FOLD_COVERAGE * evaluable_rows / cities / NUMBER_OF_FOLDS
    )

    return max(per_city, 1)

OUTPUT_DIR = Path("outputs")


def load_features(csv: str | None) -> pd.DataFrame:
    if csv:
        print(
            f"  Reading features from {csv}. The database is NOT being read, "
            "so any change to the feature pipeline since this file was "
            "written is not reflected below."
        )
        return pd.read_csv(csv, parse_dates=["hour"])

    from air_quality_intelligence.analysis.features import load_hourly_features
    from air_quality_intelligence.db.engine import get_engine

    return load_hourly_features(get_engine())


def evaluate_horizon(data: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = TARGET_TEMPLATE.format(horizon=horizon)
    evaluable = data.dropna(subset=[AQI_COLUMN, target]).copy()

    folds = create_expanding_folds(
        data,
        evaluable,
        minimum_training_days=MINIMUM_TRAINING_DAYS,
        test_observations_per_city=TEST_OBSERVATIONS_PER_CITY,
        number_of_folds=NUMBER_OF_FOLDS,
    )

    print(f"\nHorizon {horizon}h")
    print(f"  AQI-evaluable rows: {len(evaluable)}")
    per_fold = TEST_OBSERVATIONS_PER_CITY * data["city"].nunique()
    print(f"  Folds: {len(folds)} x {per_fold} test rows")

    if not folds:
        print("  Not enough data at this horizon.")
        return pd.DataFrame(), pd.DataFrame()

    # Fold coverage guard.
    #
    # create_expanding_folds advances its cutoff to the last test timestamp, so
    # a small test block moves the window forward only a few hours per fold. On
    # a long record a fixed fold count can therefore exhaust itself inside the
    # first days and report a metric measured on an unrepresentative slice.
    # This happened: 10 folds of 16 observations per city covered 3 to 12 July
    # of an 86-day record, 13% of the rows, and the headline number from it had
    # to be retracted.
    tested = pd.concat([test for _, test in folds])
    span = evaluable["hour"].max() - evaluable["hour"].min()
    covered = tested["hour"].max() - tested["hour"].min()
    fraction = len(tested) / len(evaluable)

    print(
        f"  Fold coverage: {tested['hour'].min()} to {tested['hour'].max()} "
        f"({covered.days}d of {span.days}d, {fraction:.0%} of evaluable rows)"
    )

    if fraction < MINIMUM_FOLD_COVERAGE:
        print(
            f"  WARNING: the folds test only {fraction:.0%} of the evaluable "
            f"record. This metric is not representative of the full period. "
            f"Raise --test-per-city or the fold count until coverage is at "
            f"least {MINIMUM_FOLD_COVERAGE:.0%}."
        )

    features = model_feature_names(data)
    rows = []

    for fold_number, (cutoff, test) in enumerate(folds, start=1):
        train = data[data["hour"] <= cutoff].dropna(subset=[AQI_COLUMN, target])

        if len(train) < 40:
            continue

        X_train, X_test = build_matrices(train, test, features)

        model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
        model.fit(X_train, train[target] - train[AQI_COLUMN])

        predicted = test[AQI_COLUMN].to_numpy() + model.predict(X_test)

        fold = test[["city", "hour", AQI_COLUMN, target]].copy()
        fold["fold"] = fold_number
        fold["horizon"] = horizon
        fold["random_forest"] = predicted
        fold["persistence"] = test[AQI_COLUMN].to_numpy()
        fold["climatology"] = float(train[target].mean())
        rows.append(fold)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(rows, ignore_index=True)
    actual = predictions[target].to_numpy()

    summary = []
    for name in ("random_forest", "persistence", "climatology"):
        estimate = predictions[name].to_numpy()
        summary.append(
            {
                "horizon": horizon,
                "model": name,
                "n": len(actual),
                "mae": mean_absolute_error(actual, estimate),
                "rmse": root_mean_squared_error(actual, estimate),
            }
        )

    summary = pd.DataFrame(summary)
    baseline = summary.loc[summary["model"] == "persistence"].iloc[0]
    summary["mae_gain_vs_persistence_pct"] = (
        (baseline["mae"] - summary["mae"]) / baseline["mae"] * 100
    )
    summary["rmse_gain_vs_persistence_pct"] = (
        (baseline["rmse"] - summary["rmse"]) / baseline["rmse"] * 100
    )

    for _, row in summary.iterrows():
        print(
            f"  {row['model']:14s} MAE {row['mae']:7.3f}  RMSE {row['rmse']:7.3f}"
            f"  (MAE {row['mae_gain_vs_persistence_pct']:+6.1f}% vs persistence)"
        )

    return summary, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[18, 24])
    parser.add_argument("--test-per-city", type=int, default=None,
                        help="Test observations per city per fold. Sized from "
                             "the data when omitted.")
    parser.add_argument("--csv", type=str, default=None,
                        help="READ features from this CSV instead of rebuilding "
                             "them from the database. Use --write-csv to save "
                             "one. Passing this after a change to the feature "
                             "pipeline silently evaluates stale features.")
    parser.add_argument("--write-csv", type=str, default=None,
                        help="Save the features this run used to this path.")
    args = parser.parse_args()

    features = load_features(args.csv)
    print(f"Feature rows: {len(features)}  cities: {features['city'].nunique()}")

    if args.write_csv:
        Path(args.write_csv).parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.write_csv, index=False)
        print(f"Wrote {args.write_csv}")

    data = build_horizon_dataset(features, horizons=tuple(args.horizons))

    if args.test_per_city is not None:
        globals()["TEST_OBSERVATIONS_PER_CITY"] = args.test_per_city
    else:
        # Size from the shortest horizon, which has the most evaluable rows,
        # so one block size serves every horizon in the run.
        evaluable = min(
            len(data.dropna(subset=[AQI_COLUMN, TARGET_TEMPLATE.format(horizon=h)]))
            for h in args.horizons
        )
        globals()["TEST_OBSERVATIONS_PER_CITY"] = size_test_block(
            evaluable, data["city"].nunique()
        )

    print(
        f"Test block: {TEST_OBSERVATIONS_PER_CITY} observations per city "
        f"per fold, {NUMBER_OF_FOLDS} folds"
    )

    summaries, predictions = [], []
    for horizon in args.horizons:
        summary, prediction = evaluate_horizon(data, horizon)
        if not summary.empty:
            summaries.append(summary)
            predictions.append(prediction)

    if not summaries:
        print("\nNo horizon produced a usable evaluation.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(
        OUTPUT_DIR / "horizon_forecast_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "horizon_forecast_predictions.csv", index=False)

    print(f"\nWrote {OUTPUT_DIR / 'horizon_forecast_results.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'horizon_forecast_predictions.csv'}")


if __name__ == "__main__":
    main()

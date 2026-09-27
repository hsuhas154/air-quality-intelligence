"""Evaluate multi-horizon CPCB AQI forecasting against persistence.

Run from the project root:

    python scripts/evaluate_horizon_forecast.py
    python scripts/evaluate_horizon_forecast.py --horizons 18 24
    python scripts/evaluate_horizon_forecast.py --write-csv outputs/feature_table.csv
    python scripts/evaluate_horizon_forecast.py --sarima
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
from air_quality_intelligence.analysis.significance import (
    compare_to_baseline,
    format_comparison,
)
from air_quality_intelligence.forecast.sarima import (
    DEFAULT_ORDERS,
    SEASONAL_ORDER,
    to_regular_hourly,
    walk_forward_forecast,
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

# Below this share of test rows, a SARIMA score is reported with a warning.
# The rows a diverging model fails on are the hard ones, so scoring it on the
# survivors reads better than the model deserves.
MINIMUM_SARIMA_COVERAGE = 0.90

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


def sarima_predictions(
    data: pd.DataFrame,
    folds: list,
    horizon: int,
    *,
    seasonal: bool,
) -> dict[tuple[str, pd.Timestamp], float]:
    """Walk-forward SARIMA predictions on the folds the other models use.

    Sharing the fold objects rather than rebuilding them is deliberate. A
    separate benchmark script with its own splitting logic is free to drift
    from this one, and a comparison across two different sets of test rows
    says nothing.

    SARIMA is fitted per city, because it is a single-series model and
    pooling two cities into one series would splice Delhi onto Bengaluru.
    """

    series_by_city = {
        city: to_regular_hourly(
            group, time_column="hour", value_column=AQI_COLUMN
        )
        for city, group in data.groupby("city")
    }

    seasonal_order = SEASONAL_ORDER if seasonal else None
    predictions: dict[tuple[str, pd.Timestamp], float] = {}

    for fold_number, (cutoff, test) in enumerate(folds, start=1):
        for city, group in test.groupby("city"):
            series = series_by_city.get(city)

            if series is None or series.empty:
                continue

            fold_predictions, report = walk_forward_forecast(
                series,
                cutoff,
                list(group["hour"]),
                horizon,
                city=str(city),
                orders=DEFAULT_ORDERS,
                seasonal_order=seasonal_order,
            )

            for hour, value in fold_predictions.items():
                predictions[(city, hour)] = value

            aic = f"{report.aic:.1f}" if report.aic is not None else "none"
            print(
                f"    fold {fold_number:2d} {city:<10} order {report.order} "
                f"AIC {aic:>9}  predicted {report.predictions:3d}  "
                f"failed {report.failures:3d}  diverged {report.diverged:3d}"
            )

            for note in report.notes[:3]:
                print(f"      note: {note}")

    return predictions


def summarise(
    predictions: pd.DataFrame,
    target: str,
    horizon: int,
    models: tuple[str, ...],
) -> pd.DataFrame:
    """Score each model on the rows it actually produced a prediction for."""

    rows = []

    for name in models:
        usable = predictions[[target, name]].dropna()

        if usable.empty:
            continue

        rows.append(
            {
                "horizon": horizon,
                "model": name,
                "n": len(usable),
                "mae": mean_absolute_error(usable[target], usable[name]),
                "rmse": root_mean_squared_error(usable[target], usable[name]),
            }
        )

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    baseline = summary.loc[summary["model"] == "persistence"].iloc[0]
    summary["mae_gain_vs_persistence_pct"] = (
        (baseline["mae"] - summary["mae"]) / baseline["mae"] * 100
    )
    summary["rmse_gain_vs_persistence_pct"] = (
        (baseline["rmse"] - summary["rmse"]) / baseline["rmse"] * 100
    )

    return summary


def report(summary: pd.DataFrame, indent: str = "  ") -> None:
    for _, row in summary.iterrows():
        print(
            f"{indent}{row['model']:14s} MAE {row['mae']:7.3f}  "
            f"RMSE {row['rmse']:7.3f}  "
            f"(MAE {row['mae_gain_vs_persistence_pct']:+6.1f}% vs persistence, "
            f"n={int(row['n'])})"
        )


def evaluate_horizon(
    data: pd.DataFrame,
    horizon: int,
    *,
    sarima: bool = False,
    seasonal: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    sarima_by_key: dict[tuple[str, pd.Timestamp], float] = {}

    if sarima:
        print("  SARIMA walk-forward (slow: one fit per candidate order, per")
        print("  fold, per city, then a filter step for every hour):")
        sarima_by_key = sarima_predictions(
            data, folds, horizon, seasonal=seasonal
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

        if sarima:
            fold["sarima"] = [
                sarima_by_key.get((city, hour), float("nan"))
                for city, hour in zip(fold["city"], fold["hour"], strict=True)
            ]

        rows.append(fold)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(rows, ignore_index=True)

    models = ("random_forest", "persistence", "climatology")

    if sarima:
        models = (*models, "sarima")

    summary = summarise(predictions, target, horizon, models)

    if summary.empty:
        return summary, predictions

    report(summary)

    # If SARIMA could not predict every row, the table above scores it on
    # fewer observations than the others, and the comparison is not paired.
    # Scoring every model again on exactly the rows SARIMA managed is the
    # only like-for-like reading.
    if sarima:
        covered = predictions["sarima"].notna()
        share = covered.sum() / len(predictions) if len(predictions) else 0.0

        if share < MINIMUM_SARIMA_COVERAGE:
            print(
                f"\n  WARNING: SARIMA produced a usable forecast for only "
                f"{share:.0%} of the test rows. A score computed on the "
                "survivors of a model that diverged elsewhere flatters it, "
                "because the rows it failed on are exactly the hard ones."
            )

    if sarima and "sarima" in set(summary["model"]):
        covered = predictions["sarima"].notna()

        if covered.sum() and covered.sum() < len(predictions):
            paired = summarise(
                predictions[covered], target, horizon, models
            )
            paired["subset"] = "sarima_rows"
            print(
                f"\n  On the {int(covered.sum())} of {len(predictions)} rows "
                "SARIMA could predict:"
            )
            report(paired, indent="    ")
            summary = pd.concat(
                [summary.assign(subset="all_rows"), paired],
                ignore_index=True,
            )

    print(
        "\n  Paired against persistence on the rows both predicted. The "
        "interval is a\n  moving-block bootstrap of the mean gain: "
        "consecutive hours of a 24-hour\n  rolling mean share most of their "
        "observations, so treating them as\n  independent would claim more "
        "confidence than the data supports."
    )

    for name in models:
        if name == "persistence" or name not in predictions.columns:
            continue

        comparison = compare_to_baseline(
            predictions[target].to_numpy(),
            predictions[name].to_numpy(),
            predictions["persistence"].to_numpy(),
            model_name=name,
        )

        if comparison is not None:
            print(format_comparison(comparison))

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
    parser.add_argument("--sarima", action="store_true",
                        help="Also benchmark SARIMA on the same folds. Slow: "
                             "it fits every candidate order per fold per city "
                             "and then steps a Kalman filter through every "
                             "hour. Expect tens of minutes.")
    parser.add_argument("--seasonal", action="store_true",
                        help="Give SARIMA a 24-hour seasonal term. The AQI is "
                             "already a 24-hour mean, so this mostly models "
                             "the smoothing; off by default.")
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
        summary, prediction = evaluate_horizon(
            data, horizon, sarima=args.sarima, seasonal=args.seasonal
        )
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

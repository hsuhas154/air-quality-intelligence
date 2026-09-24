"""Evaluate multi-horizon CPCB AQI forecasting against persistence.

Run from the project root:

    python scripts/evaluate_horizon_forecast.py
    python scripts/evaluate_horizon_forecast.py --horizons 18 24 --csv outputs/feature_table.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
TEST_OBSERVATIONS_PER_CITY = 16
NUMBER_OF_FOLDS = 10

OUTPUT_DIR = Path("outputs")


def load_features(csv: str | None) -> pd.DataFrame:
    if csv:
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
    print(f"  Folds: {len(folds)} x {TEST_OBSERVATIONS_PER_CITY * data['city'].nunique()} test rows")

    if not folds:
        print("  Not enough data at this horizon.")
        return pd.DataFrame(), pd.DataFrame()

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
    summary["mae_gain_vs_persistence_pct"] = (baseline["mae"] - summary["mae"]) / baseline["mae"] * 100
    summary["rmse_gain_vs_persistence_pct"] = (baseline["rmse"] - summary["rmse"]) / baseline["rmse"] * 100

    for _, row in summary.iterrows():
        print(
            f"  {row['model']:14s} MAE {row['mae']:7.3f}  RMSE {row['rmse']:7.3f}"
            f"  (MAE {row['mae_gain_vs_persistence_pct']:+6.1f}% vs persistence)"
        )

    return summary, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+", default=[18, 24])
    parser.add_argument("--csv", type=str, default=None,
                        help="Read features from CSV instead of the database.")
    args = parser.parse_args()

    features = load_features(args.csv)
    print(f"Feature rows: {len(features)}  cities: {features['city'].nunique()}")

    data = build_horizon_dataset(features, horizons=tuple(args.horizons))

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

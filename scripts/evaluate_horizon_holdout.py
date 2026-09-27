"""Final untouched holdout evaluation for the 24-hour CPCB AQI forecast.

The development set is the earliest 80% of AQI-evaluable observations per
city; the holdout is the latest 20%. Run this once. Any model or feature
change made after reading these numbers makes them a development result, not
a holdout result.

Embargo
-------
A development row at time t carries its target at t + horizon. Without an
embargo, rows in the last `horizon` hours of the development window would be
trained against observations that live inside the holdout period. At a
one-hour horizon that is a single row per city and nobody notices. At 24
hours it is 24 rows per city. Development rows whose target falls at or
after the holdout start are therefore dropped.

Run from the project root:

    python scripts/evaluate_horizon_holdout.py
    python scripts/evaluate_horizon_holdout.py --write-csv outputs/feature_table.csv
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

HORIZON = 24
HOLDOUT_FRACTION = 0.20

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = Path("results/checkpoints")


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


def split_with_embargo(
    evaluable: pd.DataFrame,
    horizon: int,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp], int]:
    """Split per city into development and holdout, embargoing the boundary."""

    development_parts = []
    holdout_parts = []
    cutoffs: dict[str, pd.Timestamp] = {}
    embargoed = 0

    for city, group in evaluable.groupby("city"):
        group = group.sort_values("hour")

        holdout_size = int(round(len(group) * holdout_fraction))
        cutoff = group.iloc[-holdout_size]["hour"]
        cutoffs[city] = cutoff

        holdout = group[group["hour"] >= cutoff]
        development = group[group["hour"] < cutoff]

        safe = development[
            development["hour"] + pd.Timedelta(hours=horizon) < cutoff
        ]
        embargoed += len(development) - len(safe)

        development_parts.append(safe)
        holdout_parts.append(holdout)

    development = pd.concat(development_parts).reset_index(drop=True)
    holdout = pd.concat(holdout_parts).reset_index(drop=True)

    return development, holdout, cutoffs, embargoed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None,
                        help="READ features from this CSV instead of the "
                             "database. Use --write-csv to save one.")
    parser.add_argument("--write-csv", type=str, default=None,
                        help="Save the features this run used to this path.")
    parser.add_argument("--sarima", action="store_true",
                        help="Also score SARIMA on the holdout. This spends "
                             "the holdout on a fourth model, so run it once "
                             "and keep whatever it says.")
    parser.add_argument("--seasonal", action="store_true",
                        help="Give SARIMA a 24-hour seasonal term.")
    args = parser.parse_args()

    target = TARGET_TEMPLATE.format(horizon=HORIZON)

    features = load_features(args.csv)

    if args.write_csv:
        Path(args.write_csv).parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.write_csv, index=False)
        print(f"Wrote {args.write_csv}")

    data = build_horizon_dataset(features, horizons=(HORIZON,))
    evaluable = data.dropna(subset=[AQI_COLUMN, target]).copy()

    development, holdout, cutoffs, embargoed = split_with_embargo(
        evaluable, HORIZON, HOLDOUT_FRACTION
    )

    print("=" * 74)
    print(f"FINAL UNTOUCHED HOLDOUT: {HORIZON}-HOUR CPCB AQI FORECAST")
    print("=" * 74)
    print(f"\nAQI-evaluable observations : {len(evaluable)}")
    print(f"Development                : {len(development)}")
    print(f"Holdout                    : {len(holdout)}")
    print(f"Embargoed at the boundary  : {embargoed}")

    for city, cutoff in sorted(cutoffs.items()):
        print(f"  {city:10s} holdout begins {cutoff}")

    # Integrity checks. Any failure here invalidates the evaluation.
    for city, cutoff in cutoffs.items():
        city_dev = development[development["city"] == city]
        city_hold = holdout[holdout["city"] == city]

        assert city_dev["hour"].max() < cutoff, f"{city}: development crosses the cutoff"
        assert city_hold["hour"].min() >= cutoff, f"{city}: holdout starts early"

        latest_target = city_dev["hour"].max() + pd.Timedelta(hours=HORIZON)
        assert latest_target < cutoff, f"{city}: a development target reaches the holdout"

    print("\nSplit integrity: PASS")

    features_used = model_feature_names(data)
    X_train, X_test = build_matrices(development, holdout, features_used)

    print(f"Model features: {X_train.shape[1]}")
    print(f"Training matrix: {X_train.shape}   Holdout matrix: {X_test.shape}")

    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, development[target] - development[AQI_COLUMN])

    predictions = holdout[["city", "hour", AQI_COLUMN, target]].copy()
    predictions["random_forest"] = holdout[AQI_COLUMN].to_numpy() + model.predict(X_test)
    predictions["persistence"] = holdout[AQI_COLUMN].to_numpy()
    predictions["climatology"] = float(development[target].mean())

    models = ("random_forest", "persistence", "climatology")

    if args.sarima:
        print("\nSARIMA walk-forward on the holdout period:")

        # Created up front so a city that produces nothing leaves NaN rather
        # than an absent column.
        predictions["sarima"] = float("nan")

        # One cutoff per city: the last development hour. The walk then runs
        # forward through the holdout exactly as it does in cross-validation,
        # so the holdout score is produced the same way the fold scores were.
        for city, group in holdout.groupby("city"):
            series = to_regular_hourly(
                data[data["city"] == city],
                time_column="hour",
                value_column=AQI_COLUMN,
            )
            cutoff = development[development["city"] == city]["hour"].max()

            city_predictions, fit_report = walk_forward_forecast(
                series,
                cutoff,
                list(group["hour"]),
                HORIZON,
                city=str(city),
                orders=DEFAULT_ORDERS,
                seasonal_order=SEASONAL_ORDER if args.seasonal else None,
            )

            aic = f"{fit_report.aic:.1f}" if fit_report.aic is not None else "none"
            print(
                f"  {city:<10} order {fit_report.order} AIC {aic:>9}  "
                f"predicted {fit_report.predictions:3d}  "
                f"failed {fit_report.failures:3d}  "
                f"diverged {fit_report.diverged:3d}"
            )

            for note in fit_report.notes[:3]:
                print(f"    note: {note}")

            mask = predictions["city"] == city
            predictions.loc[mask, "sarima"] = [
                city_predictions.get(hour, float("nan"))
                for hour in predictions.loc[mask, "hour"]
            ]

        models = (*models, "sarima")

    rows = []
    for name in models:
        usable = predictions[[target, name]].dropna()

        if usable.empty:
            continue

        rows.append(
            {
                "model": name,
                "n": len(usable),
                "mae": mean_absolute_error(usable[target], usable[name]),
                "rmse": root_mean_squared_error(usable[target], usable[name]),
            }
        )

    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["model"] == "persistence"].iloc[0]
    summary["mae_gain_pct"] = (baseline["mae"] - summary["mae"]) / baseline["mae"] * 100
    summary["rmse_gain_pct"] = (baseline["rmse"] - summary["rmse"]) / baseline["rmse"] * 100

    print("\n" + "=" * 74)
    print("HOLDOUT RESULTS")
    print("=" * 74)
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:14s} MAE {row['mae']:8.4f}   RMSE {row['rmse']:8.4f}"
            f"   MAE {row['mae_gain_pct']:+6.1f}%   RMSE {row['rmse_gain_pct']:+6.1f}%"
        )

    city_rows = []
    print("\nBy city:")
    for city, group in predictions.groupby("city"):
        truth = group[target].to_numpy()
        rf_mae = mean_absolute_error(truth, group["random_forest"])
        pe_mae = mean_absolute_error(truth, group["persistence"])
        rf_rmse = root_mean_squared_error(truth, group["random_forest"])
        pe_rmse = root_mean_squared_error(truth, group["persistence"])
        city_rows.append(
            {
                "city": city, "n": len(group),
                "rf_mae": rf_mae, "persistence_mae": pe_mae,
                "rf_rmse": rf_rmse, "persistence_rmse": pe_rmse,
                "mae_gain_pct": (pe_mae - rf_mae) / pe_mae * 100,
                "rmse_gain_pct": (pe_rmse - rf_rmse) / pe_rmse * 100,
            }
        )
        print(
            f"  {city:10s} n={len(group):4d}  RF MAE {rf_mae:8.4f} vs {pe_mae:8.4f}"
            f"  ({(pe_mae - rf_mae) / pe_mae * 100:+5.1f}%)"
        )

    rf_error = (predictions["random_forest"] - predictions[target]).abs()
    pe_error = (predictions["persistence"] - predictions[target]).abs()
    wins = int((rf_error < pe_error).sum())
    losses = int((rf_error > pe_error).sum())

    print("\nObservation level:")
    print(f"  Random Forest closer : {wins}")
    print(f"  Persistence closer   : {losses}")
    print(f"  Ties                 : {len(predictions) - wins - losses}")
    print(f"  Random Forest wins {wins / len(predictions) * 100:.1f}% of observations")

    print(
        "\nPaired against persistence, with a moving-block bootstrap of the "
        "mean gain.\nConsecutive hours of a 24-hour rolling mean share most "
        "of their observations,\nso an interval that treats them as "
        "independent claims more than it should:"
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
            print(format_comparison(comparison, indent="  "))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "horizon_holdout_results.csv", index=False)
    pd.DataFrame(city_rows).to_csv(OUTPUT_DIR / "horizon_holdout_results_by_city.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "horizon_holdout_predictions.csv", index=False)

    print(f"\nWrote {OUTPUT_DIR / 'horizon_holdout_results.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'horizon_holdout_results_by_city.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'horizon_holdout_predictions.csv'}")
    print("\nThis holdout is now locked.")


if __name__ == "__main__":
    main()

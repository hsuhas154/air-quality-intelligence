import numpy as np
import pandas as pd
import pytest

from air_quality_intelligence.analysis.horizon import (
    AQI_COLUMN,
    POLLUTANTS,
    TARGET_TEMPLATE,
    build_horizon_dataset,
    build_matrices,
    create_expanding_folds,
    model_feature_names,
)


def _frame(hours=60, cities=("Delhi", "Bengaluru")):
    rows = []
    for city in cities:
        index = pd.date_range("2026-07-01", periods=hours, freq="h", tz="UTC")
        for i, ts in enumerate(index):
            rows.append(
                {
                    "city": city, "hour": ts,
                    "pm25": 20 + i % 7, "pm10": 60 + i % 11, "no2": 30 + i % 5,
                    "so2": 25 + i % 3, "o3": 20 + i % 9, "co": 0.5 + (i % 4) / 10,
                    "pm25_coverage_valid": True,
                    "hour_of_day": ts.hour, "day_of_week": ts.dayofweek,
                }
            )
    return pd.DataFrame(rows)


def test_target_uses_real_timestamps_not_positional_shift():
    df = _frame(hours=40)
    # Remove one hour so a positional shift would silently mislabel the target.
    df = df[df["hour"] != pd.Timestamp("2026-07-02 05:00", tz="UTC")]

    out = build_horizon_dataset(df, horizons=(18,))
    target = TARGET_TEMPLATE.format(horizon=18)

    have = out.dropna(subset=[AQI_COLUMN, target])
    for _, row in have.iterrows():
        future = out[(out["city"] == row["city"]) & (out["hour"] == row["hour"] + pd.Timedelta(hours=18))]
        assert not future.empty
        assert future.iloc[0][AQI_COLUMN] == row[target]


def test_aqi_requires_all_six_pollutants():
    df = _frame(hours=40)
    df.loc[df.index[:10], "o3"] = np.nan

    out = build_horizon_dataset(df, horizons=(18,))
    assert out[AQI_COLUMN].isna().any()


def test_model_features_exclude_the_aqi_and_every_target():
    df = _frame(hours=60)
    out = build_horizon_dataset(df, horizons=(18, 24))
    features = model_feature_names(out)

    assert AQI_COLUMN not in features
    assert not [f for f in features if f.startswith("target_")]
    assert all(p in features for p in POLLUTANTS)


def test_coverage_gate_invalidates_the_aqi():
    df = _frame(hours=40)
    df.loc[df.index[:5], "pm25_coverage_valid"] = False

    out = build_horizon_dataset(df, horizons=(18,))
    gated = out[out["pm25_coverage_valid"] == False]  # noqa: E712
    assert gated[AQI_COLUMN].isna().all()


def test_folds_are_chronological_and_do_not_leak():
    df = _frame(hours=400)
    out = build_horizon_dataset(df, horizons=(18,))
    target = TARGET_TEMPLATE.format(horizon=18)
    evaluable = out.dropna(subset=[AQI_COLUMN, target])

    folds = create_expanding_folds(out, evaluable, test_observations_per_city=4, number_of_folds=3)
    assert folds

    for cutoff, test in folds:
        assert cutoff < test["hour"].min()
        assert test.groupby("city").size().nunique() == 1


def test_imputation_uses_training_medians_only():
    df = _frame(hours=60)
    out = build_horizon_dataset(df, horizons=(18,))
    features = model_feature_names(out)

    train = out.iloc[:40].copy()
    test = out.iloc[40:].copy()
    test.loc[test.index, "pm25"] = np.nan

    X_train, X_test = build_matrices(train, test, features)
    assert X_test["pm25"].notna().all()
    assert X_test["pm25"].iloc[0] == pytest.approx(train["pm25"].median())

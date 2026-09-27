"""The dashboard's data layer.

The Streamlit page is a thin shell: it loads artifacts, calls these
functions, and draws. Everything that could be wrong in a way a reader would
not notice lives here, so this is where the tests are.
"""

import numpy as np
import pandas as pd
import pytest

from air_quality_intelligence.dashboard.data import (
    AVERAGING_HOURS,
    comparison_frame,
    comparisons,
    coverage_summary,
    daily_city_aqi,
    hourly_aqi_series,
    load_artifacts,
    window_overlap_table,
)

START = pd.Timestamp("2026-07-01T00:00:00Z")


def features(hours=480, cities=("Delhi", "Bengaluru"), gap=None):
    rows = []

    for offset, city in enumerate(cities):
        index = pd.date_range(START, periods=hours, freq="h", tz="UTC")
        base = 100.0 + offset * 30

        for i, ts in enumerate(index):
            aqi = base + 30.0 * np.sin(i / 36.0)

            if gap and gap[0] <= i < gap[1]:
                aqi = np.nan

            rows.append({"city": city, "hour": ts, "hourly_aqi": aqi})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_a_directory_with_no_artifacts_loads_and_reports_what_is_missing(tmp_path):
    """A fresh clone that has not been run should show an explanation, not a
    traceback."""
    loaded = load_artifacts(tmp_path)

    assert loaded.features is None
    assert set(loaded.missing) == {
        "features",
        "forecast_results",
        "forecast_predictions",
        "holdout_results",
        "holdout_predictions",
    }


def test_an_unreadable_artifact_is_treated_as_absent(tmp_path):
    (tmp_path / "feature_table.csv").write_bytes(b"\x00\x01\x02 not a csv")

    assert "features" in load_artifacts(tmp_path).missing


def test_present_artifacts_are_loaded_with_parsed_timestamps(tmp_path):
    features(hours=48).to_csv(tmp_path / "feature_table.csv", index=False)

    loaded = load_artifacts(tmp_path)

    assert loaded.features is not None
    assert pd.api.types.is_datetime64_any_dtype(loaded.features["hour"])
    assert "features" not in loaded.missing


# ---------------------------------------------------------------------------
# The series, and the gaps in it
# ---------------------------------------------------------------------------


def test_missing_hours_stay_missing_rather_than_closing_up():
    """A correlation at lag h is only a correlation at lag h if the spacing
    is real. Dropping absent hours would compare observations that are not h
    apart and report the result as an autocorrelation."""
    frame = features(hours=200, cities=("Delhi",), gap=(50, 80))

    series = hourly_aqi_series(frame, "Delhi")

    assert len(series) == 200
    assert series.isna().sum() == 30


def test_an_unknown_city_gives_an_empty_series():
    assert hourly_aqi_series(features(hours=24), "Mumbai").empty


# ---------------------------------------------------------------------------
# The window-overlap finding
# ---------------------------------------------------------------------------


def test_window_overlap_is_the_share_of_hours_two_windows_have_in_common():
    table = window_overlap_table(features(), [1, 6, 12, 18, 24, 36])
    overlap = dict(zip(table["horizon"], table["window_overlap"], strict=True))

    assert overlap[1] == pytest.approx(23 / 24)
    assert overlap[6] == pytest.approx(0.75)
    assert overlap[12] == pytest.approx(0.5)
    assert overlap[24] == 0.0
    assert overlap[36] == 0.0, "past the averaging window there is nothing to share"


def test_the_averaging_window_matches_the_aqi_definition():
    assert AVERAGING_HOURS == 24


def test_autocorrelation_is_measured_at_the_true_time_offset():
    """A positional shift would step over gaps and mislabel the lag.

    A pure sine sampled hourly correlates with itself at exactly one period,
    so a correct lag-h correlation returns 1.0 there and a shifted one does
    not.
    """
    period = 48
    index = pd.date_range(START, periods=600, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "city": "Delhi",
            "hour": index,
            "hourly_aqi": 100 + 20 * np.sin(2 * np.pi * np.arange(600) / period),
        }
    ).drop(index=range(100, 160))  # a gap a positional shift would close

    table = window_overlap_table(frame, [period])

    assert table["autocorrelation"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_autocorrelation_falls_away_as_the_horizon_grows():
    table = window_overlap_table(features(), [1, 12, 24, 48])
    values = table["autocorrelation"].to_numpy()

    assert np.all(np.diff(values) <= 1e-9)


def test_each_city_is_reported_as_well_as_the_pool():
    table = window_overlap_table(features(), [6])

    assert "autocorrelation_Delhi" in table.columns
    assert "autocorrelation_Bengaluru" in table.columns
    assert "autocorrelation" in table.columns


def test_pooling_concatenates_observations_rather_than_averaging_coefficients():
    """Averaging each city's r would weight a city by nothing more than
    having been included, regardless of how much data it brought."""
    long_city = features(hours=480, cities=("Delhi",))
    short_city = features(hours=72, cities=("Bengaluru",))
    frame = pd.concat([long_city, short_city], ignore_index=True)

    table = window_overlap_table(frame, [6])
    row = table.iloc[0]
    mean_of_coefficients = np.mean(
        [row["autocorrelation_Delhi"], row["autocorrelation_Bengaluru"]]
    )

    assert row["n"] > len(short_city)
    assert row["autocorrelation"] != pytest.approx(mean_of_coefficients, abs=1e-12)


def test_a_city_too_short_to_correlate_is_nan_not_an_error():
    frame = features(hours=2, cities=("Delhi",))

    table = window_overlap_table(frame, [24])

    assert np.isnan(table["autocorrelation_Delhi"].iloc[0])


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def predictions(n=480):
    index = pd.date_range(START, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    truth = 100 + rng.normal(0, 20, n)

    return pd.DataFrame(
        {
            "city": "Delhi",
            "hour": index,
            "target_aqi_24h": truth,
            "persistence": truth + rng.normal(0, 10, n),
            "sarima": truth + rng.normal(0, 4, n),
            "random_forest": truth + rng.normal(0, 16, n),
        }
    )


def test_every_model_present_is_compared_against_persistence():
    results = comparisons(predictions(), "target_aqi_24h")

    assert {r.model for r in results} == {"sarima", "random_forest"}
    assert all(r.baseline == "persistence" for r in results)


def test_models_appear_in_a_fixed_order():
    frame = comparison_frame(comparisons(predictions(), "target_aqi_24h"))

    assert list(frame["model"]) == ["SARIMA", "Random Forest"]


def test_a_missing_baseline_yields_nothing_rather_than_a_wrong_baseline():
    frame = predictions().drop(columns=["persistence"])

    assert comparisons(frame, "target_aqi_24h") == []


def test_a_missing_target_yields_nothing():
    assert comparisons(predictions(), "target_aqi_18h") == []


def test_an_empty_frame_yields_nothing():
    assert comparisons(pd.DataFrame(), "target_aqi_24h") == []


def test_the_comparison_frame_carries_the_interval_and_the_verdict():
    frame = comparison_frame(comparisons(predictions(), "target_aqi_24h"))

    for column in ("gain", "ci_low", "ci_high", "win_rate", "verdict", "resolved"):
        assert column in frame.columns

    assert (frame["ci_low"] <= frame["gain"]).all()
    assert (frame["gain"] <= frame["ci_high"]).all()


def test_an_empty_comparison_still_gives_a_shaped_frame():
    """The page renders the table before it knows whether there is one."""
    frame = comparison_frame([])

    assert frame.empty
    assert "verdict" in frame.columns


# ---------------------------------------------------------------------------
# Coverage and history
# ---------------------------------------------------------------------------


def test_coverage_counts_the_gap_rather_than_reporting_only_what_is_present():
    frame = features(hours=240, cities=("Delhi",), gap=(60, 120))

    summary = coverage_summary(frame).iloc[0]

    assert summary["hours_in_span"] == 240
    assert summary["hours_with_aqi"] == 180
    assert summary["coverage"] == pytest.approx(0.75)


def test_daily_history_averages_each_city_by_day():
    frame = features(hours=48, cities=("Delhi",))

    daily = daily_city_aqi(frame)

    assert len(daily) == 2
    assert set(daily.columns) == {"city", "day", "aqi"}


def test_daily_history_survives_a_frame_with_no_usable_aqi():
    frame = features(hours=24, cities=("Delhi",))
    frame["hourly_aqi"] = np.nan

    assert daily_city_aqi(frame).empty

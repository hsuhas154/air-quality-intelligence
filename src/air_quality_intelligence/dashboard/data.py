"""What the dashboard shows, computed from the committed run artifacts.

The dashboard reads CSVs from `outputs/`, not the database. That is a
deliberate choice: those files are committed, so anyone who clones the
repository can run the dashboard immediately. Requiring the database would
mean requiring ninety days of ingestion first, and a dashboard nobody can
start is not a dashboard.

The numbers are not recomputed a second way, either. The model comparison
goes through `analysis.significance`, the same code the command-line
evaluation uses, so the dashboard cannot quietly disagree with the run that
produced its inputs. A dashboard with its own arithmetic is a second
implementation waiting to drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from air_quality_intelligence.analysis.significance import (
    PairedComparison,
    compare_to_baseline,
)

# The AQI is a mean over this many hours, which is what makes two nearby
# forecast windows share observations.
AVERAGING_HOURS = 24

MODEL_LABELS = {
    "sarima": "SARIMA",
    "persistence": "Persistence",
    "random_forest": "Random Forest",
    "climatology": "Climatology",
}

MODEL_ORDER = ("sarima", "persistence", "random_forest", "climatology")


@dataclass(frozen=True)
class Artifacts:
    """Whatever run output was found on disk.

    Every frame may be None. The dashboard renders what it has and says what
    is missing, rather than failing on a clone that has not been run yet.
    """

    features: pd.DataFrame | None = None
    forecast_results: pd.DataFrame | None = None
    forecast_predictions: pd.DataFrame | None = None
    holdout_results: pd.DataFrame | None = None
    holdout_predictions: pd.DataFrame | None = None

    @property
    def missing(self) -> list[str]:
        return [
            name
            for name, frame in (
                ("features", self.features),
                ("forecast_results", self.forecast_results),
                ("forecast_predictions", self.forecast_predictions),
                ("holdout_results", self.holdout_results),
                ("holdout_predictions", self.holdout_predictions),
            )
            if frame is None or frame.empty
        ]


def _read(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def load_artifacts(directory: str | Path = "outputs") -> Artifacts:
    """Load the run outputs, tolerating any of them being absent."""

    root = Path(directory)

    return Artifacts(
        features=_read(root / "feature_table.csv", ["hour"]),
        forecast_results=_read(root / "horizon_forecast_results.csv"),
        forecast_predictions=_read(
            root / "horizon_forecast_predictions.csv", ["hour"]
        ),
        holdout_results=_read(root / "horizon_holdout_results.csv"),
        holdout_predictions=_read(
            root / "horizon_holdout_predictions.csv", ["hour"]
        ),
    )


def hourly_aqi_series(features: pd.DataFrame, city: str) -> pd.Series:
    """One city's hourly AQI on a regular index, gaps left as NaN.

    Reindexing matters for the same reason it matters in the SARIMA module:
    a correlation at lag h is only a correlation at lag h if the spacing is
    real. Dropping missing hours would close the gaps and quietly compare
    observations that are not h hours apart.
    """

    column = "hourly_aqi" if "hourly_aqi" in features.columns else "aqi"

    city_rows = features.loc[features["city"] == city, ["hour", column]].dropna(
        subset=["hour"]
    )

    if city_rows.empty:
        return pd.Series(dtype="float64")

    series = (
        city_rows.drop_duplicates(subset=["hour"], keep="last")
        .set_index("hour")[column]
        .sort_index()
        .astype("float64")
    )

    full = pd.date_range(
        series.index.min(), series.index.max(), freq="h", tz=series.index.tz
    )

    return series.reindex(full)


def window_overlap_table(
    features: pd.DataFrame,
    horizons: range | list[int] | None = None,
) -> pd.DataFrame:
    """Autocorrelation of the AQI against how much its windows overlap.

    This is the project's central finding in one table. The AQI at hour t is
    a mean over the previous 24 hours, so the AQI at t and the AQI at t+h
    are computed from windows that share (24-h)/24 of their observations.
    A forecast at h=1 is therefore predicting a number that is already 96
    percent determined by what has been measured, and any model will look
    excellent doing it. The overlap, not the atmosphere, is what the
    autocorrelation is tracking.

    Autocorrelation is pooled across cities by concatenating the paired
    observations rather than averaging each city's coefficient, which would
    weight a city by nothing more than having been included.
    """

    horizons = list(horizons) if horizons is not None else list(range(1, 49))
    cities = sorted(features["city"].dropna().unique())
    series_by_city = {city: hourly_aqi_series(features, city) for city in cities}

    rows = []

    for horizon in horizons:
        pooled_now: list[np.ndarray] = []
        pooled_later: list[np.ndarray] = []
        row: dict[str, object] = {
            "horizon": horizon,
            "window_overlap": max(0.0, (AVERAGING_HOURS - horizon) / AVERAGING_HOURS),
        }

        for city, series in series_by_city.items():
            if series.empty:
                row[f"autocorrelation_{city}"] = np.nan
                continue

            later = series.shift(-horizon, freq="h").reindex(series.index)
            pair = pd.DataFrame({"now": series, "later": later}).dropna()

            if len(pair) < 3:
                row[f"autocorrelation_{city}"] = np.nan
                continue

            row[f"autocorrelation_{city}"] = float(
                pair["now"].corr(pair["later"])
            )
            pooled_now.append(pair["now"].to_numpy())
            pooled_later.append(pair["later"].to_numpy())

        if pooled_now:
            now = np.concatenate(pooled_now)
            later = np.concatenate(pooled_later)
            row["autocorrelation"] = (
                float(np.corrcoef(now, later)[0, 1]) if now.size >= 3 else np.nan
            )
            row["n"] = int(now.size)
        else:
            row["autocorrelation"] = np.nan
            row["n"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def comparisons(
    predictions: pd.DataFrame,
    target: str,
    baseline: str = "persistence",
) -> list[PairedComparison]:
    """Every model in the frame, paired against the baseline.

    Uses analysis.significance, so these are the same numbers the
    command-line run prints rather than a second calculation of them.
    """

    if predictions is None or predictions.empty or target not in predictions:
        return []

    if baseline not in predictions.columns:
        return []

    results = []

    for name in MODEL_ORDER:
        if name == baseline or name not in predictions.columns:
            continue

        comparison = compare_to_baseline(
            predictions[target].to_numpy(),
            predictions[name].to_numpy(),
            predictions[baseline].to_numpy(),
            model_name=name,
        )

        if comparison is not None:
            results.append(comparison)

    return results


def comparison_frame(results: list[PairedComparison]) -> pd.DataFrame:
    """The comparisons as a table, for display and for the table view."""

    if not results:
        return pd.DataFrame(
            columns=[
                "model",
                "n",
                "mae",
                "baseline_mae",
                "gain",
                "ci_low",
                "ci_high",
                "win_rate",
                "verdict",
                "resolved",
            ]
        )

    return pd.DataFrame(
        [
            {
                "model": MODEL_LABELS.get(r.model, r.model),
                "n": r.n,
                "mae": r.model_mae,
                "baseline_mae": r.baseline_mae,
                "gain": r.mae_difference,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "win_rate": r.win_rate,
                "verdict": r.verdict,
                "resolved": r.significant,
            }
            for r in results
        ]
    )


def daily_city_aqi(features: pd.DataFrame) -> pd.DataFrame:
    """Daily mean AQI per city, for the history chart."""

    column = "hourly_aqi" if "hourly_aqi" in features.columns else "aqi"
    usable = features[["city", "hour", column]].dropna()

    if usable.empty:
        return pd.DataFrame(columns=["city", "day", "aqi"])

    return (
        usable.assign(day=usable["hour"].dt.floor("D"))
        .groupby(["city", "day"], as_index=False)[column]
        .mean()
        .rename(columns={column: "aqi"})
        .sort_values(["city", "day"])
        .reset_index(drop=True)
    )


def coverage_summary(features: pd.DataFrame) -> pd.DataFrame:
    """What the data actually covers, per city, gaps included.

    The gap columns are the point. A coverage figure that reports only what
    is present invites the reader to assume the rest.
    """

    column = "hourly_aqi" if "hourly_aqi" in features.columns else "aqi"
    rows = []

    for city, group in features.groupby("city"):
        series = hourly_aqi_series(features, str(city))
        observed = int(group[column].notna().sum())
        span_hours = int(len(series))

        rows.append(
            {
                "city": city,
                "feature_rows": int(len(group)),
                "hours_with_aqi": observed,
                "hours_in_span": span_hours,
                "coverage": observed / span_hours if span_hours else np.nan,
                "first_hour": group["hour"].min(),
                "last_hour": group["hour"].max(),
                "days_with_aqi": int(
                    group.loc[group[column].notna(), "hour"].dt.floor("D").nunique()
                ),
            }
        )

    return pd.DataFrame(rows)

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from air_quality_intelligence.analysis.aqi import calculate_aqi
from air_quality_intelligence.analysis.temporal import (
    calculate_strict_averages,
)
from air_quality_intelligence.transform.units import (
    UnitConversionError,
    normalize_concentration,
)

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co",
]


MIN_PM25_STATIONS = {
    "Bengaluru": 5,
    "Delhi": 31,
}


LAG_HOURS = [
    1,
    3,
    6,
    24,
]


ROLLING_POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "o3",
]


ROLLING_WINDOWS_HOURS = [
    3,
    6,
]


def _normalize_measurement_units(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize every pollutant measurement to its canonical AQI unit.

    Canonical units:
        pm25, pm10, no2, so2, o3 -> µg/m³
        co -> mg/m³

    Non-finite measurement values are discarded before normalization.
    Invalid pollutant/unit combinations fail loudly rather than
    silently entering the AQI calculation.
    """
    measurements = measurements.copy()

    # Remove NULL/NaN/±inf measurements before unit conversion.
    # These are invalid observations, not zero concentrations.
    measurements = measurements[
        measurements["value"].notna()
        & np.isfinite(measurements["value"])
    ].copy()

    normalized_values = []
    normalized_units = []

    for row in measurements.itertuples(index=False):
        try:
            normalized_value, normalized_unit = (
                normalize_concentration(
                    pollutant=row.pollutant,
                    value=row.value,
                    unit=row.unit,
                )
            )
        except UnitConversionError as exc:
            raise ValueError(
                "Failed to normalize air-quality measurement: "
                f"pollutant={row.pollutant!r}, "
                f"value={row.value!r}, "
                f"unit={row.unit!r}"
            ) from exc

        normalized_values.append(normalized_value)
        normalized_units.append(normalized_unit)

    measurements["value"] = normalized_values
    measurements["unit"] = normalized_units

    return measurements


def _add_exact_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add exact time-based pollutant lag features.
    """
    df = df.copy()

    df = df.sort_values(
        ["city", "hour"]
    ).reset_index(drop=True)

    for pollutant in POLLUTANTS:

        if pollutant not in df.columns:
            continue

        source = df[
            [
                "city",
                "hour",
                pollutant,
            ]
        ].copy()

        source = source.rename(
            columns={
                pollutant: "value",
            }
        )

        lagged_frames = []

        for lag_hours in LAG_HOURS:

            lagged = source.copy()

            lagged["hour"] = (
                lagged["hour"]
                + pd.Timedelta(hours=lag_hours)
            )

            lagged = lagged.rename(
                columns={
                    "value": f"{pollutant}_lag{lag_hours}",
                }
            )

            lagged_frames.append(
                lagged[
                    [
                        "city",
                        "hour",
                        f"{pollutant}_lag{lag_hours}",
                    ]
                ]
            )

        pollutant_lags = lagged_frames[0]

        for lagged in lagged_frames[1:]:

            pollutant_lags = pollutant_lags.merge(
                lagged,
                on=[
                    "city",
                    "hour",
                ],
                how="outer",
            )

        df = df.merge(
            pollutant_lags,
            on=[
                "city",
                "hour",
            ],
            how="left",
            validate="one_to_one",
        )

    return df


def _add_time_based_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add timestamp-based pollutant rolling features.

    These remain experimental forecasting features and are not used
    as substitutes for CPCB AQI averaging periods.
    """
    df = df.copy()

    df = df.sort_values(
        ["city", "hour"]
    ).reset_index(drop=True)

    for pollutant in ROLLING_POLLUTANTS:

        if pollutant not in df.columns:
            continue

        for window_hours in ROLLING_WINDOWS_HOURS:

            rolling_column = (
                f"{pollutant}_roll{window_hours}"
            )

            rolling_values = pd.Series(
                index=df.index,
                dtype="float64",
            )

            for city, city_group in df.groupby(
                "city",
                sort=False,
            ):

                city_indices = city_group.index

                city_values = (
                    city_group[
                        [
                            "hour",
                            pollutant,
                        ]
                    ]
                    .set_index("hour")[pollutant]
                    .sort_index()
                )

                city_rolling = (
                    city_values
                    .rolling(
                        window=f"{window_hours}h",
                        min_periods=1,
                    )
                    .mean()
                )

                timestamp_to_value = city_rolling.to_dict()

                rolling_values.loc[city_indices] = (
                    city_group["hour"]
                    .map(timestamp_to_value)
                    .to_numpy()
                )

            df[rolling_column] = rolling_values

    return df


def _build_temporal_aqi_table(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build city-hour CPCB-style AQI inputs.

    Processing order:

        normalized sensor measurements
            -> hourly city concentrations
            -> strict pollutant-specific averaging
            -> AQI

    AQI averaging periods:
        PM2.5, PM10, NO2, SO2 = 24h
        O3, CO = 8h

    The resulting AQI is associated with the END of the required
    trailing averaging window.
    """

    hourly = (
        measurements
        .groupby(
            [
                "city",
                "hour",
                "pollutant",
            ],
            as_index=False,
        )
        .agg(
            value=("value", "mean"),
        )
    )

    hourly["hour"] = pd.to_datetime(
        hourly["hour"],
        utc=True,
    )

    temporal_input = hourly[
        [
            "city",
            "hour",
            "pollutant",
            "value",
        ]
    ].rename(
        columns={"hour": "ts"}
    )

    temporal = calculate_strict_averages(
        temporal_input,
        group_cols=("city",),
    )

    if temporal.empty:
        return pd.DataFrame(
            columns=[
                "city",
                "hour",
                "hourly_aqi",
            ]
        )

    # The temporal layer uses its canonical timestamp name, "ts".
    # Convert it back to the feature-table timestamp name, "hour".
    temporal = temporal.rename(
        columns={"ts": "hour"}
    )

    concentrations = (
        temporal
        .pivot_table(
            index=[
                "city",
                "hour",
            ],
            columns="pollutant",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    concentrations.columns.name = None

    def calculate_row_aqi(row: pd.Series) -> float | None:
        # AQI is produced only when all six project pollutants
        # have valid temporal averages for this timestamp.
        if not all(
            pollutant in row.index and pd.notna(row[pollutant])
            for pollutant in POLLUTANTS
        ):
            return None

        values = {
            pollutant: row[pollutant]
            for pollutant in POLLUTANTS
        }

        aqi, _ = calculate_aqi(values)
        return aqi


    concentrations["hourly_aqi"] = concentrations.apply(
        calculate_row_aqi,
        axis=1,
    )

    # Temporal AQI validity is the authoritative pollutant-coverage
    # condition. A row is valid only when all six required pollutant
    # temporal averages are available.
    concentrations["aqi_temporal_valid"] = (
        concentrations["hourly_aqi"].notna()
    )

    return concentrations[
        [
            "city",
            "hour",
            "hourly_aqi",
            "aqi_temporal_valid",
        ]
    ]


def load_hourly_features(
    engine: Engine,
) -> pd.DataFrame:
    """
    Build the city-level hourly forecasting feature table.

    Air-quality processing:

        raw DB measurements
            -> unit normalization
            -> hourly city aggregation
            -> strict CPCB temporal averaging
            -> AQI

    Forecast target:

        AQI at exactly t + 1 hour.
    """

    measurement_query = text(
        """
        SELECT
            s.city,
            DATE_TRUNC('hour', m.ts) AS hour,
            m.station_id,
            m.pollutant,
            m.value,
            m.unit
        FROM measurements m
        JOIN stations s
            ON s.station_id = m.station_id
        WHERE
            s.city IN ('Delhi', 'Bengaluru')
            AND m.pollutant IN (
                'pm25',
                'pm10',
                'no2',
                'so2',
                'o3',
                'co'
            )
        ORDER BY
            s.city,
            hour,
            m.pollutant
        """
    )

    weather_query = text(
        """
        SELECT
            city,
            DATE_TRUNC('hour', ts) AS hour,
            temp_c,
            humidity,
            wind_speed,
            wind_dir,
            blh
        FROM weather
        WHERE city IN ('Delhi', 'Bengaluru')
        ORDER BY
            city,
            hour
        """
    )

    total_station_query = text(
        """
        SELECT
            s.city,
            DATE_TRUNC('hour', m.ts) AS hour,
            COUNT(DISTINCT m.station_id) AS total_station_count
        FROM measurements m
        JOIN stations s
            ON s.station_id = m.station_id
        WHERE
            s.city IN ('Delhi', 'Bengaluru')
            AND m.pollutant IN (
                'pm25',
                'pm10',
                'no2',
                'so2',
                'o3',
                'co'
            )
        GROUP BY
            s.city,
            DATE_TRUNC('hour', m.ts)
        """
    )

    with engine.connect() as connection:

        measurements = pd.read_sql(
            measurement_query,
            connection,
        )

        weather = pd.read_sql(
            weather_query,
            connection,
        )

        total_station_counts = pd.read_sql(
            total_station_query,
            connection,
        )

    if measurements.empty:
        return pd.DataFrame()

    measurements["hour"] = pd.to_datetime(
        measurements["hour"],
        utc=True,
    )

    if not weather.empty:

        weather["hour"] = pd.to_datetime(
            weather["hour"],
            utc=True,
        )

    total_station_counts["hour"] = pd.to_datetime(
        total_station_counts["hour"],
        utc=True,
    )

    # ---------------------------------------------------------------
    # Unit normalization
    # ---------------------------------------------------------------

    measurements = _normalize_measurement_units(
        measurements
    )

    # ---------------------------------------------------------------
    # Hourly city-level pollutant concentrations
    # ---------------------------------------------------------------

    hourly_measurements = (
        measurements
        .groupby(
            [
                "city",
                "hour",
                "pollutant",
            ],
            as_index=False,
        )
        .agg(
            value=("value", "mean"),
            station_count=("station_id", "nunique"),
        )
    )

    concentrations = (
        hourly_measurements
        .pivot_table(
            index=[
                "city",
                "hour",
            ],
            columns="pollutant",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    concentrations.columns.name = None

    station_counts = (
        hourly_measurements
        .pivot_table(
            index=[
                "city",
                "hour",
            ],
            columns="pollutant",
            values="station_count",
            aggfunc="first",
        )
        .reset_index()
    )

    station_counts.columns.name = None

    station_counts = station_counts.rename(
        columns={
            pollutant: f"{pollutant}_station_count"
            for pollutant in POLLUTANTS
        }
    )

    df = concentrations.merge(
        station_counts,
        on=[
            "city",
            "hour",
        ],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Total station coverage
    # ---------------------------------------------------------------

    df = df.merge(
        total_station_counts,
        on=[
            "city",
            "hour",
        ],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Weather
    # ---------------------------------------------------------------

    if not weather.empty:

        df = df.merge(
            weather,
            on=[
                "city",
                "hour",
            ],
            how="left",
            validate="one_to_one",
        )

    df = (
        df
        .sort_values(
            [
                "city",
                "hour",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # PM2.5 coverage gate
    # ---------------------------------------------------------------

    df["pm25_coverage_valid"] = (
        df.apply(
            lambda row: (
                pd.notna(row.get("pm25"))
                and pd.notna(
                    row.get("pm25_station_count")
                )
                and row["pm25_station_count"]
                >= MIN_PM25_STATIONS.get(
                    row["city"],
                    1,
                )
            ),
            axis=1,
        )
    )

    # ---------------------------------------------------------------
    # Strict CPCB temporal AQI
    # ---------------------------------------------------------------

    temporal_aqi = _build_temporal_aqi_table(
        hourly_measurements
    )

    df = df.merge(
        temporal_aqi,
        on=[
            "city",
            "hour",
        ],
        how="left",
        validate="one_to_one",
    )

    # Invalidate AQI when the existing PM2.5 coverage gate fails.
    df.loc[
        ~df["pm25_coverage_valid"],
        "hourly_aqi",
    ] = pd.NA

    # ---------------------------------------------------------------
    # Temporal features
    # ---------------------------------------------------------------

    df["hour_of_day"] = (
        df["hour"].dt.hour
    )

    df["day_of_week"] = (
        df["hour"].dt.dayofweek
    )

    # ---------------------------------------------------------------
    # Exact pollutant lags
    # ---------------------------------------------------------------

    df = _add_exact_lag_features(
        df
    )

    # ---------------------------------------------------------------
    # Experimental rolling pollutant features
    # ---------------------------------------------------------------

    df = _add_time_based_rolling_features(
        df
    )

    # ---------------------------------------------------------------
    # Exact one-hour-ahead AQI target
    # ---------------------------------------------------------------

    target_lookup = (
        df[
            [
                "city",
                "hour",
                "hourly_aqi",
            ]
        ]
        .copy()
    )

    target_lookup["hour"] = (
        target_lookup["hour"]
        - pd.Timedelta(hours=1)
    )

    target_lookup = target_lookup.rename(
        columns={
            "hourly_aqi": "target_aqi_next_hour",
        }
    )

    df = df.merge(
        target_lookup,
        on=[
            "city",
            "hour",
        ],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Final deterministic ordering
    # ---------------------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "city",
                "hour",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------------

    required_columns = {
        "city",
        "hour",
        "hourly_aqi",
        "target_aqi_next_hour",
        "pm25_coverage_valid",
        "hour_of_day",
        "day_of_week",
    }

    missing_required = (
        required_columns
        - set(df.columns)
    )

    if missing_required:
        raise ValueError(
            "Required feature columns are missing: "
            f"{sorted(missing_required)}"
        )

    duplicate_rows = (
        df.duplicated(
            subset=[
                "city",
                "hour",
            ]
        )
        .sum()
    )

    if duplicate_rows:
        raise ValueError(
            "Duplicate city-hour rows detected "
            f"in hourly feature table: {duplicate_rows}"
        )

    for city, city_group in df.groupby(
        "city",
        sort=False,
    ):

        if not city_group["hour"].is_monotonic_increasing:

            raise ValueError(
                "Hourly timestamps are not sorted for city: "
                f"{city}"
            )

    # AQI must always be within the CPCB 0-500 range.
    invalid_aqi = df[
        df["hourly_aqi"].notna()
        & (
            (df["hourly_aqi"] < 0)
            | (df["hourly_aqi"] > 500)
        )
    ]

    if not invalid_aqi.empty:

        raise ValueError(
            "AQI values outside [0, 500] detected: "
            f"{len(invalid_aqi)} rows"
        )

    return df

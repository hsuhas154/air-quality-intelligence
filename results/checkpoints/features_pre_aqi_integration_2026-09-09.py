from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from air_quality_intelligence.analysis.aqi import calculate_aqi


# ============================================================================
# Configuration
# ============================================================================

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "o3",
    "co",
]


# Minimum number of stations reporting PM2.5 required before an hourly
# city-level AQI is considered valid.
#
# These thresholds are based on the currently observed station network:
#
# Bengaluru:
#     maximum observed PM2.5 station coverage ~= 10
#     minimum acceptable coverage = 5
#
# Delhi:
#     maximum observed PM2.5 station coverage ~= 61
#     minimum acceptable coverage = 31
#
# PM2.5 is used as the primary coverage gate because it is consistently
# available and is the primary pollutant used in the current forecasting
# workflow.
MIN_PM25_STATIONS = {
    "Bengaluru": 5,
    "Delhi": 31,
}


# Exact historical offsets used for pollutant lag features.
#
# IMPORTANT:
# These represent actual elapsed time, NOT dataframe row positions.
#
# For example:
#
#     current hour = 13:00
#     lag1         = concentration at exactly 12:00
#     lag3         = concentration at exactly 10:00
#
# If 12:00 does not exist, lag1 must be NaN.
LAG_HOURS = [
    1,
    3,
    6,
    24,
]


# Pollutants for which short historical rolling features are currently
# constructed.
ROLLING_POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "o3",
]


# Rolling windows are expressed as actual time durations.
#
# A 3-hour window at timestamp t contains observations in the preceding
# three-hour interval ending at t.
ROLLING_WINDOWS_HOURS = [
    3,
    6,
]


# ============================================================================
# Exact time-based lag features
# ============================================================================

def _add_exact_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add exact time-based pollutant lag features.

    Unlike pandas groupby().shift(), these lags are based on timestamps.

    Example
    -------
    Suppose the available observations are:

        10:00
        11:00
        13:00

    At 13:00:

        lag1 = value at 12:00 -> NaN
        lag3 = value at 10:00 -> valid

    The 11:00 observation is NOT incorrectly treated as lag1.

    This is important because the forecasting target is explicitly defined
    as the AQI at exactly t + 1 hour. Feature construction should use the
    same temporal discipline.
    """

    df = df.copy()

    # Ensure deterministic ordering before creating lag features.
    df = df.sort_values(
        ["city", "hour"]
    ).reset_index(drop=True)

    for pollutant in POLLUTANTS:

        if pollutant not in df.columns:
            continue

        # Keep only the columns required for the lookup.
        source = df[
            [
                "city",
                "hour",
                pollutant,
            ]
        ].copy()

        # Rename the pollutant temporarily so that the generated lag
        # column names can be controlled explicitly.
        source = source.rename(
            columns={
                pollutant: "value",
            }
        )

        lagged_frames = []

        for lag_hours in LAG_HOURS:

            lagged = source.copy()

            # Move the source timestamp FORWARD.
            #
            # A source value observed at:
            #
            #     10:00
            #
            # becomes the lag-3 value for:
            #
            #     13:00
            #
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

        # Merge all lag versions for this pollutant together.
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

        # Merge the exact-time lag values back onto the main dataframe.
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


# ============================================================================
# Time-based rolling features
# ============================================================================

def _add_time_based_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add time-based rolling pollutant averages.

    The previous implementation used:

        rolling(3)
        rolling(6)

    which operates on dataframe rows rather than elapsed time.

    That becomes problematic when an hourly observation is missing.

    This implementation instead uses actual timestamp-based windows:

        rolling("3h")
        rolling("6h")

    Therefore a missing hour creates a genuine gap rather than causing
    observations from further in the past to be pulled into the window.

    The current value is included in the rolling average.

    These features are retained for experimentation. The current selected
    forecasting feature set may or may not use them.
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

            # Create an empty Series indexed exactly like df.
            rolling_values = pd.Series(
                index=df.index,
                dtype="float64",
            )

            # Process each city independently so that observations from
            # different cities can never enter the same rolling window.
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

                # Map timestamp -> rolling value back to the original
                # dataframe rows.
                timestamp_to_value = city_rolling.to_dict()

                rolling_values.loc[city_indices] = (
                    city_group["hour"]
                    .map(timestamp_to_value)
                    .to_numpy()
                )

            df[rolling_column] = rolling_values

    return df


# ============================================================================
# Main feature-building function
# ============================================================================

def load_hourly_features(
    engine: Engine,
) -> pd.DataFrame:
    """
    Build the city-level hourly forecasting feature table.

    The resulting dataframe contains:

    1. Hourly city-level pollutant concentrations.
    2. Pollutant-specific station counts.
    3. Total station coverage.
    4. Weather variables.
    5. PM2.5 coverage validity.
    6. Current-hour AQI.
    7. Temporal features.
    8. Exact-time pollutant lag features.
    9. Time-based rolling pollutant features.
    10. An exact one-hour-ahead AQI target.

    Important temporal guarantee
    ----------------------------
    The target at timestamp t is the AQI at exactly t + 1 hour.

    Likewise:

        lag1  = pollutant concentration at exactly t - 1 hour
        lag3  = pollutant concentration at exactly t - 3 hours
        lag6  = pollutant concentration at exactly t - 6 hours
        lag24 = pollutant concentration at exactly t - 24 hours

    Missing timestamps therefore produce NaN lag values rather than silently
    substituting an older available observation.
    """

    # ========================================================================
    # 1. Load hourly pollutant concentrations AND station coverage.
    # ========================================================================

    measurement_query = text(
        """
        SELECT
            s.city,
            DATE_TRUNC('hour', m.ts) AS hour,
            m.pollutant,
            AVG(m.value) AS value,
            COUNT(DISTINCT m.station_id) AS station_count
        FROM measurements m
        JOIN stations s
            ON s.station_id = m.station_id
        WHERE m.pollutant IN (
            'pm25',
            'pm10',
            'no2',
            'so2',
            'o3',
            'co'
        )
        GROUP BY
            s.city,
            DATE_TRUNC('hour', m.ts),
            m.pollutant
        ORDER BY
            s.city,
            hour,
            m.pollutant
        """
    )

    # ========================================================================
    # 2. Load hourly weather.
    # ========================================================================

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

    with engine.connect() as connection:

        measurements = pd.read_sql(
            measurement_query,
            connection,
        )

        weather = pd.read_sql(
            weather_query,
            connection,
        )

    # ========================================================================
    # 3. Basic empty-data protection.
    # ========================================================================

    if measurements.empty:
        return pd.DataFrame()

    # ========================================================================
    # 4. Normalize timestamp types.
    #
    # PostgreSQL may return timezone-aware timestamps depending on the
    # database column definition and driver configuration.
    #
    # Keeping timestamps as pandas datetime values ensures that the exact
    # Timedelta-based lag operations below behave predictably.
    # ========================================================================

    measurements["hour"] = pd.to_datetime(
        measurements["hour"],
        utc=True,
    )

    if not weather.empty:
        weather["hour"] = pd.to_datetime(
            weather["hour"],
            utc=True,
        )

    # ========================================================================
    # 5. Build pollutant concentration table.
    # ========================================================================

    concentrations = (
        measurements
        .pivot_table(
            index=[
                "city",
                "hour",
            ],
            columns="pollutant",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    concentrations.columns.name = None

    # ========================================================================
    # 6. Build pollutant station-count table.
    #
    # Example columns:
    #
    #     pm25_station_count
    #     pm10_station_count
    #     no2_station_count
    #     so2_station_count
    #     o3_station_count
    #     co_station_count
    # ========================================================================

    station_counts = (
        measurements
        .pivot_table(
            index=[
                "city",
                "hour",
            ],
            columns="pollutant",
            values="station_count",
            aggfunc="max",
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

    # ========================================================================
    # 7. Merge concentrations and pollutant-specific station coverage.
    # ========================================================================

    df = concentrations.merge(
        station_counts,
        on=[
            "city",
            "hour",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================================
    # 8. Calculate total distinct station coverage.
    #
    # This is deliberately calculated separately.
    #
    # Summing pollutant station counts would double-count stations that
    # report multiple pollutants.
    # ========================================================================

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

        total_station_counts = pd.read_sql(
            total_station_query,
            connection,
        )

    total_station_counts["hour"] = pd.to_datetime(
        total_station_counts["hour"],
        utc=True,
    )

    df = df.merge(
        total_station_counts,
        on=[
            "city",
            "hour",
        ],
        how="left",
        validate="one_to_one",
    )

    # ========================================================================
    # 9. Merge weather.
    # ========================================================================

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

    # ========================================================================
    # 10. Sort the complete feature table chronologically.
    # ========================================================================

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

    # ========================================================================
    # 11. Determine PM2.5 coverage validity.
    #
    # An hourly AQI is only considered valid when:
    #
    #     1. PM2.5 concentration exists.
    #     2. PM2.5 station coverage reaches the city-specific threshold.
    #
    # The coverage gate prevents sparse station observations from producing
    # artificial AQI spikes.
    # ========================================================================

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

    # ========================================================================
    # 12. Calculate hourly AQI.
    #
    # Secondary pollutants remain optional.
    #
    # The current implementation intentionally preserves the existing
    # project's CPCB-style hourly approximation:
    #
    #     calculate_aqi(concentrations)
    #
    # The averaging-period hardening for the final production AQI pipeline
    # will be handled separately.
    # ========================================================================

    def calculate_row_aqi(
        row: pd.Series,
    ) -> int | None:

        if not row["pm25_coverage_valid"]:
            return None

        concentrations = {
            pollutant: row[pollutant]
            for pollutant in POLLUTANTS
            if (
                pollutant in row.index
                and pd.notna(row[pollutant])
            )
        }

        if not concentrations:
            return None

        aqi, _ = calculate_aqi(
            concentrations
        )

        return aqi

    df["hourly_aqi"] = df.apply(
        calculate_row_aqi,
        axis=1,
    )

    # ========================================================================
    # 13. Temporal features.
    # ========================================================================

    df["hour_of_day"] = (
        df["hour"].dt.hour
    )

    df["day_of_week"] = (
        df["hour"].dt.dayofweek
    )

    # ========================================================================
    # 14. Exact-time pollutant lags.
    #
    # IMPORTANT:
    #
    # This replaces the old:
    #
    #     groupby("city").shift(...)
    #
    # implementation.
    #
    # The old implementation meant:
    #
    #     "previous dataframe row"
    #
    # whereas the forecasting problem requires:
    #
    #     "value exactly N hours ago".
    #
    # The new helper enforces the latter.
    # ========================================================================

    df = _add_exact_lag_features(
        df
    )

    # ========================================================================
    # 15. Time-based rolling averages.
    #
    # These are retained for future feature experiments.
    #
    # Unlike rolling(3)/rolling(6), these windows use elapsed time rather
    # than row count.
    # ========================================================================

    df = _add_time_based_rolling_features(
        df
    )

    # ========================================================================
    # 16. Construct the TRUE one-hour-ahead AQI target.
    #
    # We deliberately DO NOT use:
    #
    #     df.groupby("city")["hourly_aqi"].shift(-1)
    #
    # because the next dataframe row may not be exactly one hour later.
    #
    # Instead, the target is constructed through an exact timestamp lookup.
    #
    # Current:
    #
    #     10:00
    #
    # Target:
    #
    #     11:00
    #
    # If 11:00 is absent or its AQI is invalid, the target is NaN.
    # ========================================================================

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

    # Move the target timestamp backward so that:
    #
    #     target 11:00
    #
    # becomes attached to:
    #
    #     current 10:00
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

    # ========================================================================
    # 17. Final deterministic ordering.
    # ========================================================================

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

    # ========================================================================
    # 18. Final sanity checks.
    #
    # These checks intentionally fail loudly if the feature table violates
    # assumptions required by the forecasting pipeline.
    # ========================================================================

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

    # The feature table should contain at most one row per city-hour.
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

    # Verify timestamp ordering within every city.
    for city, city_group in df.groupby(
        "city",
        sort=False,
    ):

        if not city_group["hour"].is_monotonic_increasing:
            raise ValueError(
                f"Hourly timestamps are not sorted for city: "
                f"{city}"
            )

    return df
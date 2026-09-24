from __future__ import annotations

import math

# CPCB AQI breakpoints for the pollutants used by the project.
# Each tuple is:
# (concentration_low, concentration_high, AQI_low, AQI_high)
BREAKPOINTS = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 51, 100),
        (60.0, 90.0, 101, 200),
        (90.0, 120.0, 201, 300),
        (120.0, 250.0, 301, 400),
        (250.0, math.inf, 401, 500),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 250.0, 101, 200),
        (250.0, 350.0, 201, 300),
        (350.0, 430.0, 301, 400),
        (430.0, math.inf, 401, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 180.0, 101, 200),
        (180.0, 280.0, 201, 300),
        (280.0, 400.0, 301, 400),
        (400.0, math.inf, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 380.0, 101, 200),
        (380.0, 800.0, 201, 300),
        (800.0, 1600.0, 301, 400),
        (1600.0, math.inf, 401, 500),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 168.0, 101, 200),
        (168.0, 208.0, 201, 300),
        (208.0, 748.0, 301, 400),
        (748.0, math.inf, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, math.inf, 401, 500),
    ],
}


def calculate_sub_index(pollutant: str, concentration: float) -> int | None:
    """Calculate a pollutant sub-index from its concentration."""

    if pollutant not in BREAKPOINTS:
        return None

    if concentration < 0:
        return None

    for c_low, c_high, i_low, i_high in BREAKPOINTS[pollutant]:
        if c_low <= concentration <= c_high:
            sub_index = (
                (i_high - i_low)
                / (c_high - c_low)
                * (concentration - c_low)
                + i_low
            )
            return round(sub_index)

    return 500


def calculate_aqi(
    concentrations: dict[str, float],
) -> tuple[int, str]:
    """Calculate overall AQI and dominant pollutant."""

    sub_indices = {}

    for pollutant, concentration in concentrations.items():
        sub_index = calculate_sub_index(pollutant, concentration)

        if sub_index is not None:
            sub_indices[pollutant] = sub_index

    if not sub_indices:
        raise ValueError("No supported pollutants available for AQI calculation.")

    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    aqi = sub_indices[dominant_pollutant]

    return aqi, dominant_pollutant

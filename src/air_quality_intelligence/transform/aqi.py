from __future__ import annotations

# CPCB-style breakpoint tables for the six pollutants used in the MVP.
# Values are concentration ranges; interpolation follows the Indian AQI sub-index method.
BREAKPOINTS = {
    "pm2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "pm10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "no2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 800, 401, 500)],
    "so2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2400, 401, 500)],
    "co": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "o3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
}


def sub_index(pollutant: str, concentration: float) -> float:
    """Calculate an AQI sub-index by linear interpolation within CPCB breakpoints."""
    key = pollutant.lower()
    if key not in BREAKPOINTS:
        raise ValueError(f"Unsupported pollutant: {pollutant}")
    for c_low, c_high, i_low, i_high in BREAKPOINTS[key]:
        if c_low <= concentration <= c_high:
            return i_low + (i_high - i_low) * (concentration - c_low) / (c_high - c_low)
    return 500.0 if concentration > BREAKPOINTS[key][-1][1] else 0.0


def calculate_aqi(values: dict[str, float]) -> tuple[int, str]:
    indices = {pollutant: sub_index(pollutant, value) for pollutant, value in values.items() if value is not None}
    if not indices:
        raise ValueError("At least one pollutant concentration is required.")
    dominant = max(indices, key=indices.get)
    return round(max(indices.values())), dominant

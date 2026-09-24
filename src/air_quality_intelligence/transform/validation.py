"""Physical plausibility checks for normalized pollutant concentrations.

Unit normalization can only be as correct as the unit label supplied by the
data provider. If a provider mislabels a feed, the conversion will be applied
faithfully and produce a value that is arithmetically correct and physically
impossible.

This module is the guard against that: after normalization, every
concentration is checked against the range that pollutant can plausibly take
in ambient urban air. Values outside the range indicate a unit problem
upstream, not an unusual air quality episode.

Ranges are deliberately wide. They are intended to catch errors of whole
orders of magnitude, not to reject genuine pollution episodes.
"""

from __future__ import annotations

# Plausible ambient ranges in the canonical AQI units used by this project.
# (minimum, maximum) inclusive.
#
# PM2.5, PM10, NO2, SO2, O3 are in µg/m³.
# CO is in mg/m³.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "pm25": (0.0, 1000.0),
    "pm10": (0.0, 2000.0),
    "no2": (0.0, 500.0),
    "so2": (0.0, 500.0),
    "o3": (0.0, 600.0),
    "co": (0.01, 50.0),
}


class ImplausibleConcentrationError(ValueError):
    """Raised when a normalized concentration is physically impossible."""


def is_plausible(pollutant: str, value: float) -> bool:
    """Return True if a normalized concentration is physically plausible."""

    bounds = PLAUSIBLE_RANGES.get(pollutant)

    if bounds is None:
        return True

    minimum, maximum = bounds

    return minimum <= value <= maximum


def check_plausible(pollutant: str, value: float) -> float:
    """Return the value, or raise if it is physically impossible."""

    if not is_plausible(pollutant, value):
        minimum, maximum = PLAUSIBLE_RANGES[pollutant]
        raise ImplausibleConcentrationError(
            f"Normalized {pollutant} concentration {value} is outside the "
            f"plausible ambient range [{minimum}, {maximum}]. "
            f"This usually means the source unit label is wrong."
        )

    return value

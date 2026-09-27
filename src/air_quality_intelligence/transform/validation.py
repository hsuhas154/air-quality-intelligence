"""Physical plausibility checks for normalized pollutant concentrations.

Unit normalization can only be as correct as the unit label supplied by the
data provider. If a provider mislabels a feed, the conversion is applied
faithfully and produces a value that is arithmetically correct and physically
impossible. That is not hypothetical here: one station family reported CO in
"ppb" when the readings were plainly ppm, which put every CO value three
orders of magnitude low.

This module is the backstop. After normalization, every concentration is
checked against a range it must fall inside if the unit label was right.

The bounds are not chosen by eye. Each is derived from a stated anchor:

Upper bounds
    ``UPPER_BOUND_HEADROOM`` times the highest concentration that pollutant's
    published breakpoint table names, read straight out of
    ``analysis.aqi.BREAKPOINTS`` so the two tables cannot drift apart.

    The multiple has to clear the worst real episode and still sit far below a
    mislabelled value. It clears the worst episode: the highest documented
    hourly mean PM2.5 over northwest India is 1341 ug/m3, measured by a
    high-density sensor network during the November 2022 stubble-burning
    season (Sci Rep, PMC10425363), against a bound here of 2500. It stays far
    below a mislabelled value: the unit confusions this guard exists to catch
    are factors of 1000 (mg against ug, ppm against ppb), so a typical Delhi
    PM2.5 of 100 ug/m3 mislabelled arrives as 100000, which is forty times the
    bound. The two regimes are separated by more than an order of magnitude on
    each side, so the exact multiple inside that gap does not change which
    values the guard rejects.

Lower bounds
    Zero for the five pollutants that can legitimately read zero when they sit
    below the instrument's detection limit.

    CO is the exception and is the reason the lower bound exists at all. CO
    has a global atmospheric floor: Northern Hemisphere background mixing
    ratios have not fallen below roughly 90 ppb even in preindustrial ice-core
    reconstructions (Clim Past 18:631, 2022), which is about 0.103 mg/m3 at
    this project's conversion factor. A city cannot be cleaner than the
    background it sits in, so an ambient CO reading well under that figure is
    a unit error rather than clean air. The bound is set at half the
    preindustrial background to leave room for instrument error, which still
    puts it two orders of magnitude above the 0.00055 mg/m3 that the
    mislabelled feed produced.

Sources for every figure quoted here are recorded in docs/aqi-references.md.
"""

from __future__ import annotations

from air_quality_intelligence.analysis.aqi import BREAKPOINTS
from air_quality_intelligence.transform.units import PPB_TO_MG_M3_CO

# Multiple of the top of the published AQI scale at which a reading stops
# being a pollution episode and starts being a unit error.
UPPER_BOUND_HEADROOM = 10.0

# Lowest Northern Hemisphere background CO in the ice-core record, in ppb.
BACKGROUND_CO_PPB = 90.0

# Ambient CO cannot sit far below the global background. Half of it leaves
# room for instrument error without admitting a thousand-fold unit error.
MINIMUM_PLAUSIBLE_CO_MG_M3 = round(
    0.5 * BACKGROUND_CO_PPB * PPB_TO_MG_M3_CO,
    3,
)

# Lower bounds that are not zero. Every other pollutant can legitimately read
# zero, because a value below the instrument's detection limit is commonly
# reported as zero, so only CO carries a floor and only for the reason given
# in the module docstring.
_NON_ZERO_MINIMUMS: dict[str, float] = {"co": MINIMUM_PLAUSIBLE_CO_MG_M3}


def _scale_top(pollutant: str) -> float:
    """Highest concentration the published breakpoint table names.

    Above this the official sub-index carries on by extending the last band's
    line, so this is where the published table stops rather than where the
    arithmetic stops.
    """
    _, c_high, _, _ = BREAKPOINTS[pollutant][-1]

    return c_high


def _build_ranges() -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}

    for pollutant in BREAKPOINTS:
        lower = _NON_ZERO_MINIMUMS.get(pollutant, 0.0)
        ranges[pollutant] = (lower, UPPER_BOUND_HEADROOM * _scale_top(pollutant))

    return ranges


# Plausible ambient ranges in the canonical AQI units used by this project,
# (minimum, maximum) inclusive.
#
# PM2.5, PM10, NO2, SO2 and O3 are in ug/m3. CO is in mg/m3.
#
# Derived, not typed in, so that a change to the AQI breakpoint table carries
# through to the guard.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = _build_ranges()


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

"""CPCB National Air Quality Index.

Breakpoints, the sub-index formula and the aggregation rules published for
India's National AQI. Every constant and every rule in this module is taken
from CPCB's own calculator spreadsheet, AQI-Calculator.xls, cell by cell.
docs/aqi-references.md records the formulas it was read from. Nothing here is
tuned.

Three properties of the official arithmetic are worth knowing before reading
the code, because each one is easy to get wrong and this module got two of
them wrong before the spreadsheet was available:

1.  The sub-index is anchored on the *previous* AQI category's upper limit,
    not on the displayed lower bound of the current band. So the PM2.5 band
    labelled 301 to 400 anchors at 300.
2.  Above the highest published breakpoint the line simply continues, using
    the slope of the last band. The top band is open, not closed, and the
    arithmetic has no ceiling.
3.  A concentration of zero or below counts as *missing*, not as a clean
    reading, when deciding whether enough pollutants are present.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# The published category scale runs 0 to 500 and CPCB does not report above
# it. The arithmetic below is uncapped, matching the official calculator, so
# this is a reporting limit applied on the way out, not a limit on the value.
AQI_REPORTING_MAXIMUM = 500

# An overall AQI is published only when at least three pollutants are present
# and at least one of them is PM2.5 or PM10.
MINIMUM_POLLUTANTS_FOR_AQI = 3
MANDATORY_POLLUTANTS: frozenset[str] = frozenset({"pm25", "pm10"})

# A sub-index needs at least 16 hours of data within its averaging period.
MINIMUM_HOURS_FOR_SUB_INDEX = 16

# Each band is (concentration_low, concentration_high, index_low, index_high).
#
# index_low is the upper limit of the previous AQI category, which is the
# anchor the official formula uses:
#
#     sub-index = previous category's upper AQI limit
#                 + (reading - previous band's upper concentration)
#                   * (AQI band width / concentration band width)
#
# Anchoring this way makes the sub-index continuous across band boundaries and
# reproduces the official spreadsheet exactly. Anchoring on the displayed lower
# bound instead (51, 101, 201, 301) shifts every value by up to one point.
#
# There is no sixth band. Above the last entry the line continues at that
# entry's slope; see _extrapolate below.
BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 50, 100),
        (60.0, 90.0, 100, 200),
        (90.0, 120.0, 200, 300),
        (120.0, 250.0, 300, 400),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 50, 100),
        (100.0, 250.0, 100, 200),
        (250.0, 350.0, 200, 300),
        (350.0, 430.0, 300, 400),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 50, 100),
        (80.0, 180.0, 100, 200),
        (180.0, 280.0, 200, 300),
        (280.0, 400.0, 300, 400),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 50, 100),
        (80.0, 380.0, 100, 200),
        (380.0, 800.0, 200, 300),
        (800.0, 1600.0, 300, 400),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 50, 100),
        (100.0, 168.0, 100, 200),
        (168.0, 208.0, 200, 300),
        # The official sheet divides this band by 539 rather than its actual
        # width of 540, and its open top band subtracts 400 rather than 748.
        # Both are transcription errors: taken literally the second one makes
        # the O3 sub-index jump from 400 to 465 across one unit of ozone. The
        # consistent rule every other pollutant follows is used instead, and
        # test_o3_top_band_is_continuous pins it.
        (208.0, 748.0, 300, 400),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.0, 2.0, 50, 100),
        (2.0, 10.0, 100, 200),
        (10.0, 17.0, 200, 300),
        (17.0, 34.0, 300, 400),
    ],
    # NH3 is in the official calculator and counts towards the three-pollutant
    # rule. This project does not ingest it, but a calculator claiming to
    # implement the National AQI should not quietly omit a pollutant the
    # official one includes.
    "nh3": [
        (0.0, 200.0, 0, 50),
        (200.0, 400.0, 50, 100),
        (400.0, 800.0, 100, 200),
        (800.0, 1200.0, 200, 300),
        (1200.0, 1800.0, 300, 400),
    ],
}

SUPPORTED_POLLUTANTS: frozenset[str] = frozenset(BREAKPOINTS)


class InsufficientPollutantsError(ValueError):
    """Raised when too few pollutants are present to publish an AQI."""


@dataclass(frozen=True)
class SubIndex:
    """One pollutant's sub-index and how it was obtained.

    value
        The sub-index as the official calculator computes it. Uncapped: above
        the highest published breakpoint the arithmetic keeps going, and so
        does this.
    extrapolated
        True when the reading sits above the highest published breakpoint, so
        value comes from continuing the last band's slope rather than from
        interpolating between two published points. The official calculator
        does exactly this, but the number leaves the published 0 to 500 scale,
        which is worth knowing when one is reported.
    """

    pollutant: str
    concentration: float
    value: int
    extrapolated: bool

    @property
    def reported_value(self) -> int:
        """The sub-index as CPCB would publish it, capped at the scale top."""

        return min(self.value, AQI_REPORTING_MAXIMUM)


def _round_half_up(value: float) -> int:
    """Round half away from zero, matching the spreadsheet's ROUND."""

    return int(math.floor(value + 0.5))


def _clean(pollutant: object) -> str:
    return str(pollutant).strip().lower()


def _usable(concentration: object) -> float | None:
    """Return a concentration as a float, or None if it cannot be used."""

    try:
        value = float(concentration)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    if math.isnan(value) or math.isinf(value):
        return None

    return value


def is_present(pollutant: str, concentration: object) -> bool:
    """Whether a reading counts as present for the minimum-pollutant rule.

    The official calculator marks a pollutant present with
    ``IF(OR(ISTEXT(C),C<=0),0,1)``, so a reading of zero or below is treated
    as missing rather than as a measurement of clean air.
    """

    if _clean(pollutant) not in BREAKPOINTS:
        return False

    value = _usable(concentration)

    return value is not None and value > 0


def calculate_sub_index_detail(
    pollutant: str,
    concentration: float,
) -> SubIndex | None:
    """Calculate a pollutant sub-index, reporting how it was obtained.

    Returns None when the pollutant is not part of the National AQI or the
    concentration is not a usable non-negative number.
    """

    key = _clean(pollutant)

    if key not in BREAKPOINTS:
        return None

    value = _usable(concentration)

    if value is None or value < 0:
        return None

    bands = BREAKPOINTS[key]

    for c_low, c_high, i_low, i_high in bands:
        if c_low <= value <= c_high:
            slope = (i_high - i_low) / (c_high - c_low)

            return SubIndex(
                pollutant=key,
                concentration=value,
                value=_round_half_up(i_low + slope * (value - c_low)),
                extrapolated=False,
            )

    # Above the highest published breakpoint. The official calculator
    # continues the last band's line rather than capping, so the sub-index can
    # exceed 500; see SubIndex.reported_value for the published form.
    c_low, c_high, i_low, i_high = bands[-1]
    slope = (i_high - i_low) / (c_high - c_low)

    return SubIndex(
        pollutant=key,
        concentration=value,
        value=_round_half_up(i_high + slope * (value - c_high)),
        extrapolated=True,
    )


def calculate_sub_index(pollutant: str, concentration: float) -> int | None:
    """Calculate a pollutant sub-index from its concentration."""

    detail = calculate_sub_index_detail(pollutant, concentration)

    return None if detail is None else detail.value


def has_sufficient_pollutants(
    concentrations: Mapping[str, object] | Iterable[str],
) -> bool:
    """Check the published minimum-pollutant rule without raising.

    At least three supported pollutants must be present and at least one of
    them must be PM2.5 or PM10. Presence follows the official calculator: a
    reading of zero or below does not count.

    Accepts a mapping of pollutant to concentration. A bare collection of
    names is also accepted, and every name in it is taken as present, which is
    only correct when the caller has already applied the zero rule.
    """

    if isinstance(concentrations, Mapping):
        available = {
            _clean(pollutant)
            for pollutant, concentration in concentrations.items()
            if is_present(pollutant, concentration)
        }
    else:
        try:
            available = {_clean(pollutant) for pollutant in concentrations}
        except TypeError:
            return False

        available &= SUPPORTED_POLLUTANTS

    if len(available) < MINIMUM_POLLUTANTS_FOR_AQI:
        return False

    return bool(available & MANDATORY_POLLUTANTS)


def calculate_aqi_detail(
    concentrations: Mapping[str, float],
    *,
    enforce_minimum_pollutants: bool = True,
) -> tuple[SubIndex, dict[str, SubIndex]]:
    """Calculate the overall AQI and return every sub-index behind it.

    The overall AQI is the maximum sub-index and the dominant pollutant is the
    one holding it. Pollutants that are absent under the zero rule are left
    out of the returned sub-indices entirely.

    enforce_minimum_pollutants applies the published rule that an AQI needs
    three pollutants including PM2.5 or PM10. Set it to False only where the
    caller has already guaranteed a fixed pollutant set.
    """

    details: dict[str, SubIndex] = {}

    for pollutant, concentration in concentrations.items():
        if not is_present(pollutant, concentration):
            continue

        detail = calculate_sub_index_detail(pollutant, concentration)

        if detail is not None:
            details[detail.pollutant] = detail

    if not details:
        raise InsufficientPollutantsError(
            "No supported pollutants available for AQI calculation."
        )

    # details already holds only the pollutants that passed the zero rule, so
    # the name-only form is the right one here. Passing the mapping itself
    # would re-test SubIndex objects as if they were concentrations.
    if enforce_minimum_pollutants and not has_sufficient_pollutants(set(details)):
        raise InsufficientPollutantsError(
            "An AQI needs at least "
            f"{MINIMUM_POLLUTANTS_FOR_AQI} pollutants including PM2.5 or "
            f"PM10; got {sorted(details)}."
        )

    dominant = max(details.values(), key=lambda detail: detail.value)

    return dominant, details


def calculate_aqi(
    concentrations: Mapping[str, float],
    *,
    enforce_minimum_pollutants: bool = True,
) -> tuple[int, str]:
    """Calculate overall AQI and dominant pollutant."""

    dominant, _ = calculate_aqi_detail(
        concentrations,
        enforce_minimum_pollutants=enforce_minimum_pollutants,
    )

    return dominant.value, dominant.pollutant

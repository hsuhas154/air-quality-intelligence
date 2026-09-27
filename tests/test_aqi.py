"""The AQI calculator, checked against CPCB's own spreadsheet.

`reference_sub_index` below is a direct transcription of the cell formulas in
AQI-Calculator.xls. The tests compare this module against that transcription
across a dense grid, so any drift from the official arithmetic fails here
rather than showing up as a plausible but wrong number in the dataset.
"""

import math

import pytest

from air_quality_intelligence.analysis.aqi import (
    AQI_REPORTING_MAXIMUM,
    BREAKPOINTS,
    InsufficientPollutantsError,
    calculate_aqi,
    calculate_aqi_detail,
    calculate_sub_index,
    calculate_sub_index_detail,
    has_sufficient_pollutants,
    is_present,
)

SIX = {"pm25": 24.4, "pm10": 77.3, "no2": 47.4, "so2": 28.0, "o3": 19.5, "co": 0.55}


def reference_sub_index(pollutant, c):
    """CPCB's spreadsheet formulas, transcribed from the cells D8 to D20.

    The O3 top band is the one departure: the sheet writes
    `400+(C18-400)*(100/539)`, which subtracts the wrong breakpoint. The
    consistent form is used here, and test_o3_top_band_is_continuous shows
    why the literal one cannot be right.
    """
    if pollutant == "pm10":
        if c <= 50:
            return c
        if c <= 100:
            return c
        if c <= 250:
            return 100 + (c - 100) * 100 / 150
        if c <= 350:
            return 200 + (c - 250)
        if c <= 430:
            return 300 + (c - 350) * (100 / 80)
        return 400 + (c - 430) * (100 / 80)

    if pollutant == "pm25":
        if c <= 30:
            return c * 50 / 30
        if c <= 60:
            return 50 + (c - 30) * 50 / 30
        if c <= 90:
            return 100 + (c - 60) * 100 / 30
        if c <= 120:
            return 200 + (c - 90) * (100 / 30)
        if c <= 250:
            return 300 + (c - 120) * (100 / 130)
        return 400 + (c - 250) * (100 / 130)

    if pollutant == "so2":
        if c <= 40:
            return c * 50 / 40
        if c <= 80:
            return 50 + (c - 40) * 50 / 40
        if c <= 380:
            return 100 + (c - 80) * 100 / 300
        if c <= 800:
            return 200 + (c - 380) * (100 / 420)
        if c <= 1600:
            return 300 + (c - 800) * (100 / 800)
        return 400 + (c - 1600) * (100 / 800)

    if pollutant == "no2":
        if c <= 40:
            return c * 50 / 40
        if c <= 80:
            return 50 + (c - 40) * 50 / 40
        if c <= 180:
            return 100 + (c - 80) * 100 / 100
        if c <= 280:
            return 200 + (c - 180) * (100 / 100)
        if c <= 400:
            return 300 + (c - 280) * (100 / 120)
        return 400 + (c - 400) * (100 / 120)

    if pollutant == "co":
        if c <= 1:
            return c * 50 / 1
        if c <= 2:
            return 50 + (c - 1) * 50 / 1
        if c <= 10:
            return 100 + (c - 2) * 100 / 8
        if c <= 17:
            return 200 + (c - 10) * (100 / 7)
        if c <= 34:
            return 300 + (c - 17) * (100 / 17)
        return 400 + (c - 34) * (100 / 17)

    if pollutant == "o3":
        if c <= 50:
            return c * 50 / 50
        if c <= 100:
            return 50 + (c - 50) * 50 / 50
        if c <= 168:
            return 100 + (c - 100) * 100 / 68
        if c <= 208:
            return 200 + (c - 168) * (100 / 40)
        if c <= 748:
            return 300 + (c - 208) * (100 / 540)
        return 400 + (c - 748) * (100 / 540)

    if pollutant == "nh3":
        if c <= 200:
            return c * 50 / 200
        if c <= 400:
            return 50 + (c - 200) * 50 / 200
        if c <= 800:
            return 100 + (c - 400) * 100 / 400
        if c <= 1200:
            return 200 + (c - 800) * (100 / 400)
        if c <= 1800:
            return 300 + (c - 1200) * (100 / 600)
        return 400 + (c - 1800) * (100 / 600)

    raise AssertionError(pollutant)


def grid(pollutant, count=400):
    """Concentrations spanning the published range and well past it."""
    top = BREAKPOINTS[pollutant][-1][1]

    return [top * 3.0 * index / count for index in range(count + 1)]


# ---------------------------------------------------------------------------
# Agreement with the official spreadsheet
# ---------------------------------------------------------------------------


def test_matches_the_official_spreadsheet_across_the_whole_range():
    for pollutant in BREAKPOINTS:
        for concentration in grid(pollutant):
            expected = math.floor(reference_sub_index(pollutant, concentration) + 0.5)

            assert calculate_sub_index(pollutant, concentration) == expected, (
                pollutant,
                concentration,
            )


def test_reproduces_the_worked_example_shipped_in_the_spreadsheet():
    """The sheet ships with inputs and cached results. AQI 114 on PM10 121."""
    sample = {
        "pm10": 121,
        "pm25": 34,
        "so2": 0,
        "no2": 8,
        "co": 0,
        "o3": 57,
        "nh3": 34,
    }

    dominant, details = calculate_aqi_detail(sample)

    assert dominant.value == 114
    assert dominant.pollutant == "pm10"
    assert details["pm25"].value == 57
    assert details["no2"].value == 10
    assert details["o3"].value == 57
    assert details["nh3"].value == 9

    # SO2 and CO were entered as zero, which the sheet marks absent.
    assert "so2" not in details
    assert "co" not in details


def test_published_worked_examples_reproduce_exactly():
    """The two worked examples on the government Arthapedia page."""
    assert calculate_sub_index("pm25", 150) == 323
    assert calculate_sub_index("pm25", 45) == 75


# ---------------------------------------------------------------------------
# Structure of the breakpoint table
# ---------------------------------------------------------------------------


def test_every_band_edge_lands_on_a_category_boundary():
    for pollutant, bands in BREAKPOINTS.items():
        for c_low, c_high, i_low, i_high in bands:
            assert calculate_sub_index(pollutant, c_low) == i_low, pollutant
            assert calculate_sub_index(pollutant, c_high) == i_high, pollutant


def test_bands_are_contiguous_and_reach_the_severe_floor():
    for pollutant, bands in BREAKPOINTS.items():
        assert bands[0][0] == 0.0, pollutant
        assert bands[0][2] == 0, pollutant
        assert bands[-1][3] == 400, pollutant

        for earlier, later in zip(bands, bands[1:], strict=False):
            assert earlier[1] == later[0], (pollutant, earlier, later)
            assert earlier[3] == later[2], (pollutant, earlier, later)


def test_no_band_has_an_infinite_bound():
    """An infinite band width returned the band floor for every high reading.

    Delhi's ninety-one day maximum was exactly 401 because of it.
    """
    for pollutant, bands in BREAKPOINTS.items():
        for c_low, c_high, _, _ in bands:
            assert math.isfinite(c_low), pollutant
            assert math.isfinite(c_high), pollutant


# ---------------------------------------------------------------------------
# The top band
# ---------------------------------------------------------------------------


def test_the_top_band_continues_rather_than_stopping():
    values = [calculate_sub_index("pm25", c) for c in (260, 300, 400, 600, 900)]

    assert len(set(values)) == len(values)
    assert values == sorted(values)
    assert values[0] > 400
    assert values[-1] > AQI_REPORTING_MAXIMUM


def test_the_top_band_uses_the_last_band_slope():
    """PM2.5 above 250 continues the 120 to 250 line, 100 AQI per 130 ug/m3."""
    at_top = calculate_sub_index("pm25", 250)
    one_band_higher = calculate_sub_index("pm25", 380)

    assert at_top == 400
    assert one_band_higher == 500


def test_extrapolated_readings_are_flagged_and_interpolated_ones_are_not():
    for pollutant, bands in BREAKPOINTS.items():
        top = bands[-1][1]

        assert calculate_sub_index_detail(pollutant, top).extrapolated is False
        assert calculate_sub_index_detail(pollutant, top * 1.5).extrapolated is True


def test_reported_value_caps_at_the_published_scale():
    detail = calculate_sub_index_detail("pm25", 900)

    assert detail.value > AQI_REPORTING_MAXIMUM
    assert detail.reported_value == AQI_REPORTING_MAXIMUM

    ordinary = calculate_sub_index_detail("pm25", 45)

    assert ordinary.reported_value == ordinary.value


def test_o3_top_band_is_continuous():
    """The official sheet's O3 top band subtracts 400 instead of 748.

    Taken literally it jumps from 400 to 465 across one unit of ozone. This
    test is what stops anyone transcribing that error back in.
    """
    below = calculate_sub_index("o3", 748)
    above = calculate_sub_index("o3", 749)

    assert below == 400
    assert above - below <= 1

    literal_sheet_formula = 400 + (749 - 400) * (100 / 539)

    assert literal_sheet_formula > 460


def test_sub_index_is_monotonic_in_concentration():
    for pollutant in BREAKPOINTS:
        previous = -1

        for concentration in grid(pollutant, count=800):
            sub_index = calculate_sub_index(pollutant, concentration)

            assert sub_index >= previous, (pollutant, concentration)
            previous = sub_index


# ---------------------------------------------------------------------------
# Presence, and the minimum-pollutant rule
# ---------------------------------------------------------------------------


def test_zero_counts_as_absent_not_as_clean_air():
    """The sheet's presence test is IF(OR(ISTEXT(C),C<=0),0,1)."""
    assert is_present("so2", 0) is False
    assert is_present("so2", -1) is False
    assert is_present("so2", 0.1) is True


def test_absent_pollutants_are_left_out_of_the_sub_indices():
    _, details = calculate_aqi_detail({**SIX, "so2": 0.0})

    assert "so2" not in details
    assert len(details) == 5


def test_minimum_pollutant_rule_needs_three_present_pollutants():
    assert has_sufficient_pollutants({"pm25": 40, "no2": 30, "so2": 20}) is True
    assert has_sufficient_pollutants({"pm25": 40, "no2": 30}) is False
    assert has_sufficient_pollutants({"pm25": 40}) is False


def test_minimum_pollutant_rule_counts_zero_readings_as_missing():
    assert has_sufficient_pollutants({"pm25": 40, "no2": 30, "so2": 0}) is False


def test_minimum_pollutant_rule_needs_pm25_or_pm10():
    assert has_sufficient_pollutants({"no2": 3, "so2": 3, "o3": 3, "co": 3}) is False
    assert has_sufficient_pollutants({"pm10": 80, "no2": 30, "so2": 20}) is True


def test_minimum_pollutant_rule_counts_nh3_like_the_official_sheet():
    assert has_sufficient_pollutants({"pm25": 40, "nh3": 300, "no2": 30}) is True


def test_minimum_pollutant_rule_ignores_unsupported_names():
    assert has_sufficient_pollutants({"pm25": 40, "no2": 30, "benzene": 5}) is False


def test_minimum_pollutant_rule_accepts_a_bare_name_collection():
    assert has_sufficient_pollutants({"pm25", "no2", "so2"}) is True
    assert has_sufficient_pollutants({"no2", "so2", "o3"}) is False


def test_aqi_refuses_a_single_low_pollutant():
    """The low end bug: one pollutant averaged alone became the day's AQI."""
    with pytest.raises(InsufficientPollutantsError):
        calculate_aqi({"so2": 1.5})


def test_minimum_pollutant_rule_can_be_waived_for_a_fixed_pollutant_set():
    aqi, _ = calculate_aqi({"so2": 1.5}, enforce_minimum_pollutants=False)

    assert aqi == 2


# ---------------------------------------------------------------------------
# Aggregation and input handling
# ---------------------------------------------------------------------------


def test_aqi_is_the_maximum_sub_index_and_names_that_pollutant():
    aqi, dominant = calculate_aqi({"pm25": 90, "pm10": 50, "no2": 10})

    assert aqi == calculate_sub_index("pm25", 90)
    assert dominant == "pm25"


def test_aqi_detail_returns_one_sub_index_per_present_pollutant():
    dominant, details = calculate_aqi_detail(SIX)

    assert set(details) == set(SIX)
    assert details[dominant.pollutant] is dominant
    assert all(detail.extrapolated is False for detail in details.values())


def test_aqi_reports_extrapolation_from_any_contributing_pollutant():
    _, details = calculate_aqi_detail({**SIX, "pm25": 900.0})

    assert any(detail.extrapolated for detail in details.values())


def test_unsupported_pollutant_is_ignored_not_fatal():
    assert calculate_sub_index("lead", 10) is None


def test_pollutant_names_are_normalised():
    assert calculate_sub_index("PM25", 45) == calculate_sub_index("pm25", 45)
    assert calculate_sub_index(" pm25 ", 45) == calculate_sub_index("pm25", 45)


def test_negative_and_non_finite_concentrations_are_rejected():
    assert calculate_sub_index("pm25", -1) is None
    assert calculate_sub_index("pm25", float("nan")) is None
    assert calculate_sub_index("pm25", float("inf")) is None
    assert calculate_sub_index("pm25", None) is None


def test_aqi_requires_at_least_one_supported_pollutant():
    with pytest.raises(ValueError):
        calculate_aqi({"lead": 10})

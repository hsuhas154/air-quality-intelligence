import pytest

from air_quality_intelligence.analysis.aqi import (
    BREAKPOINTS,
    calculate_aqi,
    calculate_sub_index,
)


def test_known_cpcb_anchor_points():
    assert calculate_sub_index("pm25", 30) == 50
    assert calculate_sub_index("pm25", 60) == 100
    assert calculate_sub_index("pm10", 50) == 50
    assert calculate_sub_index("pm10", 100) == 100


def test_sub_index_has_no_gaps_between_breakpoint_bands():
    """Every positive concentration must yield a positive sub-index.

    A banded lookup with non-contiguous bands silently returns a fallback
    value for concentrations that fall between bands, which understates
    hazardous readings rather than failing loudly.
    """
    for pollutant in BREAKPOINTS:
        upper = BREAKPOINTS[pollutant][-2][1]
        step = upper / 500.0
        concentration = step
        while concentration <= upper:
            sub_index = calculate_sub_index(pollutant, concentration)
            assert sub_index is not None, (pollutant, concentration)
            assert sub_index > 0, (pollutant, concentration)
            concentration += step


def test_sub_index_is_monotonic_in_concentration():
    for pollutant in BREAKPOINTS:
        upper = BREAKPOINTS[pollutant][-2][1]
        previous = -1
        for index in range(0, 501):
            concentration = upper * index / 500.0
            sub_index = calculate_sub_index(pollutant, concentration)
            assert sub_index >= previous, (pollutant, concentration)
            previous = sub_index


def test_aqi_is_the_maximum_sub_index_and_names_that_pollutant():
    aqi, dominant = calculate_aqi({"pm25": 90, "pm10": 50})

    assert aqi == calculate_sub_index("pm25", 90)
    assert dominant == "pm25"


def test_aqi_uses_project_pollutant_keys():
    aqi, dominant = calculate_aqi(
        {"pm25": 24.4, "pm10": 77.3, "no2": 47.4, "so2": 28.0, "o3": 19.5, "co": 0.55}
    )

    assert dominant in {"pm25", "pm10", "no2", "so2", "o3", "co"}
    assert 0 < aqi <= 500


def test_unsupported_pollutant_is_ignored_not_fatal():
    assert calculate_sub_index("lead", 10) is None


def test_negative_concentration_is_rejected():
    assert calculate_sub_index("pm25", -1) is None


def test_aqi_requires_at_least_one_supported_pollutant():
    with pytest.raises(ValueError):
        calculate_aqi({"lead": 10})

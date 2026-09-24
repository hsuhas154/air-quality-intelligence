import pytest

from air_quality_intelligence.transform.units import (
    correct_reported_unit,
    normalize_concentration,
)
from air_quality_intelligence.transform.validation import (
    ImplausibleConcentrationError,
    check_plausible,
    is_plausible,
)


def test_ordinary_readings_are_plausible():
    assert is_plausible("pm25", 24.4)
    assert is_plausible("pm10", 77.3)
    assert is_plausible("no2", 47.4)
    assert is_plausible("co", 0.55)


def test_severe_but_genuine_pollution_is_not_rejected():
    assert is_plausible("pm25", 900.0)
    assert is_plausible("pm10", 1500.0)


def test_order_of_magnitude_unit_error_is_caught():
    assert not is_plausible("co", 0.00095)

    with pytest.raises(ImplausibleConcentrationError):
        check_plausible("co", 0.00095)


def test_openaq_co_ppb_label_is_corrected_to_ppm():
    assert correct_reported_unit("co", "ppb") == "ppm"


def test_other_pollutants_keep_their_reported_unit():
    assert correct_reported_unit("no2", "ppb") == "ppb"
    assert correct_reported_unit("so2", "ppb") == "ppb"
    assert correct_reported_unit("pm25", "µg/m³") == "µg/m³"


def test_corrected_co_normalizes_into_the_plausible_range():
    unit = correct_reported_unit("co", "ppb")
    value, canonical = normalize_concentration("co", 0.48, unit)

    assert canonical == "mg/m³"
    assert is_plausible("co", value)
